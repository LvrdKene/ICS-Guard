from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

from config import Config
from services.ics_guard import IcsDatasetBuilder, IcsModelService, WaterTankSimulator
from services.modbus_plant import ModbusPlant

LIVE_DATASET_TAG = "ICS-Guard self-generated command dataset"


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    for folder in (app.config["DATA_FOLDER"], app.config["MODEL_FOLDER"], app.config["REPORT_FOLDER"]):
        Path(folder).mkdir(parents=True, exist_ok=True)
    simulator = WaterTankSimulator()

    def model_service_config(prefix: str) -> dict:
        """Build the live model service configuration from shared app settings."""
        return {
            "MODEL_FOLDER": app.config["MODEL_FOLDER"],
            "DATA_FOLDER": app.config["DATA_FOLDER"],
            "MODEL_FILENAME": app.config[f"{prefix}_MODEL_FILENAME"],
            "ANOMALY_MODEL_FILENAME": app.config[f"{prefix}_ANOMALY_MODEL_FILENAME"],
            "METADATA_FILENAME": app.config[f"{prefix}_METADATA_FILENAME"],
            "LIVE_TRAINING_DATASET_FILENAME": app.config[f"{prefix}_TRAINING_SNAPSHOT_FILENAME"],
            "DEFAULT_TEST_SIZE": app.config["DEFAULT_TEST_SIZE"],
            "DEFAULT_IFOREST_CONTAMINATION": app.config["DEFAULT_IFOREST_CONTAMINATION"],
        }

    # The live detector scores every real Modbus/simulator command (see
    # analyse_and_log below). It is trained by IcsDatasetBuilder on synthetic
    # data in the simulator's own percentage-scale feature space.
    live_model_service = IcsModelService(model_service_config("LIVE"), expected_dataset=LIVE_DATASET_TAG)
    event_log = Path(app.config["REPORT_FOLDER"]) / app.config["EVENT_LOG_FILENAME"]
    training_log = Path(app.config["REPORT_FOLDER"]) / app.config["TRAINING_LOG_FILENAME"]
    def append_training_log(kind: str, metadata: dict) -> None:
        with training_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"training_type": kind, **metadata}) + "\n")

    def csv_download(path: Path, filename: str, supports_status: bool = False) -> Response:
        if not path.exists():
            return Response("No records have been generated yet.\n", status=404, mimetype="text/plain")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            return Response("No records have been generated yet.\n", status=404, mimetype="text/plain")
        status = request.args.get("status", "all")
        date_from = request.args.get("date_from", "")
        date_to = request.args.get("date_to", "")
        search = request.args.get("search", "").strip().lower()
        if supports_status and status in {"alert", "normal", "review"}:
            rows = [row for row in rows if (
                (status == "alert" and (row.get("unsafe") or row.get("review_flag")))
                or (status == "normal" and not row.get("unsafe") and not row.get("review_flag"))
                or (status == "review" and row.get("review_flag"))
            )]
        if date_from or date_to:
            rows = [row for row in rows if (
                (not date_from or str(row.get("timestamp") or row.get("trained_at") or "")[:10] >= date_from)
                and (not date_to or str(row.get("timestamp") or row.get("trained_at") or "")[:10] <= date_to)
            )]
        if search:
            rows = [row for row in rows if search in " ".join(str(value) for value in row.values()).lower()]
        if not rows:
            return Response("No records match the selected filters.\n", status=404, mimetype="text/plain")
        return Response(pd.DataFrame(rows).to_csv(index=False), mimetype="text/csv", headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    def recent_events(limit: int = 40) -> list[dict]:
        if not event_log.exists():
            return []
        return [json.loads(line) for line in reversed(event_log.read_text(encoding="utf-8").splitlines()[-limit:]) if line.strip()]

    def analyse_and_log(events: list[dict]) -> list[dict]:
        analysed = [engineer_alert(live_model_service.analyze(event)) for event in events]
        with event_log.open("a", encoding="utf-8") as handle:
            for event in analysed:
                handle.write(json.dumps(event) + "\n")
        return analysed

    def engineer_alert(event: dict) -> dict:
        """Turn technical model output into statements an operator can act on."""
        statements = []
        explanation_prefix = "Possible context signal" if event.get("shap_scope") == "anomaly_context" else "Risk signal"
        for factor in event.get("shap_factors", []):
            if factor.get("direction") == "raises risk" and factor.get("plain"):
                statements.append(f"{explanation_prefix}: {factor['plain']}")
        if event.get("uncertainty"):
            statements.append(event["uncertainty"])
        if event.get("review_flag") and event.get("review_note") not in statements:
            statements.append(event["review_note"])
        event["engineer_explanation"] = list(dict.fromkeys(statements)) or event["explanation"]
        event["operator_summary"] = ("More plant context is needed before this packet can be judged safely."
                                     if event.get("process_context_available") is False else
                                     "Review before allowing this command: its timing or process context is outside the safe pattern."
                                     if event["unsafe"] else
                                     "A maintenance declaration immediately preceded this drain -- confirm it against a real work order."
                                     if event.get("review_flag") else "Continue monitoring: no safety rule or abnormal pattern was found.")
        return event

    def ingest_plant_command(observation: dict) -> None:
        """Normalise a real localhost Modbus write into a process-aware event."""
        event = simulator.issue_command(observation["command_name"], observation["command_value"], "controller", "NORMAL")
        event.update({"source_ip": observation.get("source_ip", event["source_ip"]), "destination_ip": observation.get("destination_ip", event["destination_ip"]), "live_modbus_plant": True})
        analyse_and_log([event])

    plant = ModbusPlant(ingest_plant_command, app.config["MODBUS_PLANT_HOST"], app.config["MODBUS_PLANT_PORT"], simulator.state)
    plant.start()
    def advance_plant() -> None:
        while True:
            time.sleep(1)
            simulator.advance(1)
    threading.Thread(target=advance_plant, name="ics-guard-plant-clock", daemon=True).start()

    @app.get("/")
    def dashboard():
        return render_template("dashboard.html", state=simulator.state(), live_model=live_model_service.metadata(), events=recent_events())

    @app.get("/api/status")
    def status():
        return jsonify({"state": simulator.state(), "live_model": live_model_service.metadata(), "plant": plant.status(), "events": recent_events()})

    @app.post("/api/control")
    def control():
        payload = request.get_json(silent=True) or {}
        action = payload.get("action")
        if action not in {"startup", "shutdown", "normal", "maintenance_on", "maintenance_off", "maintenance_drain", "open_valve", "close_valve", "start_pump", "stop_pump", "set_pump_speed", "set_valve_position", "set_tank_setpoint"}:
            return jsonify({"error": "Unknown tank-control action."}), 400
        activity = simulator.state()["maintenance_activity"]
        if activity == "shutdown" and action != "startup":
            return jsonify({"error": "The system is shut down. Send Start Up before using other controls."}), 409
        if activity != "shutdown" and action == "startup":
            return jsonify({"error": "The system is already started."}), 409
        value = payload.get("value")
        if action in {"set_pump_speed", "set_valve_position", "set_tank_setpoint"}:
            try:
                value = float(value)
            except (TypeError, ValueError):
                return jsonify({"error": "Enter a numeric percentage."}), 400
            if not 0 <= value <= 100:
                return jsonify({"error": "Percentage must be between 0 and 100."}), 400
        try:
            plant.execute_control(action, value)
            simulator.apply_operator_control(action, value)
        except (RuntimeError, ValueError) as error:
            return jsonify({"error": str(error)}), 503
        return jsonify({"state": simulator.state(), "events": recent_events(12), "plant": plant.status()})

    @app.post("/api/reset")
    def reset():
        simulator.reset()
        return jsonify({"state": simulator.state()})

    @app.post("/api/train/live")
    def train_live():
        """Train the live detector on IcsDatasetBuilder's simulator-native data.

        This is the model that actually scores every real Modbus/simulator
        command (see analyse_and_log). Generation is deterministic and uses
        the simulator's own command and process feature space.
        """
        try:
            frame = IcsDatasetBuilder().build(rows_per_class=app.config["LIVE_DATASET_ROWS_PER_CLASS"])
            metadata = live_model_service.train(frame, groups=None, dataset_metadata=None)
        except Exception as error:
            app.logger.exception("Live model training failed")
            return jsonify({"error": f"Model training failed: {error}"}), 500
        append_training_log("Live simulator detector", metadata)
        return jsonify({"model": metadata, "records": metadata["records"]})

    @app.get("/api/download/detections.csv")
    def download_detections():
        return csv_download(event_log, "ics_guard_detection_log.csv", supports_status=True)

    @app.get("/api/download/training-runs.csv")
    def download_training_runs():
        return csv_download(training_log, "ics_guard_training_runs.csv")

    @app.get("/api/download/live-training-dataset.csv")
    def download_live_training_dataset():
        dataset = Path(app.config["DATA_FOLDER"]) / app.config["LIVE_TRAINING_SNAPSHOT_FILENAME"]
        if not dataset.exists():
            return Response("Train the live simulator detector first.\n", status=404, mimetype="text/plain")
        return Response(dataset.read_bytes(), mimetype="text/csv", headers={"Content-Disposition": 'attachment; filename="ics_guard_live_training_dataset.csv"'})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
