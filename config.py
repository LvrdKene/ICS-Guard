import os
from datetime import timedelta
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.environ.get("ICS_GUARD_SECRET_KEY", "development-only-change-me")
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8)
    DATA_FOLDER = BASE_DIR / "data"
    RUNTIME_FOLDER = BASE_DIR / "instance"
    MODEL_FOLDER = RUNTIME_FOLDER / "models"
    REPORT_FOLDER = RUNTIME_FOLDER / "reports"

    # Live detector: trained by IcsDatasetBuilder on synthetic, percentage-
    # scale command/process data in WaterTankSimulator's own feature space
    # (normal + startup/shutdown/maintenance + the four required attack
    # types). This is the model that actually scores every live Modbus/
    # simulator command -- see services/ics_guard.py's analyze().
    LIVE_MODEL_FILENAME = "live_lightgbm.joblib"
    LIVE_ANOMALY_MODEL_FILENAME = "live_isolation_forest.joblib"
    LIVE_METADATA_FILENAME = "live_model_metadata.json"
    LIVE_TRAINING_SNAPSHOT_FILENAME = "live_training_snapshot.csv"
    # 20,000 examples for each of five labelled classes = 100,000 records.
    LIVE_DATASET_ROWS_PER_CLASS = 20_000

    EVENT_LOG_FILENAME = "ics_events.jsonl"
    TRAINING_LOG_FILENAME = "training_runs.jsonl"
    DEFAULT_TEST_SIZE = 0.2
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_N_ESTIMATORS = 220
    # Lower than sklearn's usual default (0.1): verified against the actual
    # isolation_only_attack.py burst (21 rapid identical writes,
    # commands_last_10s>=18) that the deterministic burst check in analyze()
    # -- not the fitted Isolation Forest boundary -- is what catches that
    # attack at every contamination level tested; the forest's own boundary
    # at 0.04, and still at 0.01, was instead flagging realistic normal
    # traffic (e.g. maintenance_off right after maintenance_drain) whose
    # anomaly score was within noise of zero. 0.002 was the lowest level
    # tested and the first to clear every case in that realistic test.
    DEFAULT_IFOREST_CONTAMINATION = 0.002
    COMMAND_PORT = 502
    # Localhost avoids accidentally exposing the hackathon simulator to a network.
    MODBUS_PLANT_HOST = os.environ.get("ICS_GUARD_MODBUS_HOST", "127.0.0.1")
    MODBUS_PLANT_PORT = int(os.environ.get("ICS_GUARD_MODBUS_PORT", "5020"))
