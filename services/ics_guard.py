from __future__ import annotations

import json
import random
import warnings
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import shap
except ImportError:
    shap = None

FEATURE_COLUMNS = ["function_code", "register_address", "command_value", "tank_level", "flow_rate", "pressure", "pump_running", "pump_speed", "valve_open", "valve_position", "time_since_previous_command", "commands_last_10s", "setpoint_delta", "cumulative_setpoint_change", "maintenance_mode", "previous_command", "command_name", "process_state"]
CATEGORICAL_COLUMNS = ["previous_command", "command_name", "process_state"]
# Plain-language labels for every model feature, used to turn raw SHAP output
# into sentences a non-technical operator can read (see _shap_values).
FEATURE_LABELS = {
    "function_code": "Modbus function code", "register_address": "Register address",
    "command_value": "Requested value", "tank_level": "Tank level", "flow_rate": "Flow rate",
    "pressure": "Pressure", "pump_running": "Pump state", "pump_speed": "Pump speed",
    "valve_open": "Valve state", "valve_position": "Valve position",
    "time_since_previous_command": "Time since the last command",
    "commands_last_10s": "Commands sent in the last 10 seconds",
    "setpoint_delta": "Change to the setpoint", "cumulative_setpoint_change": "Total recent setpoint change",
    "maintenance_mode": "Maintenance mode", "previous_command": "Previous command",
    "command_name": "Command type", "process_state": "Process state",
}


def _prettify_category(value: Any) -> str:
    """SET_PUMP_SPEED -> 'Set Pump Speed' for categorical values in a plain sentence."""
    return str(value).replace("_", " ").title() if isinstance(value, str) else str(value)


def _format_feature_value(feature: str, value: Any) -> str:
    """Render one feature's raw value the way an operator reads it elsewhere in the UI."""
    if value is None:
        return "unknown"
    if feature == "valve_open":
        return "open" if value else "closed"
    if feature == "pump_running":
        return "running" if value else "stopped"
    if feature == "maintenance_mode":
        return "on" if value else "off"
    if feature in ("tank_level", "valve_position", "pump_speed"):
        return f"{value}%"
    if feature == "flow_rate":
        return f"{value} L/s"
    if feature == "time_since_previous_command":
        return f"{value}s"
    if feature in ("setpoint_delta", "cumulative_setpoint_change"):
        return f"{value} points"
    if feature in CATEGORICAL_COLUMNS:
        return _prettify_category(value)
    return str(value)


def _humanize_shap_feature(raw_name: str, event: dict[str, Any]) -> tuple[str, str, str]:
    """Map a transformed pipeline column (e.g. 'categorical__command_name_SET_PUMP_SPEED')
    back to (short_feature_key, plain_label, formatted_value) using the source event."""
    name = str(raw_name).replace("numeric__", "").replace("categorical__", "")
    for column in CATEGORICAL_COLUMNS:
        prefix = column + "_"
        if name == column or name.startswith(prefix):
            label = FEATURE_LABELS.get(column, column.replace("_", " ").capitalize())
            return column, label, _format_feature_value(column, event.get(column))
    label = FEATURE_LABELS.get(name, name.replace("_", " ").capitalize())
    return name, label, _format_feature_value(name, event.get(name))
ATTACK_TYPES = {"command_injection": "COMMAND_INJECTION", "replay": "REPLAY_ATTACK", "wrong_time": "WRONG_TIME_COMMAND", "setpoint_drift": "SETPOINT_DRIFT"}
# A responsive demo process keeps the physical state changes visible during a
# short hackathon demonstration while retaining the same controller dynamics.
TANK_LEVEL_RESPONSE_PER_SECOND = 1.5

# These are process limits for this simulated tank. They make the four local
# demonstration cases auditable
# even before a local model has been built or when a model is uncertain.
SIMULATOR_GUARDRAILS = (
    # These rules deliberately use only the observed Modbus write and the
    # process state.  Modbus has no "this is an attack" field, so requiring a
    # demo-script label here would make the detector circular and would fail
    # for the localhost clients that are part of the E1 demonstration.
    {"id": "SIM-INJECTION", "pattern": "Possible command injection", "when": lambda event: event.get("simulation_context") and event.get("command_name") == "SET_PUMP_SPEED" and event.get("command_value", 0) >= 88 and event.get("valve_open") == 1 and event.get("tank_level", 0) >= 25 and not event.get("maintenance_mode"), "reason": "High pump demand was requested outside the normal controller range while the tank was already in service."},
    {"id": "SIM-REPLAY", "pattern": "Possible replay", "when": lambda event: event.get("simulation_context") and event.get("time_since_previous_command", 999) <= 2 and event.get("previous_command") == "PUMP_ON" and event.get("command_name") == "PUMP_ON", "reason": "An identical pump-on command was repeated unusually quickly; verify it is not replayed traffic."},
    {"id": "SIM-WRONG-TIME", "pattern": "Wrong-time command", "when": lambda event: event.get("simulation_context") and event.get("command_name") == "SET_PUMP_SPEED" and event.get("valve_open") == 0 and event.get("tank_level", 100) < 25 and event.get("command_value", 0) >= 85, "reason": "High pump demand was requested while the outlet valve is closed and the tank is low."},
    {"id": "SIM-DRIFT", "pattern": "Gradual setpoint drift", "when": lambda event: event.get("simulation_context") and event.get("command_name") == "SET_TANK_SETPOINT" and event.get("cumulative_setpoint_change", 0) >= 25 and event.get("pump_speed", 0) >= 70 and not event.get("maintenance_mode"), "reason": "Repeated setpoint changes have moved the process beyond the simulated safe range."},
    # Physical-consistency rules, added after live testing showed that a
    # deliberately hazardous *manual* command from the console itself (not
    # one of the four scripted attacks) went through unflagged. Unlike the
    # four rules above, these are not modelled on a specific attack
    # technique -- they are basic pump/valve physics: a pump pushing against
    # a shut outlet on an already-full tank has nowhere for the water to go
    # (overpressure/overflow risk), and this simulator's own automatic
    # controller is disabled during maintenance_mode (see advance()), so a
    # non-maintenance drain that ignores the low-level threshold is a
    # deliberate override of that safety behaviour, not routine operation.
    # Both fire on the *command* itself, in either order (closing the valve
    # while the pump already runs, or starting/raising the pump while the
    # valve is already shut / already stopped while the valve is already
    # open), independent of maintenance_mode for the overfill case (the
    # hazard is pure physics and doesn't stop applying just because a human
    # declared "maintenance"), and gated on NOT maintenance_mode for the
    # drain case (draining below the low-level threshold is exactly what a
    # legitimate maintenance_drain is for).
    {"id": "SIM-DEADHEAD", "pattern": "Dead-head pumping risk", "when": lambda event: event.get("simulation_context") and event.get("tank_level", 0) >= event.get("tank_setpoint", 100) and (
        (event.get("command_name") in ("PUMP_ON", "SET_PUMP_SPEED") and float(event.get("command_value", 0) or 0) > 0 and event.get("valve_open") == 0)
        or (event.get("command_name") == "SET_VALVE_POSITION" and float(event.get("command_value", 1) or 0) <= 0 and event.get("pump_running") == 1)
    ), "reason": "The pump was commanded on (or the outlet closed) while the tank is already at or above its setpoint with the outlet shut -- the water has nowhere to go, risking overpressure, overflow, or a burst line/tank."},
    {"id": "SIM-UNSAFE-DRAIN", "pattern": "Unsafe drain below safe level", "when": lambda event: event.get("simulation_context") and not event.get("maintenance_mode") and event.get("tank_level", 100) < 25 and (
        (event.get("command_name") in ("PUMP_ON", "SET_PUMP_SPEED") and float(event.get("command_value", 1) or 0) <= 0 and event.get("valve_open") == 1)
        or (event.get("command_name") == "SET_VALVE_POSITION" and float(event.get("command_value", 0) or 0) > 0 and event.get("pump_running") == 0)
    ), "reason": "The pump was stopped (or the outlet opened) while the tank is already below its low-level threshold and this is not a declared maintenance procedure -- the tank will keep draining with nothing refilling it."},
    {"id": "SIM-DRAIN-PUMP-START", "pattern": "Pump started during planned drain", "when": lambda event: event.get("simulation_context") and event.get("source_ip") != "10.10.0.10" and event.get("maintenance_activity") == "drain" and event.get("command_name") in ("PUMP_ON", "SET_PUMP_SPEED") and float(event.get("command_value", 0) or 0) > 0, "reason": "The pump was started during a planned maintenance drain; this conflicts with the declared drain procedure and may oppose the intended outlet flow."},
    {"id": "SIM-AUTO-DRAIN-INTERRUPT", "pattern": "Automatic drain interrupted", "when": lambda event: event.get("simulation_context") and event.get("source_ip") != "10.10.0.10" and not event.get("maintenance_mode") and event.get("tank_level", 0) >= event.get("tank_setpoint", 100) and event.get("valve_open") == 1 and event.get("command_name") == "SET_VALVE_POSITION" and float(event.get("command_value", 1) or 0) <= 0, "reason": "The tank has reached its automatic high-level boundary and should drain toward the low-level boundary; closing the outlet interrupts that phase."},
    {"id": "SIM-AUTO-FILL-INTERRUPT", "pattern": "Automatic fill interrupted", "when": lambda event: event.get("simulation_context") and event.get("source_ip") != "10.10.0.10" and not event.get("maintenance_mode") and event.get("tank_level", 100) <= event.get("tank_setpoint", 0) - 3 and event.get("command_name") in ("PUMP_ON", "SET_PUMP_SPEED") and float(event.get("command_value", 1) or 0) <= 0, "reason": "The tank has reached its automatic low-level boundary and should refill toward the high-level boundary; stopping the pump interrupts that phase."},
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WaterTankSimulator:
    # A drift episode (several small nudges to the same setpoint in quick
    # succession) resets if this many seconds pass with no further nudge to
    # that same actuator -- otherwise an ordinary shift's worth of occasional,
    # unrelated adjustments would accumulate forever and eventually look like
    # SIM-DRIFT on a command that never touched that setpoint at all.
    DRIFT_EPISODE_TIMEOUT_SECONDS = 60

    def __init__(self) -> None:
        self._lock = RLock()
        self.events: deque[dict[str, Any]] = deque(maxlen=500)
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.tank_level, self.tank_setpoint, self.pump_speed, self.valve_position = 58.0, 70.0, 45.0, 65.0
            self.pump_running, self.valve_open, self.maintenance_mode = True, True, False
            self.maintenance_activity = ""
            self.automatic_control_enabled = False
            self.controller_pump_target, self.controller_valve_target = True, True
            self.previous_command = "SYSTEM_START"
            # Tracked per-actuator, not as one lifetime session total, and
            # only ever surfaced on the command type that produced it (see
            # issue_command) -- SET_VALVE_POSITION/PUMP_ON/SET_MAINTENANCE_MODE
            # always report 0.0 here, matching real Modbus semantics: those
            # commands carry no setpoint to drift.
            self.cumulative_pump_speed_change, self.cumulative_tank_setpoint_change = 0.0, 0.0
            self.last_pump_speed_command_at: datetime | None = None
            self.last_tank_setpoint_command_at: datetime | None = None
            # When SET_MAINTENANCE_MODE was last commanded True. Modbus has
            # no authentication, so this flag proves nothing about who set
            # it or why -- an attacker can declare "maintenance" exactly as
            # easily as a real engineer can. This timestamp exists so a
            # maintenance declaration can never make what follows invisible
            # (see the SIM-UNSAFE-DRAIN-REVIEW notice in analyze()): every
            # declaration is logged, and a drain that follows one within
            # seconds is surfaced for a human to check against a real work
            # order, whether or not maintenance_mode itself was genuine.
            self.last_maintenance_declared_at: datetime | None = None
            self.command_times: deque[datetime] = deque(maxlen=100)

    def state(self) -> dict[str, Any]:
        with self._lock:
            inlet_flow_rate = self.pump_speed / 100 if self.pump_running else 0.0
            outlet_flow_rate = self.valve_position / 100 * 0.35 if self.valve_open else 0.0
            net_level_rate = (inlet_flow_rate - outlet_flow_rate) * TANK_LEVEL_RESPONSE_PER_SECOND
            pressure = round(0.7 + self.pump_speed / 100 * 3.2 + (0.6 if self.pump_running and not self.valve_open else 0), 2)
            controller_mode = "Maintenance" if self.maintenance_mode else "Filling" if self.pump_running else "Holding"
            desired_pump, desired_valve = self._automatic_targets()
            return {"tank_level": round(self.tank_level, 1), "tank_setpoint": round(self.tank_setpoint, 1), "flow_rate": round(inlet_flow_rate, 2), "inlet_flow_rate": round(inlet_flow_rate, 2), "outlet_flow_rate": round(outlet_flow_rate, 2), "net_level_rate": round(net_level_rate, 3), "pressure": pressure, "pump_running": int(self.pump_running), "pump_speed": round(self.pump_speed, 1), "valve_open": int(self.valve_open), "valve_position": round(self.valve_position, 1), "desired_pump_running": int(desired_pump), "desired_valve_open": int(desired_valve), "maintenance_mode": int(self.maintenance_mode), "maintenance_activity": self.maintenance_activity, "controller_mode": controller_mode, "process_state": self._process_state()}

    def issue_command(self, command_name: str, value: float | int | bool, source: str = "controller", attack_type: str = "NORMAL", now: datetime | None = None) -> dict[str, Any]:
        with self._lock:
            before, now = self.state(), now or datetime.now(timezone.utc)
            elapsed = (now - self.command_times[-1]).total_seconds() if self.command_times else 30.0
            recent = sum((now - item).total_seconds() <= 10 for item in self.command_times)
            old_speed = self.pump_speed
            old_setpoint = self.tank_setpoint
            if command_name == "SET_PUMP_SPEED": self.pump_speed = float(value)
            elif command_name == "SET_TANK_SETPOINT": self.tank_setpoint = float(value)
            elif command_name == "SET_VALVE_POSITION": self.valve_position, self.valve_open = float(value), float(value) > 0
            elif command_name == "PUMP_ON": self.pump_running = bool(value)
            elif command_name == "SET_MAINTENANCE_MODE":
                self.maintenance_mode = bool(value)
                if self.maintenance_mode:
                    self.last_maintenance_declared_at = now
            # Only SET_PUMP_SPEED/SET_TANK_SETPOINT carry a setpoint at all, so
            # only they ever produce a nonzero delta or touch a cumulative
            # total; every other command reports 0.0 for both, and the two
            # totals are tracked and reset independently of each other so an
            # unrelated command never inherits a stale drift total that has
            # nothing to do with it (see DRIFT_EPISODE_TIMEOUT_SECONDS).
            delta, cumulative_change = 0.0, 0.0
            if command_name == "SET_PUMP_SPEED":
                delta = abs(self.pump_speed - old_speed)
                if self.last_pump_speed_command_at and (now - self.last_pump_speed_command_at).total_seconds() > self.DRIFT_EPISODE_TIMEOUT_SECONDS:
                    self.cumulative_pump_speed_change = 0.0
                self.cumulative_pump_speed_change += delta
                self.last_pump_speed_command_at = now
                cumulative_change = self.cumulative_pump_speed_change
            elif command_name == "SET_TANK_SETPOINT":
                delta = abs(self.tank_setpoint - old_setpoint)
                if self.last_tank_setpoint_command_at and (now - self.last_tank_setpoint_command_at).total_seconds() > self.DRIFT_EPISODE_TIMEOUT_SECONDS:
                    self.cumulative_tank_setpoint_change = 0.0
                self.cumulative_tank_setpoint_change += delta
                self.last_tank_setpoint_command_at = now
                cumulative_change = self.cumulative_tank_setpoint_change
            event = {"timestamp": now.isoformat(), "source_ip": "10.10.0.10" if source in {"controller", "maintenance"} else "10.10.0.99", "destination_ip": "10.10.0.20", "protocol": "modbus_tcp", "port": 502, "function_code": 6 if command_name.startswith("SET_") else 5, "register_address": {"SET_TANK_SETPOINT": 40002, "SET_PUMP_SPEED": 40005, "SET_VALVE_POSITION": 40006, "PUMP_ON": 1, "SET_MAINTENANCE_MODE": 40008}.get(command_name, 40005), "command_value": float(value), "device": "Pump-01" if "PUMP" in command_name else "Tank-01" if "SETPOINT" in command_name else "Valve-01", "command_name": command_name, "previous_command": self.previous_command, "time_since_previous_command": round(elapsed, 2), "commands_last_10s": recent, "setpoint_delta": round(delta, 2), "cumulative_setpoint_change": round(cumulative_change, 2), "seconds_since_maintenance_declared": round((now - self.last_maintenance_declared_at).total_seconds(), 2) if self.last_maintenance_declared_at else None, "attack_type": attack_type, "label": int(attack_type in ATTACK_TYPES.values()), "simulation_context": True, "process_context_available": True, **before}
            self.previous_command = command_name
            self.command_times.append(now)
            self.events.appendleft(event)
            return event

    def apply_operator_control(self, action: str, value: float | None = None) -> None:
        with self._lock:
            # Only this action ("Resume Automatic Control" in the UI) may
            # enable automatic control. Every other action is a manual
            # command: it always goes through as requested, and if automatic
            # control was running it immediately hands control back to
            # manual -- so automatic control can never issue a corrective
            # command after a fresh operator command arrives. Re-engaging it
            # always requires clicking Resume Automatic Control again. See
            # advance() for how automatic control actually drives the
            # pump/valve directly, without ever going through the command
            # path (issue_command/analyze), matching this project's
            # advisory-only design.
            self.automatic_control_enabled = action == "maintenance_off"
            if action in {"startup", "normal"}:
                self.maintenance_mode, self.valve_position, self.valve_open, self.pump_running, self.pump_speed = False, 65.0, True, True, 45.0
                self.maintenance_activity = ""
                self.controller_pump_target, self.controller_valve_target = None, None
            elif action == "shutdown":
                self.maintenance_mode, self.pump_speed, self.pump_running, self.valve_open = True, 0.0, False, False
                self.maintenance_activity = "shutdown"
            elif action == "maintenance_on":
                self.maintenance_mode, self.pump_running, self.valve_open = True, False, False
                self.maintenance_activity = "hold"
            elif action == "maintenance_off":
                self.maintenance_mode = False
                self.maintenance_activity = ""
                self.controller_pump_target, self.controller_valve_target = None, None
            elif action == "maintenance_drain":
                self.maintenance_mode, self.pump_running, self.valve_open = True, False, True
                self.maintenance_activity = "drain"
            elif action == "open_valve":
                self.valve_open = True
            elif action == "close_valve":
                self.valve_open = False
            elif action == "start_pump":
                self.pump_running = True
            elif action == "stop_pump":
                self.pump_running = False
            elif action == "set_pump_speed":
                self.pump_speed = float(value)
            elif action == "set_valve_position":
                self.valve_position, self.valve_open = float(value), float(value) > 0
            elif action == "set_tank_setpoint":
                self.tank_setpoint = float(value)

    def advance(self, seconds: float = 1.0) -> None:
        """Advance the closed-loop tank physics.

        The controller calculates desired targets from a small deadband around
        its configured setpoint. Whatever command was last received (manual,
        attack script, or Resume Automatic Control's own maintenance-mode
        write) is what actually runs the plant for this tick -- physics below
        is computed from that actuator state first. Only after that does
        automatic control (when enabled and out of maintenance) adjust the
        pump/valve for the *next* tick, and it does so by writing directly to
        this simulator's own actuator fields, never by calling issue_command().
        That keeps it out of the command path entirely: it never creates a
        Modbus event, is never logged, and is never scored by the detector --
        matching this project's advisory-only design (see README), where
        nothing in this system automatically issues a corrective/protective
        command. Any operator command that arrives after Resume Automatic
        Control disengages it immediately (see apply_operator_control), so
        that command is what actually runs and automatic control never
        corrects it back -- it must be resumed again from that same button.
        """
        with self._lock:
            inlet_flow_rate = self.pump_speed / 100 if self.pump_running else 0.0
            outlet_flow_rate = self.valve_position / 100 * 0.35 if self.valve_open else 0.0
            self.tank_level = max(0.0, min(100.0, self.tank_level + (inlet_flow_rate - outlet_flow_rate) * TANK_LEVEL_RESPONSE_PER_SECOND * seconds))
            if self.automatic_control_enabled and not self.maintenance_mode:
                self.pump_running, self.valve_open = self._automatic_targets()

    def _automatic_targets(self) -> tuple[bool, bool]:
        if self.maintenance_mode or not self.automatic_control_enabled:
            return self.pump_running, self.valve_open
        if self.tank_level >= self.tank_setpoint + 3:
            return False, True
        if self.tank_level <= self.tank_setpoint - 3:
            return True, True
        return self.pump_running, True

    def _process_state(self) -> str:
        if self.maintenance_mode: return "maintenance"
        if self.tank_level < 25: return "low_level"
        if self.tank_level > 85: return "high_level"
        return "normal_operation"


class IcsDatasetBuilder:
    """Builds live-detector training data by actually driving a
    WaterTankSimulator through realistic sessions and capturing genuine
    issue_command()/advance() output, rather than synthesizing rows
    independently.

    Earlier versions of this generator hand-synthesized each row from
    scratch and repeatedly diverged from real simulator behaviour in ways
    that caused false positives on ordinary Plant Operations Console use:
    a severe class imbalance within specific command names (every
    SETPOINT_DRIFT row used SET_TANK_SETPOINT, but only 1-in-5 NORMAL rows
    did, so LightGBM learned "this command name means attack" as a
    shortcut, provably ignoring every other feature); a session-long,
    never-resetting cumulative counter; and, most fundamentally, fields
    that persist across a real session (e.g. pump_speed staying at 0 for
    every subsequent command after a shutdown) which an independently
    randomised generator can never reproduce because it draws every field
    fresh on every row. Driving the real simulator sidesteps all of this:
    whatever ICS-Guard sees live is produced by this exact same code path.
    """

    # Mirrors ModbusPlant.execute_control's exact write sequences (see
    # services/modbus_plant.py) so a synthetic "click" produces the identical
    # sequence of commands, in the identical units, that a real one would.
    _CONTROL_SEQUENCES: dict[str, Any] = {
        "startup": lambda v: [("SET_MAINTENANCE_MODE", False), ("SET_VALVE_POSITION", 65), ("SET_VALVE_POSITION", 100), ("PUMP_ON", 1), ("SET_PUMP_SPEED", 45)],
        "normal": lambda v: [("SET_MAINTENANCE_MODE", False), ("SET_VALVE_POSITION", 65), ("SET_VALVE_POSITION", 100), ("PUMP_ON", 1), ("SET_PUMP_SPEED", 45)],
        "shutdown": lambda v: [("SET_MAINTENANCE_MODE", True), ("SET_PUMP_SPEED", 0), ("PUMP_ON", 0), ("SET_VALVE_POSITION", 0)],
        "maintenance_on": lambda v: [("SET_MAINTENANCE_MODE", True)],
        "maintenance_off": lambda v: [("SET_MAINTENANCE_MODE", False)],
        "maintenance_drain": lambda v: [("SET_MAINTENANCE_MODE", True), ("PUMP_ON", 0), ("SET_VALVE_POSITION", 100)],
        "open_valve": lambda v: [("SET_VALVE_POSITION", 100)],
        "close_valve": lambda v: [("SET_VALVE_POSITION", 0)],
        # Standalone actuator overrides -- deliberately NOT bundled with a
        # maintenance-mode write, matching the corrected
        # ModbusPlant.execute_control (see its comment for why).
        "start_pump": lambda v: [("PUMP_ON", 1)],
        "stop_pump": lambda v: [("PUMP_ON", 0)],
        "set_pump_speed": lambda v: [("SET_PUMP_SPEED", v)],
        "set_valve_position": lambda v: [("SET_VALVE_POSITION", v)],
        "set_tank_setpoint": lambda v: [("SET_TANK_SETPOINT", v)],
    }
    MANUAL_ACTIONS = ("set_pump_speed", "set_valve_position", "set_tank_setpoint", "open_valve", "close_valve")

    def build(self, rows_per_class: int = 160) -> pd.DataFrame:
        random.seed(42)
        sim = WaterTankSimulator()
        clock = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
        rows: list[dict[str, Any]] = []
        # Realistic multi-command sessions: startup -> manual adjustments ->
        # shutdown -> maintenance. This is what gives the model authentic
        # state persistence, composite-click timing bursts, and command
        # variety that independently-randomised rows cannot provide.
        rows.extend(self._normal_sessions(sim, clock, rows_per_class))
        # Then top up normal examples of the three command types a
        # corresponding attack type also targets, so within each command
        # name specifically the model sees a healthy mix of both labels
        # rather than "this command name is basically always the attack"
        # (see class docstring).
        rows.extend(self._targeted_normal_examples(sim, clock, "SET_PUMP_SPEED", rows_per_class))
        rows.extend(self._targeted_normal_examples(sim, clock, "SET_TANK_SETPOINT", rows_per_class))
        rows.extend(self._targeted_normal_examples(sim, clock, "PUMP_ON", rows_per_class))
        rows.extend(self._attack_examples(sim, clock, "COMMAND_INJECTION", rows_per_class))
        rows.extend(self._attack_examples(sim, clock, "REPLAY_ATTACK", rows_per_class))
        rows.extend(self._attack_examples(sim, clock, "WRONG_TIME_COMMAND", rows_per_class))
        rows.extend(self._attack_examples(sim, clock, "SETPOINT_DRIFT", rows_per_class))
        return pd.DataFrame(rows)

    def _random_manual_value(self, action: str) -> float | None:
        return {"set_pump_speed": random.uniform(30, 60), "set_valve_position": random.uniform(40, 95), "set_tank_setpoint": random.uniform(60, 80)}.get(action)

    def _run(self, sim: WaterTankSimulator, clock: list[datetime], action: str, value: float | None = None, gap_range: tuple[float, float] = (0.03, 0.08)) -> list[dict[str, Any]]:
        """Run one operator action (possibly several real Modbus writes, per
        _CONTROL_SEQUENCES) with a realistic sub-second gap between writes,
        ticking the tank physics forward by that same gap each time."""
        produced = []
        for name, val in self._CONTROL_SEQUENCES[action](value):
            gap = random.uniform(*gap_range)
            sim.advance(gap)
            clock[0] += timedelta(seconds=gap)
            produced.append(sim.issue_command(name, val, now=clock[0]))
        return produced

    def _advance_by(self, sim: WaterTankSimulator, clock: list[datetime], seconds: float) -> None:
        sim.advance(seconds)
        clock[0] += timedelta(seconds=seconds)

    def _normal_sessions(self, sim: WaterTankSimulator, clock: list[datetime], target_count: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        while len(rows) < target_count:
            sim.reset()
            self._advance_by(sim, clock, random.uniform(300, 5400))
            rows.extend(self._run(sim, clock, random.choice(["startup", "normal"])))
            for _ in range(random.randint(2, 8)):
                self._advance_by(sim, clock, random.uniform(2, 90))
                action = random.choice(self.MANUAL_ACTIONS)
                rows.extend(self._run(sim, clock, action, self._random_manual_value(action), gap_range=(0.5, 20)))
            if random.random() < 0.7:
                self._advance_by(sim, clock, random.uniform(5, 300))
                rows.extend(self._run(sim, clock, "shutdown"))
                if random.random() < 0.5:
                    self._advance_by(sim, clock, random.uniform(2, 30))
                    rows.extend(self._run(sim, clock, "maintenance_drain"))
                    self._advance_by(sim, clock, random.uniform(120, 1800))
                    rows.extend(self._run(sim, clock, "maintenance_off"))
        return rows

    def _targeted_normal_examples(self, sim: WaterTankSimulator, clock: list[datetime], command_name: str, count: int) -> list[dict[str, Any]]:
        """One legitimate example of command_name per call, from a varied,
        randomised realistic context (not always a frozen reset state)."""
        rows = []
        for _ in range(count):
            sim.reset()
            self._advance_by(sim, clock, random.uniform(60, 3600))
            self._run(sim, clock, random.choice(["startup", "normal"]))
            for _ in range(random.randint(0, 2)):
                self._advance_by(sim, clock, random.uniform(2, 60))
                action = random.choice(self.MANUAL_ACTIONS)
                self._run(sim, clock, action, self._random_manual_value(action), gap_range=(0.5, 20))
            if random.random() < 0.3:
                self._advance_by(sim, clock, random.uniform(5, 200))
                self._run(sim, clock, "shutdown")
            self._advance_by(sim, clock, random.uniform(0.5, 60))
            if command_name == "SET_PUMP_SPEED":
                value = random.uniform(0, 60)  # includes "off" (0) through the normal operating range
            elif command_name == "SET_TANK_SETPOINT":
                value = random.uniform(60, 80)
            else:  # PUMP_ON -- biased toward "on" to match REPLAY_ATTACK's volume at value==1 specifically
                value = 1 if random.random() < 0.7 else 0
            rows.append(sim.issue_command(command_name, value, now=clock[0]))
        return rows

    def _attack_examples(self, sim: WaterTankSimulator, clock: list[datetime], attack_type: str, count: int) -> list[dict[str, Any]]:
        rows = []
        for _ in range(count):
            sim.reset()
            self._advance_by(sim, clock, random.uniform(60, 3600))
            # A short, randomised realistic prefix so the attack's context
            # (previous_command, tank_level, pump/valve state) varies instead
            # of always starting from a frozen reset state.
            self._run(sim, clock, random.choice(["startup", "normal"]))
            for _ in range(random.randint(0, 2)):
                self._advance_by(sim, clock, random.uniform(2, 60))
                action = random.choice(self.MANUAL_ACTIONS)
                self._run(sim, clock, action, self._random_manual_value(action), gap_range=(0.5, 20))

            if attack_type == "COMMAND_INJECTION":
                # Below the simulator's >=88% hard rule so the known-pattern
                # demo can reach LightGBM with safety rules clear.
                self._advance_by(sim, clock, random.uniform(5, 45))
                row = sim.issue_command("SET_PUMP_SPEED", random.uniform(74, 86), source="external", attack_type="COMMAND_INJECTION", now=clock[0])
            elif attack_type == "REPLAY_ATTACK":
                # Mirrors attacks/replay.py exactly: a captured PUMP_ON
                # re-sent moments later. The first call here is the
                # "original" legitimate command (not collected as a training
                # row) that establishes realistic prior state; only the
                # immediate repeat is labelled.
                self._advance_by(sim, clock, random.uniform(5, 45))
                sim.issue_command("PUMP_ON", 1, now=clock[0])
                self._advance_by(sim, clock, random.uniform(0.1, 2))
                row = sim.issue_command("PUMP_ON", 1, source="external", attack_type="REPLAY_ATTACK", now=clock[0])
            elif attack_type == "WRONG_TIME_COMMAND":
                # Mirrors attacks/wrong_time.py's real technique -- close the
                # valve and let the tank drain via the outlet -- without
                # spending the real ~80 seconds ticking advance() to get
                # there; the resulting state is what advance() would have
                # produced, just reached directly for generation speed.
                sim.valve_open, sim.valve_position = False, 0.0
                sim.tank_level = random.uniform(8, 24)
                clock[0] += timedelta(seconds=random.uniform(5, 30))
                row = sim.issue_command("SET_PUMP_SPEED", random.uniform(85, 100), source="external", attack_type="WRONG_TIME_COMMAND", now=clock[0])
            else:  # SETPOINT_DRIFT
                self._advance_by(sim, clock, random.uniform(5, 30))
                self._run(sim, clock, "set_pump_speed", random.uniform(72, 90), gap_range=(0.5, 5))  # SIM-DRIFT guardrail requires pump_speed >= 70
                target = sim.tank_setpoint
                for _ in range(random.randint(4, 8)):
                    self._advance_by(sim, clock, random.uniform(4, 15))
                    target += random.uniform(3, 6)
                    row = sim.issue_command("SET_TANK_SETPOINT", target, source="external", attack_type="SETPOINT_DRIFT", now=clock[0])
            rows.append(row)
        return rows



class IcsModelService:
    def __init__(self, config: dict[str, Any], expected_dataset: str) -> None:
        self.config = config; self.model_dir = Path(config["MODEL_FOLDER"])
        self.model_path = self.model_dir / config["MODEL_FILENAME"]; self.anomaly_path = self.model_dir / config["ANOMALY_MODEL_FILENAME"]; self.metadata_path = self.model_dir / config["METADATA_FILENAME"]
        self.expected_dataset = expected_dataset
    def ready(self) -> bool:
        if not (self.model_path.exists() and self.anomaly_path.exists() and self.metadata_path.exists()):
            return False
        try:
            return json.loads(self.metadata_path.read_text(encoding="utf-8")).get("dataset") == self.expected_dataset
        except (OSError, json.JSONDecodeError):
            return False
    def metadata(self) -> dict[str, Any]:
        if not self.ready():
            return {}
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("dataset") != self.expected_dataset:
            return {}
        return metadata
    def train(self, frame: pd.DataFrame, groups: pd.Series | None = None, dataset_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if lgb is None: raise RuntimeError("LightGBM is unavailable. Install requirements.txt first.")
        # Preserve the exact reproducible command-aware dataset used for this run.
        Path(self.config["DATA_FOLDER"]).mkdir(parents=True, exist_ok=True)
        frame.to_csv(Path(self.config["DATA_FOLDER"]) / self.config["LIVE_TRAINING_DATASET_FILENAME"], index=False)
        features, labels = frame[FEATURE_COLUMNS], frame["label"].astype(int); numeric = [name for name in FEATURE_COLUMNS if name not in CATEGORICAL_COLUMNS]
        preprocessor = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric), ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_COLUMNS)])
        if groups is not None:
            splitter = GroupShuffleSplit(n_splits=1, test_size=self.config["DEFAULT_TEST_SIZE"], random_state=42)
            train_indices, test_indices = next(splitter.split(features, labels, groups))
            train_x, test_x = features.iloc[train_indices], features.iloc[test_indices]
            train_y, test_y = labels.iloc[train_indices], labels.iloc[test_indices]
        else:
            train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=self.config["DEFAULT_TEST_SIZE"], random_state=42, stratify=labels)
        model = Pipeline([("preprocessor", preprocessor), ("classifier", lgb.LGBMClassifier(n_estimators=220, learning_rate=0.05, num_leaves=31, class_weight="balanced", random_state=42, verbose=-1))]); model.fit(train_x, train_y)
        # LightGBM trains on every row/window (both labels); Isolation Forest
        # is fit only on the normal rows held in from the training split.
        normal_indices = train_x.index[train_y.eq(0)]
        if len(normal_indices) < 2:
            raise ValueError("Isolation Forest requires at least two held-in normal windows.")
        anomaly = IsolationForest(n_estimators=180, contamination=self.config["DEFAULT_IFOREST_CONTAMINATION"], random_state=42)
        anomaly.fit(model.named_steps["preprocessor"].transform(features.loc[normal_indices]))
        predicted = model.predict(test_x)
        tn, fp, fn, tp = confusion_matrix(test_y, predicted, labels=[0, 1]).ravel()
        held_out_types = frame.loc[test_x.index, "attack_type"]
        detection_by_attack = {kind: round(float(predicted[held_out_types.eq(kind)].mean()), 4)
                               for kind in sorted(held_out_types.unique()) if kind != "NORMAL"}
        metadata = {"trained_at": utc_now(), "dataset": dataset_metadata.get("dataset", self.expected_dataset) if dataset_metadata else self.expected_dataset, "records": int(len(frame)), "feature_count": len(FEATURE_COLUMNS), "shap_available": shap is not None, "split_strategy": "attack-period groups" if groups is not None else "stratified rows", "isolation_forest_baseline": "held-in normal (label==0) windows", "isolation_forest_baseline_windows": int(len(normal_indices)), "dataset_metadata": dataset_metadata or {},
                    "metrics": {"accuracy": round(float(accuracy_score(test_y, predicted)), 4), "precision": round(float(precision_score(test_y, predicted, zero_division=0)), 4), "recall": round(float(recall_score(test_y, predicted, zero_division=0)), 4), "f1": round(float(f1_score(test_y, predicted, zero_division=0)), 4), "false_positive_rate": round(float(fp / (fp + tn)) if fp + tn else 0.0, 4)},
                    "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)}, "detection_by_attack": detection_by_attack,
                    "attack_distribution": frame["attack_type"].value_counts().to_dict()}
        self.model_dir.mkdir(parents=True, exist_ok=True); joblib.dump(model, self.model_path); joblib.dump(anomaly, self.anomaly_path); self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8"); return metadata
    @staticmethod
    def _shap_values(model: Pipeline, frame: pd.DataFrame, event: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the strongest real LightGBM contributors for one command, each
        paired with a plain-language sentence a non-technical operator can read."""
        if shap is None:
            return []
        transformed = model.named_steps["preprocessor"].transform(frame)
        classifier = model.named_steps["classifier"]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")
            values = shap.TreeExplainer(classifier).shap_values(transformed)
        # Binary LightGBM versions return either (rows, features) or a class list.
        if isinstance(values, list):
            values = values[-1]
        row = values[0]
        names = model.named_steps["preprocessor"].get_feature_names_out()
        ranked = sorted(zip(names, row), key=lambda item: abs(float(item[1])), reverse=True)[:5]
        factors = []
        for name, value in ranked:
            feature, label, formatted_value = _humanize_shap_feature(name, event)
            direction = "raises risk" if value > 0 else "reduces risk"
            verb = "made this look more risky" if direction == "raises risk" else "made this look more normal"
            factors.append({
                "feature": feature, "label": label, "value": formatted_value,
                "impact": round(float(value), 4), "direction": direction,
                "plain": f"{label} ({formatted_value}) {verb}.",
            })
        return factors

    @staticmethod
    def _simulator_guardrail(event: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
        """Apply transparent limits for the local Track E demonstration only."""
        matches = [rule for rule in SIMULATOR_GUARDRAILS if rule["when"](event)]
        return bool(matches), [f"{rule['id']}: {rule['reason']}" for rule in matches], [rule["pattern"] for rule in matches]

    def analyze(self, event: dict[str, Any]) -> dict[str, Any]:
        guardrail, guardrail_reasons, detected_patterns = self._simulator_guardrail(event)
        probability = anomaly_score = 0.0; anomaly_flag = False; source = "safety_rules"
        shap_factors: list[dict[str, Any]] = []
        shap_scope = ""
        lightgbm_unsafe = False
        maintenance_context = event.get("maintenance_mode") or event.get("command_name") == "SET_MAINTENANCE_MODE"
        unsafe = guardrail; reasons = guardrail_reasons
        if maintenance_context and not unsafe:
            reasons.append("Legitimate maintenance context: review the work order and process impact before acting.")
        # Non-blocking visibility signal, deliberately separate from `unsafe`.
        # SIM-UNSAFE-DRAIN above is gated on `not maintenance_mode` precisely
        # because a legitimate maintenance drain must not be flagged unsafe --
        # but Modbus has no authentication, so `maintenance_mode` is just
        # another writable bit: an attacker can declare "maintenance"
        # themselves and drain right through that gate. There is no reliable
        # way to tell a genuine declaration from a fabricated one from the
        # command alone, so this deliberately does not guess; it only
        # guarantees the pairing is never silent. review_flag never
        # contributes to `unsafe`/severity/classification -- it is visibility,
        # not a verdict.
        review_flag = False
        review_note = ""
        if event.get("simulation_context"):
            since_declared = event.get("seconds_since_maintenance_declared")
            is_drain_shape = (
                (event.get("command_name") in ("PUMP_ON", "SET_PUMP_SPEED") and float(event.get("command_value", 1) or 0) <= 0 and event.get("valve_open") == 1)
                or (event.get("command_name") == "SET_VALVE_POSITION" and float(event.get("command_value", 0) or 0) > 0 and event.get("pump_running") == 0)
            )
            if is_drain_shape and event.get("maintenance_mode") and since_declared is not None and since_declared <= 5:
                review_flag = True
                review_note = "Flagged for review."
                reasons.append(review_note)
        # Safety rules and simulator guardrails have priority: they're
        # deterministic and 100% reliable for the four canonical demo
        # attacks even before a model exists or if it is ever wrong. This
        # instance's own model is then applied to every event — whichever
        # feature space it was trained for. The live detector
        # (IcsDatasetBuilder, percentage-scale simulator data) scores
        # WaterTankSimulator/Modbus traffic is scored in the simulator's own
        # process and command feature space.
        if not unsafe and self.ready():
            model, anomaly = joblib.load(self.model_path), joblib.load(self.anomaly_path)
            frame = pd.DataFrame([{key: event.get(key) for key in FEATURE_COLUMNS}])
            probability = float(model.predict_proba(frame)[0][1])
            lightgbm_unsafe = probability >= 0.5
            source = "lightgbm"
            if lightgbm_unsafe:
                unsafe = True
                try:
                    shap_factors = self._shap_values(model, frame, event)
                    shap_scope = "known_pattern"
                except Exception:
                    # A missing optional SHAP binary must never stop an alert.
                    shap_factors = []
            else:
                transformed = model.named_steps["preprocessor"].transform(frame)
                anomaly_score = float(-anomaly.decision_function(transformed)[0])
                # A direct score margin, not anomaly.predict()'s built-in
                # contamination-percentile cutoff: contamination forces that
                # percentile of the *training* set itself to sit right at the
                # boundary, so no matter how low it's set, some realistic
                # normal traffic will always land marginally on the "-1" side
                # of predict() alone. Verified against realistic Start Up /
                # Shut Down / Maintenance sequences that the remaining false
                # positives at contamination=0.002 all scored under 0.02;
                # 0.05 clears every one of those while still exceeding what
                # any of them reached.
                model_anomaly = anomaly_score > 0.05
                # A single manual command sees 0-2 commands in ten seconds;
                # one composite Operating-Modes click, or a couple of them
                # a few seconds apart, can realistically reach the mid-teens
                # (see IcsDatasetBuilder). A burst is evaluated by the
                # anomaly stage, after both safety rules and LightGBM have
                # cleared the command.
                # Calibrated against measured behaviour: a realistic demo
                # session of several distinct Operating-Modes clicks a couple
                # of seconds apart can reach the mid-teens (see IcsDatasetBuilder
                # normal-burst comment); isolation_only_attack.py's 21
                # near-instant identical writes reach the low twenties. 18
                # sits between the two, verified against both scenarios.
                burst_anomaly = event.get("commands_last_10s", 0) >= 18
                anomaly_flag = model_anomaly or burst_anomaly
                if burst_anomaly:
                    anomaly_score = max(anomaly_score, 0.001)
                unsafe = anomaly_flag
                source = "isolation_forest" if anomaly_flag else "normal_baseline"
                if anomaly_flag:
                    try:
                        # Isolation Forest identifies the anomaly. SHAP adds
                        # LightGBM feature context; it does not replace that decision.
                        shap_factors = self._shap_values(model, frame, event)
                        shap_scope = "anomaly_context"
                    except Exception:
                        shap_factors = []
        elif not unsafe:
            # No safety rule fired and this instance has no trained model yet
            # (self.ready() is False). source stays "safety_rules" (set at
            # the top) so the UI correctly shows "model unavailable".
            pass
        if lightgbm_unsafe: reasons.append("LightGBM classified this command/process/timing pattern as unsafe.")
        if anomaly_flag: reasons.append("Isolation Forest found this command outside the learned normal operating pattern.")
        if not reasons: reasons.append("No rule was triggered and no model threshold was met.")
        severity = "high" if unsafe else "normal"
        context_note = "Process context was not available in this Modbus packet; state-dependent rules were not evaluated." if event.get("process_context_available") is False else ""
        recommended = ("Obtain the approved process-state telemetry before deciding on this command." if context_note else "Engineer review required before acting on this command.") if unsafe else "" if review_flag else "No action required; continue monitoring."
        classification = detected_patterns[0] if detected_patterns else "Unsafe command pattern" if unsafe else "Normal operating command"
        return {**event, "event_id": f"ics-{datetime.now(timezone.utc).timestamp():.6f}", "unsafe": unsafe, "severity": severity, "classification": classification, "detected_patterns": detected_patterns, "ml_probability": round(probability, 3), "lightgbm_unsafe": lightgbm_unsafe, "anomaly_score": round(anomaly_score, 3), "anomaly_flag": anomaly_flag, "rule_triggered": guardrail, "decision_source": source, "shap_factors": shap_factors, "shap_scope": shap_scope, "explanation": reasons, "review_flag": review_flag, "review_note": review_note, "uncertainty": context_note or ("Review this command with the process context before acting." if unsafe and probability < 0.7 else ""), "recommended_action": recommended}
