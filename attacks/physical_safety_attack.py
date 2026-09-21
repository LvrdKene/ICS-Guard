"""Layer 1 demo: two real Modbus command sequences that are individually
"valid" writes but are physically dangerous given the plant's own state --
exactly the class of command a well-formed-packet checker would miss, and
the reason SIM-DEADHEAD / SIM-UNSAFE-DRAIN exist in services/ics_guard.py.

Unlike the four required attack scripts, neither of these mimics a known
attack technique. They are pure pump/valve physics, and they fire the same
way whether the command came from this script or a real operator clicking
the Plant Operations Console -- these two are safety-console-context bugs,
not signature-matching demos.
"""
import time
from _client import client, send

connection = client()
try:
    print("--- Hazard 1: pump commanded into a full tank with the outlet shut ---")
    send(connection.write_register(7, 0), "leave maintenance mode")
    send(connection.write_register(5, 10000), "open outlet valve fully")
    send(connection.write_coil(0, True), "start inlet pump")
    send(connection.write_register(4, 4500), "SET_PUMP_SPEED = 45%")
    setpoint = connection.read_holding_registers(1, count=1).registers[0] / 100
    print(f"Waiting for the tank to reach its setpoint ({setpoint:.0f}%)...")
    for _ in range(120):
        level = connection.read_holding_registers(0, count=1).registers[0] / 100
        if level >= setpoint:
            break
        time.sleep(1)
    else:
        raise SystemExit("Tank did not reach its setpoint; check the plant is running.")
    send(connection.write_coil(1, False), "close outlet valve")
    send(connection.write_coil(0, True), "start pump (re-affirm) with the outlet now shut")
    print(f"Closed the valve and (re-)started the pump at tank level {level:.1f}% -- water has nowhere to go.")

    print()
    print("--- Hazard 2: draining below the low-level threshold outside maintenance ---")
    send(connection.write_register(7, 1), "enter maintenance mode")
    send(connection.write_coil(0, False), "stop inlet pump")
    send(connection.write_coil(1, True), "open outlet valve fully")
    print("Waiting for the tank to drain below its low-level threshold (25%)...")
    for _ in range(120):
        level = connection.read_holding_registers(0, count=1).registers[0] / 100
        if level < 25:
            break
        time.sleep(1)
    else:
        raise SystemExit("Tank did not drain below 25%; check the maintenance drain is running.")
    send(connection.write_register(7, 0), "leave maintenance mode")
    send(connection.write_coil(0, False), "stop pump (re-affirm) outside maintenance")
    send(connection.write_coil(1, True), "open valve (re-affirm) outside maintenance")
    print(f"Stopped the pump and opened the valve at tank level {level:.1f}% with maintenance mode off.")
finally:
    connection.close()
