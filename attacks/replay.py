"""Reproducible demo: re-send a captured valid Pump ON command."""
from _client import client, send
connection = client()
try:
    send(connection.write_coil(0, True), "first PUMP_ON")
    send(connection.write_coil(0, True), "replayed PUMP_ON")
    print("Sent replayed Modbus PUMP_ON command")
finally:
    connection.close()
