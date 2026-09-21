"""Reproducible demo: gradual Modbus tank-setpoint drift."""
from _client import client, send
connection = client()
try:
    # Drift must occur during normal operation, not the legitimate maintenance
    # state used for the planned-drain demonstration.
    send(connection.write_register(7, 0), "exit maintenance mode")
    send(connection.write_coil(1, True), "open outlet valve")
    send(connection.write_coil(0, True), "start pump")
    send(connection.write_register(4, 7000), "pump speed = 70%")
    for value in (7000, 7500, 8000, 8500, 9000, 9500):
        send(connection.write_register(1, value), f"tank setpoint = {value / 100}%")
    print("Sent six incremental Modbus tank-setpoint changes (70% -> 95%)")
finally:
    connection.close()
