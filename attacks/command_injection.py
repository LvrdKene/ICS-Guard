"""Reproducible demo: an unauthorised high pump-speed Modbus write."""
from _client import client, send
connection = client()
try:
    # The demo starts in maintenance hold; return to normal operation so this
    # high demand is assessed as an unauthorised in-service command.
    send(connection.write_register(7, 0), "exit maintenance mode")
    send(connection.write_register(4, 9500), "SET_PUMP_SPEED = 95%")
    print("Sent Modbus command injection: SET_PUMP_SPEED = 95%")
finally:
    connection.close()
