"""Layer 1 demo: a real Modbus write that violates a simulator safety rule."""
from _client import client

connection = client()
try:
    # 95% is above the documented local hard limit while the normal plant is in service.
    connection.write_coil(1, True)
    connection.write_register(4, 9500)
    print("Safety-rule demo sent: SET_PUMP_SPEED = 95%")
finally:
    connection.close()
