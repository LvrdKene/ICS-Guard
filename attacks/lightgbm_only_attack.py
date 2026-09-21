"""Layer 2 demo: a learned unsafe command pattern below every hard-rule limit.

Build the local model before running this script.  The 82% request is within
the simulator's explicit safety-rule threshold (< 88%) but belongs to the
known injection pattern used to train LightGBM.
"""
from _client import client

connection = client()
try:
    connection.write_coil(1, True)
    connection.write_register(4, 8200)
    print("LightGBM-only demo sent: SET_PUMP_SPEED = 82%")
finally:
    connection.close()
