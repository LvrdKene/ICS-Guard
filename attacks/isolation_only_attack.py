"""Layer 3 demo: a novel Modbus command sequence for anomaly detection.

Build the local model before running this script. Rapid valve-control chatter
is not a simulator rule violation and is not a labelled attack class. It is
outside the learned normal command-frequency baseline, so Isolation Forest
reviews the final write after LightGBM clears it.
"""
from _client import client

connection = client()
try:
    for _ in range(20):
        connection.write_register(5, 6500)
    connection.write_register(5, 400)
    print("Isolation-Forest-only demo sent: 21 rapid valve-position writes")
finally:
    connection.close()
