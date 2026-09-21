# ICS-Guard reproducible Modbus test clients

Start the Flask application first. It starts a local Modbus TCP demo plant at
`127.0.0.1:5020` by default. Every script below sends actual Modbus writes;
none of them send HTTP scenario events.

The four required attack scripts:

```powershell
.\venv\Scripts\python.exe attacks\command_injection.py
.\venv\Scripts\python.exe attacks\replay.py
.\venv\Scripts\python.exe attacks\wrong_time.py
.\venv\Scripts\python.exe attacks\setpoint_drift.py
```

Two more scripts exercise the ML layers specifically, once the Live
Simulator Detector has been built from the dashboard (each is designed to
sit just below/outside the deterministic safety rules, so it's genuinely
LightGBM or Isolation Forest that catches it, not a hand-coded threshold):

```powershell
.\venv\Scripts\python.exe attacks\lightgbm_only_attack.py
.\venv\Scripts\python.exe attacks\isolation_only_attack.py
```

And `physical_safety_attack.py` reproduces two hazardous command
combinations that aren't attack techniques at all -- they're pump/valve
physics (pumping into an already-full tank with the outlet shut; draining
below the low-level threshold outside a declared maintenance procedure) --
and are flagged the same way whether they come from this script or a real
click in the Plant Operations Console:

```powershell
.\venv\Scripts\python.exe attacks\physical_safety_attack.py
```

They are safe localhost-only demonstration clients. They never target an external address or physical device.

