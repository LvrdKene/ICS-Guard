# ICS-Guard

ICS-Guard is a Track E1 hackathon prototype for **Catching Unsafe Commands in Your Own Control System**.

It simulates a water tank, pump and valve controlled over a real, localhost-only Modbus TCP demo server (default `127.0.0.1:5020`). The detector evaluates whether a command is unsafe in its process context and advises an engineer; it never changes plant equipment automatically.

## Demonstrated scenarios

- normal controller operation and legitimate maintenance
- command injection
- replayed command
- valid command at the wrong moment
- gradual setpoint drift

## Detection approach

- A simulator-native dataset is generated into fixed process-command windows for live model training
- Isolation Forest uses only the normal operating windows as its baseline
- grouped evaluation keeps generated attack scenarios together to avoid row-level leakage
- transparent simulator guardrails for the four reproducible Track E attack cases, with process context
- LightGBM for known unsafe command patterns
- Isolation Forest for unusual normal-process behaviour
- engineer-facing safety explanations and recommended review actions

## Run locally

Use Python 3.11 or newer, then install dependencies and start Flask:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
flask --app app run --debug
```

Open `http://127.0.0.1:5000`, build the **local detection model**, and use the **Plant Operations Console** to issue planned maintenance, pump, valve, speed and setpoint commands through the local Modbus server. Every command is monitored and logged. Training uses simulator-native command and process data.

## Reproducible Modbus test clients

With the app running, the scripts in `attacks/` send the four demo patterns through
the local Modbus TCP plant. They are intentionally localhost-only and never target
a physical device. See [`attacks/README.md`](attacks/README.md) for commands.

ICS-Guard derives its normal/unsafe decision and any named pattern from the
command, timing, and simulated process state rather than from an attack label.

## Prototype boundary

This is a simulated, advisory-only hackathon prototype. It does not connect to a real plant or automatically issue protective commands. All four reproducible attack clients are kept in `attacks/` and send only to the local demo server.
