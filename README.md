This project targets the gap between packet-level validity and process-level safety. ICS-Guard is built around a simulated water-treatment process and asks whether each incoming control command is safe in its current context. The implementation is advisory: it identifies risk, explains why, and asks for engineer review rather than automatically changing plant equipment.

What ICS-Guard does
i.  Receives real Modbus/TCP writes through a localhost-only demo plant
ii. Maintains a closed-loop water-tank simulation with tank level, setpoint, flow, pressure, pump, valve, and maintenance state.
iii. Evaluates each command using deterministic simulator safety rules first, then LightGBM and finally Isolation Forest.
iv. Uses SHAP to turn model behaviour into operator-readable feature explanations.
v.  Logs analysed events and presents live decisions, evidence, filters, model status, and downloadable reports in the dashboard.
