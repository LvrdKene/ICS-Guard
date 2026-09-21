"""Local Modbus/TCP water-tank plant for the E1 demo (localhost only)."""
from __future__ import annotations

import socket
import struct
import threading
from typing import Callable

from pymodbus.client import ModbusTcpClient


class ModbusPlant:
    """A narrow real Modbus/TCP server; it never connects to physical equipment."""
    REGISTER_MAP = {
        "coil 00001": "Pump ON/OFF", "coil 00002": "Valve OPEN/CLOSED",
        "holding 40001": "Tank level (% x100, read-only process value)",
        "holding 40002": "Tank level setpoint (% x100)",
        "holding 40003": "Flow rate (L/s x100, read-only process value)",
        "holding 40005": "Pump speed (% x100)", "holding 40006": "Valve position (% x100)",
        "holding 40008": "Maintenance mode (0 = normal control, 1 = maintenance)",
    }

    def __init__(self, on_command: Callable[[dict], None], host: str = "127.0.0.1", port: int = 5020, state_provider: Callable[[], dict] | None = None):
        self.on_command, self.host, self.port, self.state_provider = on_command, host, port, state_provider
        self.running, self.error = False, ""
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def status(self) -> dict:
        return {"running": self.running, "host": self.host, "port": self.port, "protocol": "Modbus TCP",
                "register_map": self.REGISTER_MAP, "error": self.error}

    def start(self) -> None:
        if self.running: return
        def serve() -> None:
            try:
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((self.host, self.port)); listener.listen(8); listener.settimeout(.5)
                self._listener, self.running, self.error = listener, True, ""
                while self.running:
                    try: connection, address = listener.accept()
                    except TimeoutError: continue
                    threading.Thread(target=self._handle_client, args=(connection, address[0]), daemon=True).start()
            except Exception as error:
                self.error = str(error)
            finally:
                self.running = False
                if self._listener: self._listener.close()
        self._thread = threading.Thread(target=serve, name="ics-guard-modbus-plant", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._listener: self._listener.close()

    def _handle_client(self, connection: socket.socket, source_ip: str) -> None:
        with connection:
            while self.running:
                header = self._receive(connection, 7)
                if not header: return
                transaction, protocol, length, unit = struct.unpack(">HHHB", header)
                if protocol != 0 or length < 2: return
                pdu = self._receive(connection, length - 1)
                if not pdu: return
                function = pdu[0]
                if function in (5, 6) and len(pdu) == 5:
                    address, value = struct.unpack(">HH", pdu[1:])
                    if function == 5: value = int(value == 0xFF00)
                    self._observe_write(function, address, value, source_ip)
                    response = pdu
                elif function in (1, 3) and len(pdu) == 5:
                    address, count = struct.unpack(">HH", pdu[1:])
                    response = self._read_response(function, address, count)
                else:
                    response = bytes([function | 0x80, 1])
                connection.sendall(struct.pack(">HHHB", transaction, 0, len(response) + 1, unit) + response)

    @staticmethod
    def _receive(connection: socket.socket, count: int) -> bytes:
        data = b""
        while len(data) < count:
            part = connection.recv(count - len(data))
            if not part: return b""
            data += part
        return data

    def _observe_write(self, function: int, address: int, value: int, source_ip: str) -> None:
        if function == 5 and address == 0: name, command_value = "PUMP_ON", value
        elif function == 5 and address == 1: name, command_value = "SET_VALVE_POSITION", 100 if value else 0
        elif function == 6 and address == 1: name, command_value = "SET_TANK_SETPOINT", value / 100
        elif function == 6 and address == 4: name, command_value = "SET_PUMP_SPEED", value / 100
        elif function == 6 and address == 5: name, command_value = "SET_VALVE_POSITION", value / 100
        elif function == 6 and address == 7: name, command_value = "SET_MAINTENANCE_MODE", bool(value)
        else: return
        self.on_command({"command_name": name, "command_value": command_value, "source_ip": source_ip,
                         "destination_ip": self.host})

    def _read_response(self, function: int, address: int, count: int) -> bytes:
        """Expose current simulator telemetry through normal Modbus read functions."""
        state = self.state_provider() if self.state_provider else {}
        if function == 1:
            values = [bool(state.get("pump_running")) if address + index == 0 else bool(state.get("valve_open")) if address + index == 1 else False for index in range(count)]
            packed = sum((int(value) << index) for index, value in enumerate(values)).to_bytes((count + 7) // 8, "little")
            return bytes([function, len(packed)]) + packed
        register_values = {0: round(float(state.get("tank_level", 0)) * 100), 1: round(float(state.get("tank_setpoint", 0)) * 100), 2: round(float(state.get("flow_rate", 0)) * 100), 4: round(float(state.get("pump_speed", 0)) * 100), 5: round(float(state.get("valve_position", 0)) * 100)}
        values = [int(register_values.get(address + index, 0)) for index in range(count)]
        payload = b"".join(struct.pack(">H", value) for value in values)
        return bytes([function, len(payload)]) + payload

    def execute_control(self, action: str, value: float | None = None) -> None:
        """Issue operator controls through the same local Modbus TCP path as clients."""
        if not self.running: self.start()
        client = ModbusTcpClient(self.host, port=self.port, timeout=3)
        if not client.connect(): raise RuntimeError(self.error or "Local Modbus plant is not available yet. Try again in a moment.")
        try:
            def send(response):
                if response.isError():
                    raise RuntimeError(f"Local Modbus plant rejected the {action} command: {response}")

            if action == "startup":
                send(client.write_register(7, 0)); send(client.write_register(5, 6500)); send(client.write_coil(1, True)); send(client.write_coil(0, True)); send(client.write_register(4, 4500))
            elif action == "normal":
                send(client.write_register(7, 0)); send(client.write_register(5, 6500)); send(client.write_coil(1, True)); send(client.write_coil(0, True)); send(client.write_register(4, 4500))
            elif action == "shutdown":
                # Maintenance mode holds the automatic controller while the
                # planned shutdown takes place, preventing an immediate restart
                # or a continuing drain through the outlet.
                send(client.write_register(7, 1)); send(client.write_register(4, 0)); send(client.write_coil(0, False)); send(client.write_coil(1, False))
            elif action == "maintenance_on":
                send(client.write_register(7, 1)); send(client.write_coil(0, False)); send(client.write_coil(1, False))
            elif action == "maintenance_off": send(client.write_register(7, 0))
            elif action == "maintenance_drain":
                # A deliberate, observable maintenance procedure.
                send(client.write_register(7, 1)); send(client.write_coil(0, False)); send(client.write_coil(1, True))
            elif action == "open_valve": send(client.write_coil(1, True))
            elif action == "close_valve": send(client.write_coil(1, False))
            # Deliberately NOT bundled with a maintenance-mode write (unlike
            # shutdown/maintenance_drain above): these are direct, standalone
            # actuator overrides, exactly as the dashboard groups them under
            # "Manual Equipment Commands" rather than "Legitimate
            # Maintenance". Silently forcing maintenance mode here used to
            # mean every Stop Pump click retroactively looked like a
            # declared maintenance action even when nobody asked for one,
            # which masked genuinely improper use of this exact command
            # (see SIM-UNSAFE-DRAIN / SIM-DEADHEAD in services/ics_guard.py).
            # The automatic controller records its desired state but does not
            # overwrite a command received through this path.
            elif action == "start_pump": send(client.write_coil(0, True))
            elif action == "stop_pump": send(client.write_coil(0, False))
            elif action == "set_pump_speed": send(client.write_register(4, round(float(value or 0) * 100)))
            elif action == "set_valve_position": send(client.write_register(5, round(float(value or 0) * 100)))
            elif action == "set_tank_setpoint": send(client.write_register(1, round(float(value or 0) * 100)))
            else: raise ValueError("Unsupported tank-control action.")
        finally:
            client.close()
