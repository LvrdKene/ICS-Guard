"""Shared localhost-only Modbus client helpers for the hackathon demonstrations."""
from __future__ import annotations
import os
from pymodbus.client import ModbusTcpClient
HOST = "127.0.0.1"
PORT = int(os.environ.get("ICS_GUARD_MODBUS_PORT", "5020"))
def client() -> ModbusTcpClient:
    connection = ModbusTcpClient(HOST, port=PORT, timeout=3)
    if not connection.connect(): raise SystemExit(f"ICS-Guard demo plant is not reachable at {HOST}:{PORT}. Start app.py first.")
    return connection


def send(response, description: str) -> None:
    """Fail a demo clearly if the local Modbus server rejects a write."""
    if response.isError():
        raise RuntimeError(f"Modbus rejected {description}: {response}")
