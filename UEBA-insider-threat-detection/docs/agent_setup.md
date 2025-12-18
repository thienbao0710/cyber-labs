# UEBA Log Agent – Setup & Behavior

This document explains how the UEBA log agent operates and how it is deployed on user workstations.

---

## 1. Agent Purpose

The UEBA agent is a lightweight collector responsible for:

- Extracting local user activity from Windows
- Normalizing events into structured records
- Sending logs to the central ingest gateway

The agent simulates how endpoint telemetry is collected in real enterprises.

---

## 2. Deployment

- Agent type: PowerShell script
- Location on workstation:
C:\demo\agent\


Each Windows VM runs **one agent instance** tied to the currently logged-in user.

---

## 3. Log Sources

The agent collects events from:

### 3.1 Logon Activity
- Windows Security Event Log
- Event IDs:
- 4624 (Logon)
- 4625 (Logon Failed)
- 4634 / 4647 (Logoff)

### 3.2 File Activity
- Sysmon (Microsoft-Windows-Sysmon/Operational)
- Monitored areas:
- `C:\demo\userdata\`
- Removable USB drives

Tracked actions:
- File create
- File write
- File delete
- File copy to USB

### 3.3 HTTP Activity
- Sysmon NetworkConnect events
- Browser-based traffic only (Chrome, Edge, Firefox)
- Filters applied to remove:
- System traffic
- Microsoft telemetry
- Background services

### 3.4 Device Activity
- USB insertion detection
- Logical removable drives

---

## 4. Identity Resolution

The agent determines the active user using:
1. `quser`
2. WMI (`Win32_ComputerSystem`)
3. Environment variables

Machine accounts (`WS-XX$`) are automatically excluded.

---

## 5. Data Transmission

- Protocol: HTTP
- Format: JSON
- Endpoint:
http://<UEBA_HOST>:8010/ingest


Logs are sent in batches every **N minutes** (default: 5).

---

## 6. State Management

The agent maintains a local state file:
agent_state.json


This ensures:
- No duplicate log transmission
- Incremental log collection
- Safe restart behavior

---

## 7. Notes

- Agent must be run with **Administrator privileges** to access Security logs.
- The agent is intentionally simple and non-persistent.
- It is designed for demo and educational purposes only.
