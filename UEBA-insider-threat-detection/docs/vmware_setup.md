# VMware Setup – UEBA Demo Environment

This document describes the VMware-based enterprise simulation used to demonstrate the UEBA insider threat detection pipeline.

---

## 1. Overview

The demo environment emulates a small enterprise internal network consisting of:

- One **Domain Controller (DC)**
- Multiple **Windows 10 user workstations**
- Central **UEBA host** (log ingest + analytics + dashboard)

All components are deployed using **VMware Workstation**.

---

## 2. Virtual Machine Topology

### 2.1 Domain Controller

- OS: Windows Server 2019
- Roles:
  - Active Directory Domain Services (AD DS)
  - DNS
- Domain name: `company.local`

The Domain Controller manages:
- User accounts
- Departments
- Computer objects (workstations)

### 2.2 User Workstations

- OS: Windows 10
- Machines:
  - `WS-USER1`
  - `WS-USER2`
  - `WS-USER3`
  - `WS-USER4`

Each workstation:
- Is joined to the domain `company.local`
- Represents a distinct employee
- Runs a lightweight UEBA log agent

All workstations share the same base configuration but differ in **user behavior** during demo scenarios.

---

## 3. Active Directory Structure

### 3.1 Organizational Units (OU)

```
company.local
├─ Departments
│ ├─ IT
│ ├─ Sales
│ ├─ ProjectManagement
│ └─ SoftwareManagement
├─ Workstations
│ ├─ WS-USER1
│ ├─ WS-USER2
│ ├─ WS-USER3
│ └─ WS-USER4
```

Users are assigned to departments to enable **context-aware behavioral analysis**.

---

## 4. Networking

- Network mode: **NAT**
- All VMs can communicate with the UEBA host
- Typical addressing:
  - UEBA Host: `10.10.10.1`
  - Ingest port: `8010`

---

## 5. Notes

- VM images are **not included** in this repository due to size.
- The environment is simplified for demonstration purposes.
- Security hardening is intentionally minimal to focus on behavioral visibility.

---

## 6. Purpose

This VMware setup enables:
- Realistic user behavior generation
- Controlled insider threat simulations
- End-to-end validation of the UEBA pipeline
