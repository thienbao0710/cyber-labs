# Demo Scenarios – Insider Threat Simulation

This document describes the attack scenarios demonstrated during the UEBA live demo.

---

## 1. Demo Objective

The goal of the demo is to validate that the UEBA system can:

- Learn normal user behavior over time
- Detect subtle behavioral deviations
- Highlight risky users and days for SOC investigation

The demo focuses on **behavioral anomalies**, not malware detection.

---

## 2. Baseline Period

Before executing any attack scenario:

- A **30-day baseline** of normal activity is generated
- Each user performs routine actions:
  - Regular logon times
  - Typical file usage
  - Normal web browsing

This baseline allows the model to learn:
- Personal habits
- Department-level patterns
- Time-of-day behavior

---

## 3. Attack Scenario: Cross-User Insider Activity

### Scenario Summary

- **User 4** (Sales department) performs suspicious actions:
  - Logs in outside working hours
  - Accesses another user's machine
  - Copies sensitive files to USB
  - Deletes files after exfiltration

### Key Characteristics

- Actions appear legitimate in isolation
- No malware or exploit is involved
- Risk emerges only through **behavioral context**

---

## 4. Observable Signals

The UEBA system detects anomalies based on:

- After-hours logon behavior
- Cross-machine access
- Sudden spike in file operations
- USB usage combined with file deletion
- Deviation from historical user baseline

---

## 5. Dashboard View

During the demo, SOC analysts can observe:

### 5.1 Live Overview
- Global anomaly trends
- Daily anomaly volume

### 5.2 User Drill-down
- Per-user behavior timelines
- Anomaly score evolution

### 5.3 Alerts
- Ranked suspicious user-days
- Department-level aggregation

### 5.4 Log Review
- Raw normalized logs
- Supporting evidence for investigation

---

## 6. SOC Relevance

This demo reflects realistic SOC workflows:

- No signatures
- No fixed thresholds
- Analyst-driven Top-K investigation
- Focus on alert quality, not quantity

The UEBA system acts as a **behavioral analytics layer** on top of SIEM.

---

## 7. Limitations

- Synthetic behavior
- Batch-oriented processing
- No real-time enforcement
- Simplified enterprise scale

---

## 8. Purpose

These scenarios are designed for:
- Capstone defense
- Portfolio demonstration
- UEBA concept validation

They are **not intended for production deployment**.
