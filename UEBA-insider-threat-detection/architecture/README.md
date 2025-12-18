# System Architecture – UEBA Insider Threat Detection

This document describes the **logical architecture** of the UEBA Insider Threat Detection system.
It focuses on **component responsibilities, data flow, and design rationale**, rather than
implementation details or deployment instructions.

The goal of this architecture is to demonstrate how **behavior-based analytics** can be applied
to detect insider-style threats in enterprise environments where labeled attack data is limited
or unavailable.

---

## 1. High-Level Architecture

The system is organized as a modular pipeline that separates **log collection**, **behavior modeling**,
and **SOC investigation**.
```
[ Windows User VMs ]
|
| (Logon / File / HTTP / Device Events)
v
[ Log Agent ]
|
| JSON over HTTP
v
[ Ingest Gateway ]
|
| Normalized Logs
v
[ UEBA Pipeline ]

Preprocessing

User–Day Feature Engineering

Anomaly Detection (Isolation Forest / Autoencoder)
|
v
[ Evaluation & Ranking ]
|
v
[ SOC Dashboard ]
```

This layered design reflects how UEBA systems are typically integrated **on top of SIEM or log
management platforms**, rather than replacing them.

---

## 2. Component Breakdown

### 2.1 Windows User Virtual Machines
- Simulate enterprise endpoints used by different employees
- Each VM represents a distinct user with:
  - Department and role (managed via Active Directory)
  - Normal and abnormal behavior patterns
- Operating System:
  - Windows 10 (endpoints)
  - Windows Server 2019 (Domain Controller)

---

### 2.2 Log Agent
- Runs on each Windows endpoint
- Collects security-relevant events from:
  - Windows Event Log (Security, System)
  - Sysmon (file, network, device activity)
- Normalizes raw events into structured records
- Sends logs periodically to the ingest gateway via HTTP (JSON)

**Purpose:**  
Decouples endpoint log collection from central analytics and simulates realistic endpoint telemetry.

---

### 2.3 Ingest Gateway
- Centralized log receiver
- Responsibilities:
  - Receive logs from multiple agents
  - Perform lightweight validation and normalization
  - Store logs in a staging area for downstream processing
- Acts as a simplified SIEM-like ingestion layer

**Purpose:**  
Isolates data collection from analytics and allows the UEBA pipeline to remain batch-oriented.

---

### 2.4 UEBA Pipeline
The core analytical component of the system.

#### Preprocessing
- Cleans and consolidates logs from multiple sources
- Normalizes timestamps, users, hosts, and event schemas

#### User–Day Feature Engineering
- Aggregates events at **User–Day granularity**
- Extracts behavioral features such as:
  - Activity volume
  - Resource diversity
  - Temporal patterns
- Builds stable behavioral baselines for each user

#### Anomaly Detection
- Applies **unsupervised learning models**:
  - Isolation Forest (primary)
  - Autoencoder (comparative)
- Produces anomaly scores rather than binary alerts

**Purpose:**  
Detects deviations from learned normal behavior without relying on labeled attack data.

---

### 2.5 Evaluation & Ranking
- Evaluates model outputs using SOC-oriented metrics:
  - ROC-AUC, PR-AUC
  - Precision@K
  - False Positive Rate@K
- Ranks anomalies to support **Top-K investigation workflows**

**Purpose:**  
Ensure model outputs are usable by SOC analysts and reduce alert fatigue.

---

### 2.6 SOC Dashboard
- Visualization and investigation layer
- Provides:
  - High-level overview of detected anomalies
  - User drill-down views
  - Ranked alerts for analyst triage
  - Raw log inspection
- Focuses on **investigation-first design**, not alert flooding

**Purpose:**  
Bridge the gap between machine learning outputs and SOC analyst decision-making.

---

## 3. Data Flow Summary

1. User activity generates logs on Windows endpoints
2. Log agents collect and normalize events locally
3. Logs are sent to the ingest gateway over HTTP
4. UEBA pipeline processes logs in batch mode
5. Models assign anomaly scores to user-day records
6. Results are evaluated, ranked, and visualized in the dashboard

This flow mirrors how UEBA systems complement existing SIEM deployments in real environments.

---

## 4. Design Decisions & Rationale

- **User–Day aggregation** is chosen to balance signal stability and scalability
- **Unsupervised learning** is used due to lack of reliable insider threat labels
- **Anomaly ranking** is preferred over fixed thresholds to reduce false positives
- **Separation of demo and modeling components** avoids coupling analytics with infrastructure
- **Batch processing** simplifies reproducibility and evaluation for academic settings

---

## 5. Scope and Non-Goals

### In Scope
- Insider threat detection via behavioral analytics
- SOC-oriented evaluation and investigation
- Demonstration in a controlled virtual environment

### Out of Scope
- Real-time streaming detection
- Production-grade SIEM integration
- Automated response or prevention
- Large-scale enterprise deployment

---

## 6. Relationship to Other Documentation

- **Deployment, VMware configuration, and agents:** see `/docs`
- **Demo environment and dashboard backend:** see `/demo`
- **Core analytics implementation:** see `/src`

---

## 7. Intended Audience

This document is intended for:
- Academic reviewers and supervisors
- SOC practitioners evaluating UEBA concepts
- Readers interested in behavior-based insider threat detection

It is not intended as a deployment or operations manual.
