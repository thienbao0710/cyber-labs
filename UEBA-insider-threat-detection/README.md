# UEBA – Insider Threat Detection

This repository presents an end-to-end **User and Entity Behavior Analytics (UEBA)**
pipeline for detecting anomalous user behavior in internal systems using
**unsupervised learning**, with a focus on **insider threat detection**.

The project demonstrates how behavioral baselining and anomaly ranking can
complement traditional SIEM systems, especially in environments with limited
or unreliable labeled attack data.

---

## Problem Statement

Insider threats are difficult to detect because malicious actions are often
embedded within **legitimate user activity**. Rule-based SIEM detections and
signature-based approaches struggle to identify subtle deviations in daily
behavior.

This lab addresses that challenge by:
- Modeling **normal user behavior over time**
- Detecting **behavioral deviations** rather than known attack patterns
- Prioritizing **high-risk users and days** for SOC analyst investigation

---

## High-Level Architecture

```text
Raw Logs
↓
Preprocessing & Normalization
↓
User–Day Feature Engineering
↓
Anomaly Detection (Isolation Forest)
↓
SOC-Oriented Evaluation & Ranking
```

---

## Methodology

### 1. Log Preprocessing
- Normalize and consolidate heterogeneous log sources
- Standardize timestamps, users, hosts, and actions
- Prepare clean input for feature extraction

### 2. User–Day Feature Engineering
- Aggregate events at **User–Day granularity**
- Extract volume, diversity, and temporal features
- Build stable behavior profiles suitable for baselining

### 3. Anomaly Detection
- Apply **Isolation Forest** for unsupervised anomaly detection
- Rank user-days by anomaly score instead of fixed thresholds
- Support Top-K investigation workflows

### 4. Evaluation (SOC-Oriented)
- ROC-AUC and PR-AUC for overall separability
- **Precision@K** and **False Positive Rate@K** for analyst usability
- Emphasis on *alert quality*, not alert volume

---

## Repository Structure

```text
UEBA-insider-threat-detection/
├─ src/                          
│  ├─ preprocessing/
│  │  ├─ normalize_logs.py
│  │  └─ merge_logs.py           
│  ├─ feature_engineering/
│  ├─ modeling/
│  └─ evaluation/
│
├─ demo/                         # DEMO / VMWARE SIMULATION
│  ├─ ingest_gateway.py          # receive logs from agents
│  ├─ dashboard.py                # demo backend (serve dashboard)
│  └─ README.md                  # how demo works
│
├─ dashboard/
│  ├─ index.html                 # html_dashboard
│  └─ README.md
│
├─ docs/                         # DOCUMENTATION (VMware, agent, scenarios)
│  ├─ vmware_setup.md
│  ├─ agent_setup.md
│  └─ demo_scenarios.md
│
├─ requirements.txt
└─ README.md
```

---

## Data

- The pipeline is designed for **enterprise user activity logs**
  (logon, file access, HTTP, device activity, etc.).
- Only **small synthetic samples** are used for demonstration.
- Full datasets are **excluded** due to size and licensing constraints.

---

## How to Run

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
2. Preprocessing
python src/preprocessing/normalize_logs.py
python src/preprocessing/merge_logs.py
3. Feature Engineering
python src/feature_engineering/build_user_day_features.py
python src/feature_engineering/make_labels.py
4. Modeling (Isolation Forest)
python src/modeling/train_isolation_forest.py
python src/modeling/score_isolation_forest.py
5. Evaluation
python src/evaluation/evaluate_isolation_forest.py
```

Notes

This demo assumes pre-cleaned sample data.

Parameters can be adjusted for different organizational baselines.

SOC Relevance
This project reflects a realistic UEBA workflow used in SOC environments:

Reduces alert fatigue by ranking anomalies instead of firing rules

Detects insider-style behaviors missed by signature-based tools

Acts as a behavioral analytics layer on top of SIEM

Supports analyst triage with high-precision Top-K alerts

Limitations
Synthetic data does not fully capture real-world noise

User–Day aggregation may hide short-lived session-level anomalies

Batch-oriented processing (no real-time streaming)

Future Improvements
Session-level or sequence modeling (LSTM / Transformer)

Hybrid Rule + ML detection to reduce false positives

Adaptive baselines to handle concept drift

Integration with network telemetry (proxy, DNS, firewall)

Near real-time scoring and alerting

Disclaimer
This repository is intended for educational and portfolio purposes only.
It does not contain real organizational data or confidential information.
