UEBA Demo – VMware Simulation & Dashboard

This directory contains the end-to-end demo environment used to simulate
realistic insider threat scenarios for the UEBA pipeline.

The demo emulates an enterprise internal network using VMware virtual
machines, lightweight log agents, a central ingest gateway, and a web-based
dashboard for SOC-style investigation.

This demo is separate from the core UEBA modeling pipeline and is intended
for demonstration, validation, and presentation purposes.

Demo Architecture Overview
```test
[ Windows VM Users ]
        │
        │  (Sysmon / Security Logs)
        ▼
[ Log Agent (PowerShell / Python) ]
        │
        │  HTTP / JSON
        ▼
[ ingest_gateway.py ]
        │
        │  Normalized logs
        ▼
[ UEBA Pipeline (src/) ]
        │
        ▼
[ dashboard.py ] ──► [ HTML Dashboard ]
```
Components
1. VMware Virtual Machines

Multiple Windows VMs simulate different users in an internal organization

Each VM represents:

A distinct user account

Different behavior patterns (normal vs anomalous)

Logs are generated from:

Logon activity

File access

HTTP activity

Device usage (optional)

⚠️ VM images are not included in this repository.

2. Log Ingest Gateway

File: demo/ingest_gateway.py

Responsibilities:

Receive logs from agents via HTTP

Validate and normalize incoming log records

Write logs into a central staging area for UEBA processing

This component simulates a lightweight SIEM ingest layer in an enterprise
environment.

3. Demo Backend API

File: demo/dashboard.py

Responsibilities:

Serve processed UEBA results

Expose APIs for:

Top-K anomalous users / days

Anomaly scores

Evaluation metrics

Act as the backend for the HTML dashboard

This backend is not production-grade and is intentionally simplified for
demo clarity.

4. HTML Dashboard

Location: dashboard/index.html

Features:

Overview of detected anomalies

Ranked list of suspicious user-days

Visualization of anomaly scores

SOC-oriented view focused on investigation, not alert flooding

The dashboard is implemented using static HTML + JavaScript and served by
the demo backend.

How to Run the Demo
1. Environment Setup
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

2. Start Ingest Gateway
python demo/ingest_gateway.py


The gateway listens for log events sent by agents.

3. Generate Logs

Log agents (VM or local simulation) send JSON events to the ingest gateway

Logs are stored and prepared for preprocessing

Agent scripts and VM configuration details are documented in /docs

4. Run UEBA Pipeline (Core)
python src/preprocessing/normalize_logs.py
python src/preprocessing/merge_logs.py

python src/feature_engineering/build_user_day_features.py
python src/feature_engineering/make_labels.py

python src/modeling/train_isolation_forest.py
python src/modeling/score_isolation_forest.py

python src/evaluation/evaluate_isolation_forest.py

5. Launch Dashboard Backend
python demo/dashboard.py


Then open:

dashboard/index.html

Demo Scenarios

Typical demo scenarios include:

Excessive after-hours logins

Sudden spikes in file access

Cross-machine login behavior

Abnormal HTTP destinations

Multi-day gradual behavior drift

Each scenario:

Appears legitimate in isolation

Becomes suspicious only when analyzed behaviorally

SOC Relevance

This demo reflects realistic SOC workflows:

No signature-based detection

No fixed thresholds

Analyst-driven Top-K investigation

Emphasis on alert quality, not alert volume

The UEBA system acts as a behavioral analytics layer on top of SIEM.

Limitations

Simplified VMware setup

Lightweight, non-persistent agents

Batch-oriented processing (not real-time)

Synthetic / simulated user behavior

Disclaimer

This demo is intended for educational and portfolio purposes only.

No real organizational data is included

No confidential systems are replicated

Not intended for production deployment
