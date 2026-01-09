# IoT Thermal Control Pipeline

This walkthrough uses the IoT example config to illustrate an end-to-end flow.

Config:
- `configs/iot_config.yaml`
  - `evaluation.entrypoint` points at `pipelines.iot.evaluator` (registry key: `iot`).

What it demonstrates:
- RAFT-style training data enrichment (optional).
- Inference-time RAG for live requests.
- Prompt structure and schema constraints.

Use it as an inspiration for new domains (pricing tables, rules, SOPs, etc).

Suggested flow (example):

`documents` accepts a list of strings or objects shaped like `{"id": "...", "text": "...", "metadata": {...}}`.

```bash
# Ingest documents for training-time retrieval (RAFT).
python -c "from main import TripodOrchestrator; TripodOrchestrator().execute('ingest', {'documents': ['policy doc', 'heuristic doc'], 'target': 'raft'})"

# Ingest documents for inference-time retrieval (RAG).
python -c "from main import TripodOrchestrator; TripodOrchestrator().execute('ingest', {'documents': ['policy doc', 'heuristic doc'], 'target': 'rag'})"

# Build SFT training data from raw JSON/JSONL (only if your dataset lacks a `text` field).
python -c "from main import TripodOrchestrator; TripodOrchestrator().execute('prepare_train')"

# Train the adapter using the prepared SFT dataset.
python -c "from main import TripodOrchestrator; TripodOrchestrator().execute('train')"

# Run the IoT evaluation hook (schema-only metrics; replace with your evaluator).
python -c "from main import TripodOrchestrator; TripodOrchestrator().execute('evaluate')"
```
