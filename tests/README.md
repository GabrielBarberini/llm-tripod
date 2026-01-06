# Tests

This folder documents the bundled smoke tests and their intent. It can also host
future `pytest` tests.

## End-to-end smoke (`tests/smoke_e2e.py`)

End-to-end smoke test: dataset generation, RAFT (training retrieval), RAG ingest,
SFT training, and eval. This script is a template for new domains.
It builds its own SFT dataset, so you do not need to run `prepare_train`.

Run:

```bash
python3 tests/smoke_e2e.py --n 6000 --eval-samples 200
```

Useful flags:
- `--num-policies 50`: number of distinct policy docs (example generator).
- `--holdout-policies / --no-holdout-policies`: whether eval uses unseen document IDs.
- `--report-dir <path>`: override report output location.
- `--no-save-predictions`: skip writing per-sample JSONL (faster, smaller).

Note: `--num-policies`/`--holdout-policies` apply to the bundled dataset generator
(`tests/generate_smoke_dataset.py`).

Evaluation passes (inference RAG toggle):
- `base_with_rag`
- `base_without_rag`
- `tuned_no_raft_with_rag`
- `tuned_no_raft_without_rag`
- `tuned_raft_with_rag` (only when `raft.enabled` is true)
- `tuned_raft_without_rag` (only when `raft.enabled` is true)

Metrics reported by the default smoke evaluator:
- `action_accuracy`: exact match on the parsed `action` field.
- `param_accuracy`: schema-specific parameter match (exactness/tolerance rules live in the evaluator).
- `thermal_param_accuracy`: accuracy over the `set_thermal_profile` subset only.

When RAFT is enabled, `summary.json` also includes `raft_lift_*_param_accuracy` deltas
that compare RAFT-trained adapters against no-RAFT adapters under the same inference
RAG setting.

Sampling knobs for the evaluator live under `evaluation.generation` in
`configs/smoke_e2e_config.yaml` (default is deterministic).

Artifacts produced under `training_data/smoke/reports/<run_id>/`:
- `summary.md`, `summary.json`, `run.log`
- `predictions/*.jsonl`

## Local smoke (`tests/smoke_local.py`)

Lightweight, single-machine smoke test focused on local training + inference behavior.
It builds a tiny in-memory vector store, fine-tunes a small model, and scores on a
held-out split.

Run:

```bash
python3 tests/smoke_local.py
```

Notes:
- Downloads Hugging Face models on first run (internet required).
- Uses a small hardcoded dataset and a few-shot prompt to keep runtime short.
