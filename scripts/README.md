# Smoke Tests

This directory contains the smoke-test scripts used to validate the framework.
These scripts are examples and should be adapted for your domain.

## `e2e_smoke.py`

End-to-end smoke test: dataset generation, RAG ingest, SFT training, and eval.

Evaluation passes (inference RAG toggle):
- `base_with_rag`
- `base_without_rag`
- `tuned_with_rag`
- `tuned_without_rag`

Metrics reported by the default smoke evaluator:
- `action_accuracy`: exact match on the parsed `action` field.
- `param_accuracy`: schema-specific parameter match (exactness/tolerance rules live in the evaluator).
- `thermal_param_accuracy`: accuracy over the `set_thermal_profile` subset only.

Artifacts produced under `training_data/smoke/reports/<run_id>/`:
- `summary.md`, `summary.json`, `run.log`
- `predictions/*.jsonl`

## `smoke_cross_section.py`

Cross-sectional smoke test focused on lightweight training/inference behavior.
