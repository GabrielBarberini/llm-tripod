# Configs

This folder contains YAML configs used by Tripod and the bundled smoke tests.
The sections below explain what each config block does and where it is used.

## Files

- `configs/smoke_e2e_config.yaml`: end-to-end smoke test config used by
  `tests/smoke_e2e.py`.
- `configs/iot_domain_config.yaml`: example production config
  (treat as inspiration and replace paths/values).

## What each section controls

### `training.*` (core/training.py)

- `training.enabled`: skip adapter training when false.
- `training.base_model`: base checkpoint loaded by the trainer and evaluator.
- `training.dataset_path`: training JSON/JSONL/text path (the smoke test
  overwrites this with its generated dataset paths). When the path points to
  JSON/JSONL rows without a `text` field, `TripodOrchestrator.execute("prepare_train")`
  builds an SFT dataset using `prompting.*` plus optional RAFT retrieval. If you
  provide `text` rows directly, each row should be `PROMPT + response_marker + TARGET`.
- Raw rows can include task-specific input fields plus `expected` (or `target`)
  for the desired output. Optional fields like `rag_context`,
  `raft_query`/`rag_query`, or `raft_filters`/`rag_filters` let you control
  retrieval per row.
- `training.adapter_output_dir`: output directory for LoRA/QLoRA adapters.
- `training.hyperparameters.response_marker`: delimiter that splits prompt vs
  completion inside the training text.
- `training.hyperparameters.mask_prompt`: if true, only completion tokens
  contribute to loss (labels are masked with `-100`).

### `raft.*` (RAFT) + `rag.*` (RAG) (core/rag.py)

- `raft.enabled`: enable training-time retrieval (RAFT) when building
  training examples in `tests/smoke_e2e.py` or via `TripodOrchestrator.execute("prepare_train")`.
- `rag.enabled`: enable inference-time retrieval in `main.py` and
  the smoke evaluator.
- `rag.vector_db_type` / `raft.vector_db_type`: store adapter selector; `local`
  is built-in (numpy-backed). Add new adapters in `core/vectordb.py`.
- `rag.vector_db_path` / `raft.vector_db_path`: local vector store path (shared or separate).
- `rag.ingestion.embedding_model` / `raft.ingestion.embedding_model`: sentence-transformers model used to embed docs.
- `rag.retrieval.top_k` / `raft.retrieval.top_k`: number of docs to return per query.
- `rag.retrieval.strategy` / `raft.retrieval.strategy`: retrieval strategy; the built-in
  local store supports `similarity` only. Add new strategies when you implement
  a custom adapter in `core/vectordb.py`.

### `prompting.*` (core/prompting.py)

- `prompting.system_prompt` and `prompting.user_prompt_structure`: prompt template.
- `prompting.backend`: `raw` renders the template; `dspy` executes a DSPy program.
- `prompting.dspy.*`: DSPy options used when `prompting.backend: dspy`.

### `evaluation.*` (pipeline evaluation)

- `evaluation.*` is consumed by evaluation scripts (for example `tests/smoke_e2e.py`)
  and by the `TripodOrchestrator.execute("evaluate")` stub (logs `test_set_path`).
- `evaluation.entrypoint`: module path to import before evaluation (registers evaluators).
- `evaluation.evaluator`: registry key for the evaluator to run (defaults to `stub`).
- `evaluation.test_set_path`: evaluation dataset path used by your evaluator.
- `evaluation.metrics`: labels for report outputs (evaluation-defined).
- `evaluation.generation.top_p`: nucleus sampling for evaluator decoding; if
  `top_p < 1.0` and `do_sample` is not set, sampling is enabled automatically.
- `evaluation.generation.do_sample`: explicit sampling toggle for evaluator decoding.

Note: retrieval uses `top_k`; `top_p` is only used for generation sampling.

### Entry points

- `TripodOrchestrator.execute("ingest", {"documents": [...], "target": "raft"})`
  builds the RAFT vector store.
- `TripodOrchestrator.execute("ingest", {"documents": [...], "target": "rag"})`
  builds the inference RAG vector store.
- `TripodOrchestrator.execute("prepare_train")` builds an SFT dataset from raw JSON/JSONL.
- `TripodOrchestrator.execute("train")` runs training against
  `training.dataset_path`.
