# training_data

Local, unversioned data and artifacts live here.

Suggested layout:
- `training_data/vectordb/`: local vector store files (docs + embeddings).
- `training_data/datasets/`: your domain train/test files (JSONL, parquet, etc).
- `training_data/adapters/`: optional local LoRA/QLoRA adapter outputs.
- `training_data/reports/`: evaluation artifacts and smoke-test reports.

Keep large datasets and adapters out of git; use this directory as a local
scratchpad referenced by your config paths.

## Dataset Formats

Tripod supports two JSONL dataset formats depending on your workflow:

### 1. Raw JSONL (for `prepare_train`)

Use this format when you want Tripod to build SFT training text from structured
rows. Point `training.dataset_path` to this file and run
`TripodOrchestrator.execute("prepare_train")` to convert it.

**Required fields:**
- `expected` or `target`: the desired output (string or JSON-serializable object)

**Optional fields:**
- `task_label`: task label (replaces `{{ task_label }}` in `prompting.system_prompt`)
- `input_data`: task input data (dict, list, or any JSON-serializable value)
- `rag_context`: pre-computed retrieval context (string)
- `raft_query` or `rag_query`: query string for RAFT retrieval during prep
- `raft_filters` or `rag_filters`: metadata filters for retrieval (dict)

**Example raw JSONL:**
```jsonl
{"task_label": "Thermal Control", "input_data": {"temp": 78.5, "vibration": 1.2}, "expected": {"action": "schedule_maintenance", "parameters": {"fan_speed": 80}, "reasoning": "High vibration detected"}}
{"task_label": "Thermal Control", "input_data": {"temp": 75.0, "vibration": 0.5}, "expected": {"action": "maintain", "parameters": {"fan_speed": 50}, "reasoning": "Normal operating conditions"}}
```

**Note:** The field name `input_data` is domain-agnostic and works for any task type (IoT sensors, pricing queries, document inputs, etc.).

**After `prepare_train`:**
The orchestrator renders prompts using `prompting.system_prompt` and
`prompting.user_prompt_structure`, optionally enriches with RAFT retrieval,
and writes a new `*_sft.jsonl` file with the SFT format below.

### 2. SFT JSONL (direct training input)

Use this format when you already have pre-formatted SFT text or want to
provide training data directly. Point `training.dataset_path` to this file
and run `TripodOrchestrator.execute("train")` directly.

**Required field:**
- `text`: complete SFT string in the format `PROMPT + response_marker + TARGET`

The `response_marker` (default: `"\nASSISTANT:\n"`) separates prompt from
completion. When `training.hyperparameters.mask_prompt` is true, only tokens
after the marker contribute to loss.

**Example SFT JSONL:**
```jsonl
{"text": "SYSTEM:\nYou are an expert Industrial IoT Controller...\n\nUSER:\n### HISTORY:\nContext_1 (score=0.85): Policy: keep temp < 80C...\n\n### INPUT:\n{\"temp\": 78.5, \"vibration\": 1.2}\n\n### INSTRUCTION:\nOutput JSON now.\nASSISTANT:\n{\"action\": \"schedule_maintenance\", \"parameters\": {\"fan_speed\": 80}, \"reasoning\": \"High vibration detected\"}"}
{"text": "SYSTEM:\nYou are an expert Industrial IoT Controller...\n\nUSER:\n### HISTORY:\n\n### INPUT:\n{\"temp\": 75.0, \"vibration\": 0.5}\n\n### INSTRUCTION:\nOutput JSON now.\nASSISTANT:\n{\"action\": \"maintain\", \"parameters\": {\"fan_speed\": 50}, \"reasoning\": \"Normal operating conditions\"}"}
```

**Note:** In prompt templates, use `{{ input_data }}` to inject the input data into your prompts.

**Note:** The exact prompt structure depends on your `prompting.*` config.
The smoke test example (`tests/fixtures/generate_smoke_dataset.py`) shows how to
build raw JSONL; `tests/acceptance/smoke_e2e.py` shows how to build SFT JSONL directly.

## Path Configuration

Set `training.dataset_path` in your YAML config to point to your dataset file:

```yaml
training:
  dataset_path: "./training_data/datasets/my_domain/train.jsonl"
```

For relative paths, the working directory is the project root. Absolute paths
and remote URIs (S3, etc.) are also supported if your dataset loader supports them.
