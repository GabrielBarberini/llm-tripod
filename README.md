# LLM Tripod

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/MPP-v1.2.0-blue)](https://deepwiki.com/GabrielBarberini/llm-tripod)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/GabrielBarberini/llm-tripod)

Tripod is a lightweight, modular **LLM integration-test harness** for fast end-to-end iteration (not a full training platform). It separates the pipeline into three “legs”:

1. **Training (Leg 1)**: Fine-tune a small LLM with LoRA/QLoRA (`core/training.py`).
2. **RAG (Leg 2)**: Retrieve relevant reference snippets from a local vector store (`core/rag.py`, `core/vectordb.py`).
3. **Prompting (Leg 3)**: Build a deterministic prompt + output schema (`core/prompting.py`).

Use it to run repeatable end-to-end experiments and iterate on **data, retrieval, and prompting** until the target task meets your metrics. Configs expose a minimal set of controls; extend `core/training.py` and `core/rag.py` for domain-specific requirements. Smoke tests and example pipelines are templates for adaptation, not benchmarks.

## Repository Structure

- `core/`: implementations for `TrainingLeg`, `RAGLeg`, `PromptLeg`, plus config models.
- `configs/`: YAML configs.
  - `configs/smoke_e2e_config.yaml`: end-to-end smoke test config.
  - `configs/iot_domain_config.yaml`: example production config (IoT-themed; treat as inspiration).
  - `configs/README.md`: explains each config section and how it is used at runtime.
- `tests/`: test scripts to validate modules and integrations; see `tests/README.md`.
- `pipelines/`: example Tripod pipelines to use as a reference for extension.
- `training_data/`: local data/artifacts; see `training_data/README.md` for layout guidance.
- `main.py`: `TripodOrchestrator` entry point.

## Glossary (Key Terms)

- **Base model**: the pretrained checkpoint (`training.base_model`), e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
- **Adapter**: LoRA/QLoRA weights saved to `training.adapter_output_dir`.
- **Tuned model**: base model + adapter loaded together (via `peft.PeftModel`).
- **SFT**: supervised fine-tuning; in this repo it refers to building prompt + target training text before adapter training.
- **RAFT**: retrieval-augmented fine-tuning; retrieved text is concatenated into training prompts so the model learns to use retrieval at inference.
- **RAG context**: retrieved text inserted into the prompt (domain rules, docs, examples, guidelines).
- **With RAG / without RAG**: ablation toggle that either injects retrieved context into the prompt or leaves it empty (applied at inference/eval time via `rag.enabled`; training can still be RAFT-enriched via `raft.enabled`).
- **Holdout IDs / `--holdout-policies`**: example smoke-dataset setting where evaluation uses document IDs that never appear in training (forces generalization via retrieval instead of memorization). In your domain, “IDs” could be SKUs, policy numbers, error codes, etc.
- **Metrics**: task-specific evaluation signals defined by your evaluation script; see `tests/README.md` for the smoke-test metrics.

Important: with **holdout enabled**, “without RAG” metrics can be low by design (the prompt may not contain the missing information).

## Flow of Information summary

### Training (offline on GPU node)

```mermaid
flowchart TD
  Data[Train examples] --> RAFT[RAFT retrieval]
  RAFT --> SFT[Build SFT text]
  SFT --> Train[TrainingLeg LoRA/QLoRA]
  Train --> Adapter[Adapter dir]
```

- Runs **in phases**: ingest → build training file → train.
- In the end-to-end smoke test (`tests/smoke_e2e.py`), the SFT text is built as `PROMPT + ASSISTANT + TARGET` and the `ASSISTANT:` delimiter is used to mask prompt tokens during training (completion-style SFT).
- RAFT enrichment happens during training-file construction when `raft.enabled` is true; if it is false or ingestion is skipped, training examples get empty context.

### Inference

```mermaid
flowchart LR
  Input[Input payload] --> RAG[RAGLeg inference]
  RAG --> Prompt[PromptLeg raw or dspy]
  Prompt --> LLM[LLM inference engine]
  LLM --> Output[Structured output]
```

- Runs **sequentially**: RAG → Prompt → LLM.
- In `main.py`, the “LLM inference engine” is a stub that prints the prompt for inspection.

### Evaluation

- Runs **sequential loops** over the eval split.
- Pass naming and metrics are defined by the evaluation script (see `tests/README.md` for smoke-test details).
- When `raft.enabled` is true, the smoke script evaluates both tuned no-RAFT and tuned RAFT adapters under the same inference-time RAG toggle.
- The "with_rag" passes inject retrieved context at prompt time (uses `rag`); the "without_rag" passes force empty context regardless of how the training data was built.
- `TripodOrchestrator.execute("evaluate")` loads `evaluation.entrypoint` and runs `evaluation.evaluator`; if unset, it falls back to the stub evaluator.

For more detail on flows and feature flags, see `FLOW_OF_INFORMATION.md`.

## Setup

Create and activate a venv, then install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-train.txt
```

Notes:
- Training downloads models from Hugging Face and caches them under `~/.cache/huggingface/` by default (override with `HF_HOME`).
- QLoRA (4-bit) activates automatically when CUDA is available.

Dev tooling (format/lint/test helpers):

```bash
pip install -r requirements-dev.txt
```

## Running

### Prompt-only inference scaffold (raw backend, no LLM call)

Before running inference, you need a minimal RAG store (or disable RAG):

```bash
python -c "from main import TripodOrchestrator; TripodOrchestrator().execute('ingest', {'documents': ['Policy: keep temp < 80C; if vibration > 1.0 schedule maintenance.']})"
```

This writes `docs.jsonl` + `embeddings.npy` into `rag.vector_db_path`
(default `training_data/vectordb`). If you want prompt-only rendering with no
retrieval, set `rag.enabled: false` in your config. The first ingest downloads
the embedding model from Hugging Face.

Ensure your config uses the raw backend (`prompting.backend: "raw"`). The
default IoT config uses DSPy; if you keep it, you must configure a DSPy LM
before inference.

```bash
python3 main.py
```

This prints the final prompt after running RAG + prompting, so you can validate the “context → prompt” wiring.

### DSPy prompting

Set `prompting.backend: "dspy"` and configure a DSPy LM before running inference:

```python
import dspy

# Configure your LM before calling TripodOrchestrator.execute(...).
dspy.settings.configure(lm=...)
```

Tripod will return the DSPy prediction string. Use `prompting.backend: "raw"` for prompt-only rendering.

### Integration smoke test

```bash
python3 tests/smoke_e2e.py --n 6000 --eval-samples 200
```

This is a framework validation loop (data generation + RAFT/RAG + training + evaluation). It is a test scaffold, not the domain example.
For a domain example, see `pipelines/README.md`. See `tests/README.md` for flags and report artifacts.

## Interfacing With Tripod (Entry Points)

- `TripodOrchestrator.execute("prepare_train")`: build an SFT JSONL dataset from raw JSON/JSONL (`training.dataset_path`).
- `TripodOrchestrator.execute("train")`: run LoRA/QLoRA training (expects an SFT dataset at `training.dataset_path`).
- `TripodOrchestrator.execute("ingest", {"documents": [...], "target": "raft"})`: build the RAFT vector store for training-time retrieval.
- `TripodOrchestrator.execute("ingest", {"documents": [...], "target": "rag"})`: build the inference RAG vector store (default target is `rag`).
- `TripodOrchestrator.execute("inference", {"domain": "...", "sensor_data": {...}})`: runs RAG + prompting and prints the prompt (LLM call is intentionally pluggable).
- `TripodOrchestrator.execute("evaluate")`: dispatches to a registered evaluator (see `evaluation.entrypoint` + `evaluation.evaluator`); falls back to a stub logger if none is registered.

## Configuration (YAML)

Tripod is configured via YAML under `configs/`.
Use `configs/smoke_e2e_config.yaml` as a working template, and see `configs/README.md`
for the full config reference and runtime behavior.

## RAFT vs RAG

- **RAFT (`raft.*`)**: retrieval runs offline during dataset construction using the configured retriever. Retrieved text is concatenated into each training prompt before SFT so the model learns to use retrieved context at inference; the training loop itself never queries a retriever.
- **RAG (`rag.*`)**: inference-time retrieval that runs per request, injecting context into the prompt right before generation.
- In `tests/smoke_e2e.py`, RAFT toggles whether a second adapter is trained; the report compares RAFT vs no-RAFT adapters under identical inference-time RAG settings.
- You can point both modes at the same vector store or keep separate stores for controlled experiments.

## Adapting To Your Domain

Use the IoT pipeline walkthrough in `pipelines/README.md` as the concrete example; the smoke tests focus on framework validation. Your schema and metrics are always task-specific.

Typical steps:

1. **Define your output schema** (JSON, tool-call, etc) in your config (`prompting.system_prompt`).
2. **Provide training/eval data**:
   - If you keep the JSONL approach, generate `train.jsonl` / `test.jsonl` with your fields and ground truth.
3. **Update parsing + scoring**:
   - Align evaluation logic to your schema and tolerances in your pipeline or test harness.
4. **Decide what retrieval means for your domain (RAFT + RAG)**:
   - Ingest your docs/snippets with metadata, and filter/retrieve appropriately in your pipeline code.

## Example Pipelines

See `pipelines/README.md` for the IoT walkthrough and extension notes.

## Observability & Debugging

Where to look when something is off:

- **Training quality**
  - Logs: `core.training` + Trainer loss lines in `run.log`
  - If outputs stop being JSON, first verify the SFT delimiter (`ASSISTANT:`) is present in both training data and eval prompts.
- **Inference retrieval + prompt quality**
  - Logs: `core.rag` (“Retrieving top k…”) + `core.prompting` (“Constructing prompt…”)
  - Artifacts: vector store at `rag.vector_db_path` (`docs.jsonl`, `embeddings.npy`) and optionally `raft.vector_db_path`
  - Smoke: check `predictions/*.jsonl` → `rag_context` and `prompt` for retrieval gaps or schema drift.
- **Evaluation quality**
  - `summary.json` contains all run args (including holdout settings) and per-pass metrics.

If you want more verbosity, change `logging.basicConfig(level=logging.INFO, ...)` to `DEBUG` in the relevant entry script or pipeline.
