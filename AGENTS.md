# Repository Guidelines
This repository hosts a modular Tripod framework for LLM control tasks (IoT example), combining three legs—training (LoRA/PEFT), retrieval, and prompt engineering—configured via YAML.

## Project Structure & Module Organization
- `main.py`: High-level orchestrator wiring the legs and loading config.
- `core/training.py`, `core/rag.py`, `core/prompting.py`: Leg implementations; extend or swap logic inside each file.
- `core/vectordb.py`: Local vector store adapter.
- `core/evaluation.py`: Evaluator registry + stub hook.
- `core/config.py`, `core/base.py`: Pydantic config schemas and shared base class.
- `configs/smoke_e2e_config.yaml`: End-to-end smoke config.
- `configs/iot_domain_config.yaml`: Example production config (IoT-themed).
- `configs/README.md`: Config-to-runtime mapping.
- `pipelines/README.md`, `pipelines/iot/`: Example pipelines and walkthroughs.
- `tests/README.md`: Smoke-test passes, metrics, and report artifacts.
- `training_data/`: Local root for datasets, vectordb, adapters, and reports. Create as needed; keep large assets out of Git.
- Outputs are configurable; the IoT config uses `../artifacts`/`../experiments`, while smoke defaults live under `training_data/`.

## Setup, Build, and Development Commands
- Python 3.10+ recommended; install deps: `pip install pydantic pyyaml`.
- Dev tooling (format/lint/test): `pip install -r requirements-dev.txt`.
- Inference demo (uses `configs/iot_domain_config.yaml`; requires RAG store + DSPy LM unless you switch to `prompting.backend: "raw"`):  
  ```bash
  python -c "from main import TripodOrchestrator; TripodOrchestrator().execute('inference', {'input_data': {'temp': 78.5, 'vibration': 1.2}})"
  ```
- Training placeholder flow:  
  ```bash
  python -c "from main import TripodOrchestrator; TripodOrchestrator().execute('train')"
  ```
- Training dataset prep (JSON/JSONL → SFT):  
  ```bash
  python -c "from main import TripodOrchestrator; TripodOrchestrator().execute('prepare_train')"
  ```
- Ingestion placeholder (document list, defaults to RAG):  
  ```bash
  python -c "from main import TripodOrchestrator; TripodOrchestrator().execute('ingest', {'documents': ['sample note']})"
  ```
- Ingestion targeting RAFT:  
  ```bash
  python -c "from main import TripodOrchestrator; TripodOrchestrator().execute('ingest', {'documents': ['sample note'], 'target': 'raft'})"
  ```
- Evaluation hook (uses `evaluation.entrypoint`/`evaluation.evaluator` or falls back to stub):  
  ```bash
  python -c "from main import TripodOrchestrator; TripodOrchestrator().execute('evaluate')"
  ```
- To use a different config file, pass `TripodOrchestrator('configs/your_config.yaml')`.
- Local smoke (downloads tiny HF models): `python tests/acceptance/smoke_local.py`
- End-to-end smoke (GPU recommended): `python tests/acceptance/smoke_e2e.py`
- Docker (optional): `docker build -t tripod .` then `docker run --rm --gpus all tripod python main.py`

## Coding Style & Naming Conventions
- PEP 8 with type hints; config surfaces stay typed via Pydantic models.
- Classes in `CamelCase`; functions/variables in `snake_case`; YAML keys remain lower_snake.
- Use `logging` (preconfigured) for diagnostics; reserve `print` for intentional user-facing output (e.g., rendered prompt).
- Prefer structural pattern matching (`match`/destructuring) for parsing and branching when it clarifies intent.
- Keep code slim: avoid inline comments and verbose metadata unless it reduces cognitive load.
- No wildcard imports; prefer explicit imports.
- Prefer explicit config over hidden defaults; validate at the boundary (raise early on invalid inputs).

## Testing Guidelines
- Place tests under `tests/` as `test_*.py`; run with `pytest -q`.
- Cover TripodConfig parsing, prompt rendering substitutions, RAG/RAFT selection knobs (`rag.*`, `raft.*`), and training leg toggles.
- When adding modes or legs, include an integration-style test that exercises `TripodOrchestrator.execute`.

## Commit & Pull Request Guidelines
- Commits: imperative, present-tense summaries (e.g., `Add rag ingestion stub`); keep scope focused.
- PRs: include what changed, sample command/output (prompt snippet or log line), and note config path or dependency updates.
- Link issues/tasks when available; add screenshots/logs only when they clarify behavior.

## Security & Configuration Tips
- Keep credentials/endpoints out of configs; the sample S3 URI is illustrative.
- Do not commit datasets or adapters; store them externally and reference via config paths (`training_data/`, `../artifacts/`, `../experiments/`).
- When sharing configs, scrub environment-specific paths or secrets.
