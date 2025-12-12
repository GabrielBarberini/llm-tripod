# Repository Guidelines
This repository hosts a modular Tripod framework for Industrial IoT control, combining three legs—training (LoRA/PEFT), retrieval-augmented generation, and prompt engineering—configured via YAML.

## Project Structure & Module Organization
- `main.py`: High-level orchestrator wiring the legs and loading config.
- `core/training.py`, `core/rag.py`, `core/prompting.py`: Leg implementations; extend or swap logic inside each file.
- `core/config.py`, `core/base.py`: Pydantic config schemas and shared base class.
- `configs/iot_domain_config.yaml`: Domain config (paths, hyperparameters, prompts). Place any hardcoded values or additional YAMLs here.
- `training_data/`: Local root for vectordb/test sets referenced by config. Create as needed; keep large assets out of Git.
- Outputs (adapters/experiments) default to `../artifacts`/`../experiments` to keep the repo lean; adjust per environment.

## Setup, Build, and Development Commands
- Python 3.10+ recommended; install deps: `pip install pydantic pyyaml`.
- Inference demo (uses `configs/iot_domain_config.yaml`):  
  ```bash
  python -c "from main import TripodOrchestrator; TripodOrchestrator().execute('inference', {'sensor_data': {'temp': 78.5, 'vibration': 1.2}})"
  ```
- Training placeholder flow:  
  ```bash
  python -c "from main import TripodOrchestrator; TripodOrchestrator().execute('train')"
  ```
- Ingestion placeholder (document list):  
  ```bash
  python -c "from main import TripodOrchestrator; TripodOrchestrator().execute('ingest', {'documents': ['sample note']})"
  ```
- To use a different config file, pass `TripodOrchestrator('configs/your_config.yaml')`.
- Smoke cross-section (downloads tiny HF models): `python scripts/smoke_cross_section.py`
- Docker (optional): `docker build -t tripod .` then `docker run --rm --gpus all tripod python main.py`

## Coding Style & Naming Conventions
- PEP 8 with type hints; config surfaces stay typed via Pydantic models.
- Classes in `CamelCase`; functions/variables in `snake_case`; YAML keys remain lower_snake.
- Use `logging` (preconfigured) for diagnostics; reserve `print` for intentional user-facing output (e.g., rendered prompt).

## Testing Guidelines
- Place tests under `tests/` as `test_*.py`; run with `pytest -q`.
- Cover TripodConfig parsing, prompt rendering substitutions, RAG selection knobs (`top_k`, `strategy`, thresholds), and training leg toggles.
- When adding modes or legs, include an integration-style test that exercises `TripodOrchestrator.execute`.

## Commit & Pull Request Guidelines
- Commits: imperative, present-tense summaries (e.g., `Add rag ingestion stub`); keep scope focused.
- PRs: include what changed, sample command/output (prompt snippet or log line), and note config path or dependency updates.
- Link issues/tasks when available; add screenshots/logs only when they clarify behavior.

## Security & Configuration Tips
- Keep credentials/endpoints out of configs; the sample S3 URI is illustrative.
- Do not commit datasets or adapters; store them externally and reference via config paths (`training_data/`, `../artifacts/`, `../experiments/`).
- When sharing configs, scrub environment-specific paths or secrets.
