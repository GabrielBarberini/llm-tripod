# Acceptance tests

This folder contains end-to-end smoke tests that validate the full Tripod wiring
(dataset generation → retrieval → prompting → training → evaluation). These are
not intended to run as part of `make test`.

## End-to-end smoke (`tests/acceptance/smoke_e2e.py`)

Run:

```bash
python3 tests/acceptance/smoke_e2e.py --n 6000 --eval-samples 200
```

## Local smoke (`tests/acceptance/smoke_local.py`)

Run:

```bash
python3 tests/acceptance/smoke_local.py
```

