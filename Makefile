VENV_BIN ?= ./.venv/bin
PYTHON ?= python3
DOCKER ?= docker
IMAGE ?= tripod

ifneq (,$(wildcard $(VENV_BIN)/black))
BLACK := $(VENV_BIN)/black
else
BLACK := $(PYTHON) -m black
endif

ifneq (,$(wildcard $(VENV_BIN)/ruff))
RUFF := $(VENV_BIN)/ruff
else
RUFF := $(PYTHON) -m ruff
endif

ifneq (,$(wildcard $(VENV_BIN)/pytest))
PYTEST := $(VENV_BIN)/pytest
else
PYTEST := $(PYTHON) -m pytest
endif

format: black ruff

black:
	$(BLACK) ./core ./scripts ./main.py || true
	if [ -d ./tests ]; then $(BLACK) ./tests || true; else echo "No ./tests directory."; fi

ruff:
	$(RUFF) check --fix ./core ./scripts ./main.py || true
	if [ -d ./tests ]; then $(RUFF) check --fix ./tests || true; else echo "No ./tests directory."; fi

test:
	if [ -d ./tests ]; then $(PYTEST) .; else echo "No ./tests directory."; fi

build:
	$(DOCKER) build -t $(IMAGE) .

.PHONY: black ruff format test build
