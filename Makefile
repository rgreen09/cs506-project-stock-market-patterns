.DEFAULT_GOAL := help

# Override to choose interpreter, e.g. `make PYTHON=python run`
PYTHON ?= python3

.PHONY: help install run clean

help:
	@echo "make install  Install Python deps from requirements.txt (PYTHON=$(PYTHON))"
	@echo "make run      Run pipeline via: $(PYTHON) -m src.main run --all"
	@echo "make clean    No-op placeholder (kept for interface completeness)"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run: install
	$(PYTHON) -m src.main run --all

clean:
	@echo "Nothing to clean (placeholder target)."

