# Configuration
ENV_NAME = wp1

.PHONY: help install install-uv install-conda install-hybrid clean lint format test

help:
	@echo "Available commands:"
	@echo "  make install-uv      - Install using UV (local machine with FFmpeg, requires sudo)"
	@echo "  make install-conda   - Install using Conda/Mamba (all dependencies)"
	@echo "  make install-hybrid  - Conda for FFmpeg + UV for packages (RECOMMENDED for HPC)"
	@echo "  make install         - Alias for install-hybrid"
	@echo "  make lint           - Run linting (ruff)"
	@echo "  make format         - Format code (ruff + isort)"
	@echo "  make test           - Run tests"
	@echo "  make clean          - Remove build artifacts and cache"
	@echo "  make clean-all      - Remove everything including venv/conda env"

install: install-hybrid

install-hybrid:
	@echo "Installing with Conda (FFmpeg) + UV (Python packages)..."
	@if command -v mamba >/dev/null 2>&1; then \
		echo "Using mamba (faster)..."; \
		mamba env create -f environment-minimal.yml -n $(ENV_NAME); \
	else \
		echo "Using conda..."; \
		conda env create -f environment-minimal.yml -n $(ENV_NAME); \
	fi
	@echo "Installing Python packages with UV..."
	@command -v uv >/dev/null 2>&1 || { echo "UV not found. Installing..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(ENV_NAME) && uv pip install -r requirements-lock-uv.txt && pip install -e ."
	@echo "✓ Installation complete! Activate with: conda activate $(ENV_NAME)"

install-uv:
	@echo "Installing with UV..."
	@command -v ffmpeg >/dev/null 2>&1 || { echo "⚠️  WARNING: FFmpeg not found. Install it with:"; echo "  Ubuntu/Debian: sudo apt install -y ffmpeg"; echo "  macOS: brew install ffmpeg"; exit 1; }
	@command -v uv >/dev/null 2>&1 || { echo "UV not found. Installing..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	uv venv --python 3.10
	. .venv/bin/activate && uv pip install -r requirements-lock-uv.txt
	. .venv/bin/activate && pip install -e .
	@echo "✓ Installation complete! Activate with: source .venv/bin/activate"

install-conda:
	@echo "Installing with Conda/Mamba..."
	@if command -v mamba >/dev/null 2>&1; then \
		echo "Using mamba (faster)..."; \
		mamba env create -f environment.yml; \
	else \
		echo "Using conda..."; \
		conda env create -f environment.yml; \
	fi
	@echo "Installing package in editable mode..."
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate wp1_pyannote && pip install -e ."
	@echo "✓ Installation complete! Activate with: conda activate wp1_pyannote"

lint:
	@echo "Running linting checks..."
	@command -v flake8 >/dev/null 2>&1 || { echo "Installing flake8..."; pip install flake8; }
	@command -v black >/dev/null 2>&1 || { echo "Installing black..."; pip install black; }
	@command -v isort >/dev/null 2>&1 || { echo "Installing isort..."; pip install isort; }
	@command -v mypy >/dev/null 2>&1 || { echo "Installing mypy..."; pip install mypy; }
	flake8 .
	isort --check --diff .
	black --check .
	mypy .
	@echo "✓ Linting complete"

format:
	@echo "Formatting code..."
	@command -v isort >/dev/null 2>&1 || { echo "Installing isort..."; pip install isort; }
	@command -v black >/dev/null 2>&1 || { echo "Installing black..."; pip install black; }
	isort .
	black .
	@echo "✓ Formatting complete"

clean:
	@echo "Cleaning build artifacts and cache..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleanup complete"
