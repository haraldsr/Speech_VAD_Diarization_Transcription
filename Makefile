# Configuration
ENV_NAME = vdt

.PHONY: help install install-dev install-conda gen-lock lint format clean app

help:
	@echo "Available commands:"
	@echo "  make install         - Install with lockfile (auto-detects GPU/CPU)"
	@echo "  make install-dev     - Install from requirements.txt (for development)"
	@echo "  make install-conda   - Install using Conda only (slower)"
	@echo ""
	@echo "  make app             - Launch the Streamlit GUI (auto-detects conda/venv)"
	@echo ""
	@echo "  make gen-lock        - Generate lockfile (auto-names based on GPU detection)"
	@echo ""
	@echo "  make lint            - Run linting (flake8, isort, black, mypy)"
	@echo "  make format          - Format code (isort + black)"
	@echo "  make clean           - Remove build artifacts and cache"

app:
	@echo "Launching Streamlit GUI..."
	@if conda info --envs 2>/dev/null | grep -qE "^$(ENV_NAME)[[:space:]]"; then \
		echo "✓ Using conda env: $(ENV_NAME)"; \
		bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(ENV_NAME) && streamlit run app_gui.py"; \
	elif [ -f ".venv/bin/activate" ]; then \
		echo "✓ Using venv: .venv"; \
		bash -c "source .venv/bin/activate && streamlit run app_gui.py"; \
	elif [ -f "venv/bin/activate" ]; then \
		echo "✓ Using venv: venv"; \
		bash -c "source venv/bin/activate && streamlit run app_gui.py"; \
	else \
		echo "⚠ No conda env '$(ENV_NAME)' or venv found — running with current Python."; \
		streamlit run app_gui.py; \
	fi

install:
	@echo "Auto-detecting GPU availability..."
	@if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then \
		LOCK_FILE="requirements-lock-uv-gpu.txt"; \
		echo "✓ GPU detected - using $$LOCK_FILE"; \
	else \
		LOCK_FILE="requirements-lock-uv-cpu.txt"; \
		echo "✗ No GPU detected - using $$LOCK_FILE"; \
	fi; \
	if [ ! -f "$$LOCK_FILE" ]; then \
		echo "⚠️  Lockfile $$LOCK_FILE not found!"; \
		echo "Run 'make install-dev' to install from requirements.txt instead."; \
		exit 1; \
	fi; \
	if command -v mamba >/dev/null 2>&1; then \
		echo "Using mamba (faster)..."; \
		mamba env create -f environment-minimal.yml -n $(ENV_NAME) || true; \
	else \
		echo "Using conda..."; \
		conda env create -f environment-minimal.yml -n $(ENV_NAME) || true; \
	fi; \
	echo "Installing Python packages with UV..."; \
	command -v uv >/dev/null 2>&1 || { echo "UV not found. Installing..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }; \
	bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(ENV_NAME) && uv pip install -r $$LOCK_FILE && pip install -e ."; \
	echo "✓ Installation complete! Activate with: conda activate $(ENV_NAME)"

install-dev:
	@echo "Installing from requirements.txt (development mode)..."
	@if command -v mamba >/dev/null 2>&1; then \
		echo "Using mamba (faster)..."; \
		mamba env create -f environment-minimal.yml -n $(ENV_NAME) || true; \
	else \
		echo "Using conda..."; \
		conda env create -f environment-minimal.yml -n $(ENV_NAME) || true; \
	fi
	@echo "Installing Python packages with UV..."
	@command -v uv >/dev/null 2>&1 || { echo "UV not found. Installing..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(ENV_NAME) && uv pip install -r requirements.txt && pip install -e ."
	@echo "✓ Installation complete! Activate with: conda activate $(ENV_NAME)"

gen-lock:
	@echo "Generating lockfile from active environment..."
	@chmod +x scripts/generate_uv_lock.sh
	@if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then \
		LOCK_FILE="requirements-lock-uv-gpu.txt"; \
		echo "✓ GPU detected - creating $$LOCK_FILE"; \
	else \
		LOCK_FILE="requirements-lock-uv-cpu.txt"; \
		echo "✗ No GPU detected - creating $$LOCK_FILE"; \
	fi; \
	scripts/generate_uv_lock.sh $$LOCK_FILE; \
	echo "✓ Lockfile generated: $$LOCK_FILE"

install-conda:
	@echo "Installing with Conda/Mamba (all dependencies)..."
	@if command -v mamba >/dev/null 2>&1; then \
		echo "Using mamba (faster)..."; \
		mamba env create -f environment.yml; \
	else \
		echo "Using conda..."; \
		conda env create -f environment.yml; \
	fi
	@echo "Installing package in editable mode..."
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(ENV_NAME) && pip install -e ."
	@echo "✓ Installation complete! Activate with: conda activate $(ENV_NAME)"

lint:
	@echo "Running linting checks..."
	flake8 .
	isort --check --diff .
	black --check .
	mypy .
	@echo "✓ Linting complete"

format:
	@echo "Formatting code..."
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
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleanup complete"
