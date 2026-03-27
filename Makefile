.PHONY: help setup setup-dev pull-models serve serve-bg stop \
       test test-ef test-audio \
       gen-data-ef preview-data \
       finetune-ef eval export validate-export \
       convert-gguf quantize-gguf ollama-load ollama-test \
       lint format clean

PYTHON ?= python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
UVICORN := $(VENV)/bin/uvicorn
OLLAMA_HOST ?= http://localhost:11434
MODEL := phi4-mini:latest

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────────────

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

setup: $(VENV)/bin/activate ## Install core dependencies
	$(PIP) install -e "."
	@echo "\n✓ Core dependencies installed"

setup-dev: $(VENV)/bin/activate ## Install all dependencies (dev + audio + finetune)
	$(PIP) install -e ".[dev,audio,finetune]"
	@echo "\n✓ All dependencies installed"

setup-audio: $(VENV)/bin/activate ## Install audio dependencies
	$(PIP) install -e ".[audio]"
	@echo "\n✓ Audio dependencies installed"

pull-models: ## Pull Phi-4-mini and check Ollama is running
	@command -v ollama >/dev/null 2>&1 || { echo "Error: ollama not installed. See https://ollama.ai"; exit 1; }
	ollama pull $(MODEL)
	@echo "\n✓ Model pulled: $(MODEL)"

# ── Serving ──────────────────────────────────────────────────────────────────

serve: setup ## Start the gateway server (foreground)
	EH_OLLAMA_HOST=$(OLLAMA_HOST) $(UVICORN) src.gateway.app:app \
		--host 0.0.0.0 --port 8000 --reload

serve-bg: setup ## Start the gateway server (background)
	EH_OLLAMA_HOST=$(OLLAMA_HOST) $(UVICORN) src.gateway.app:app \
		--host 0.0.0.0 --port 8000 --reload &
	@echo "Gateway running on http://localhost:8000"

stop: ## Stop background gateway server
	@pkill -f "uvicorn src.gateway.app:app" 2>/dev/null || echo "No server running"

# ── Testing ──────────────────────────────────────────────────────────────────

test: setup-dev ## Run all tests
	$(VENV)/bin/pytest tests/ -v

test-ef: setup ## Test executive function model (baseline eval)
	$(PY) -m src.eval.run --model ef

test-audio: setup-audio ## Test audio round-trip (mic → transcription → response → speaker)
	$(PY) -m src.audio.test_pipeline

# ── Data Generation ─────────────────────────────────────────────────────────

gen-data-ef: setup ## Generate synthetic executive dysfunction training data
	$(PY) -m src.data.generate --output data/generated/ef --mode combo --count 10 --seed 42

preview-data: ## Preview generated training data samples
	$(PY) -m src.data.preview

# ── Fine-Tuning ──────────────────────────────────────────────────────────────

finetune-ef: setup-dev ## Fine-tune Phi-4-mini for executive dysfunction support
	@rm -rf unsloth_compiled_cache
	TORCHDYNAMO_DISABLE=1 $(PY) -m src.finetune.train --config configs/finetune_ef.yaml

eval: setup-dev ## Run evaluation suite (baseline vs fine-tuned)
	$(PY) -m src.eval.run --compare

export: ## Merge LoRA adapters → safetensors (models/executive-helper-ef/)
	$(PY) -m src.finetune.export

validate-export: ## Validate exported model dir before GGUF conversion
	$(PY) scripts/validate_export.py models/executive-helper-ef

convert-gguf: ## Convert merged safetensors → GGUF bf16
	$(PY) llama.cpp/convert_hf_to_gguf.py models/executive-helper-ef \
		--outfile models/executive-helper-ef.bf16.gguf --outtype bf16
	@echo ""
	@echo "✓ GGUF bf16 written to models/executive-helper-ef.bf16.gguf"
	@echo "  Next: make quantize-gguf"

quantize-gguf: ## Quantize bf16 GGUF → Q4_K_M
	llama.cpp/llama-quantize \
		models/executive-helper-ef.bf16.gguf \
		models/executive-helper-ef.Q4_K_M.gguf Q4_K_M
	@echo ""
	@echo "✓ Quantized to models/executive-helper-ef.Q4_K_M.gguf"
	@echo "  Next: make ollama-load"

ollama-load: ## Load quantized GGUF into Ollama
	cd models && ollama create executive-helper-ef -f Modelfile.executive-helper-ef
	@echo ""
	@echo "✓ Model loaded into Ollama as executive-helper-ef"
	@echo "  Next: make ollama-test"

ollama-test: ## Quick smoke test of the Ollama model
	ollama run executive-helper-ef "I need to clean my apartment but I can't start."

# ── Code Quality ─────────────────────────────────────────────────────────────

lint: setup-dev ## Run linter
	$(VENV)/bin/ruff check src/ tests/

format: setup-dev ## Auto-format code
	$(VENV)/bin/ruff format src/ tests/
	$(VENV)/bin/ruff check --fix src/ tests/

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
