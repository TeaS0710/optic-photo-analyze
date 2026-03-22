PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTHONPATH ?= src
INPUT_DIR ?= data/input
OUTPUT_DIR ?= output
MODEL ?= gemma3:latest
API_HOST ?= http://127.0.0.1:11434
LIMIT ?= 0
SYNC_TIMEOUT ?= 300
PROMPT_OVERRIDES ?=
LLM_POSTPROCESS ?= 0
TEMPORARY_CONTEXT ?=
ANALYSIS_DIR ?= $(OUTPUT_DIR)
TRAINING_OUTPUT ?= training_records.jsonl
TASK ?= write

.PHONY: help run run-one export export-critique export-write export-interpret clean

help:
	@printf '%s\n' \
	'Targets:' \
	'  make run               Run the full image pipeline' \
	'  make run-one           Run the pipeline on one image only (LIMIT=1)' \
	'  make export            Export training records (TASK=write|critique|interpret)' \
	'  make export-critique   Export critique training records' \
	'  make export-write      Export write training records' \
	'  make export-interpret  Export interpret training records' \
	'  make clean             Remove generated analysis outputs' \
	'' \
	'Variables overrideable:' \
	'  MODEL, API_HOST, INPUT_DIR, OUTPUT_DIR, LIMIT, SYNC_TIMEOUT, PROMPT_OVERRIDES, LLM_POSTPROCESS, TEMPORARY_CONTEXT, TASK, TRAINING_OUTPUT'

run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/ollama_scene_pipeline_v3.py \
		--input-dir $(INPUT_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--model "$(MODEL)" \
		--api-host "$(API_HOST)" \
		--limit $(LIMIT) $(if $(filter 1 true yes on,$(LLM_POSTPROCESS)),--llm-postprocess,) \
		--sync-timeout $(SYNC_TIMEOUT) $(if $(PROMPT_OVERRIDES),--prompt-overrides "$(PROMPT_OVERRIDES)",) $(if $(TEMPORARY_CONTEXT),--temporary-context "$(TEMPORARY_CONTEXT)",)

run-one:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/ollama_scene_pipeline_v3.py \
		--input-dir $(INPUT_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--model "$(MODEL)" \
		--api-host "$(API_HOST)" \
		--limit 1 $(if $(filter 1 true yes on,$(LLM_POSTPROCESS)),--llm-postprocess,) \
		--sync-timeout $(SYNC_TIMEOUT) $(if $(PROMPT_OVERRIDES),--prompt-overrides "$(PROMPT_OVERRIDES)",) $(if $(TEMPORARY_CONTEXT),--temporary-context "$(TEMPORARY_CONTEXT)",)

export:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/export_training_records.py \
		--analysis-dir $(ANALYSIS_DIR) \
		--output $(TRAINING_OUTPUT) \
		--task $(TASK) $(if $(PROMPT_OVERRIDES),--prompt-overrides "$(PROMPT_OVERRIDES)",)

export-critique:
	$(MAKE) export TASK=critique TRAINING_OUTPUT=training_critique.jsonl

export-write:
	$(MAKE) export TASK=write TRAINING_OUTPUT=training_write.jsonl

export-interpret:
	$(MAKE) export TASK=interpret TRAINING_OUTPUT=training_interpret.jsonl

clean:
	rm -f $(OUTPUT_DIR)/*.analysis.json $(OUTPUT_DIR)/*.report.md $(OUTPUT_DIR)/manifest.json
