PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTHONPATH ?= src
INPUT_DIR ?= data/input
OUTPUT_DIR ?= output
MODEL ?= qwen3-vl:235b-cloud
VISION_MODEL ?= $(MODEL)
REASONING_MODEL ?= qwen3.5:397b-cloud
API_HOST ?= http://127.0.0.1:11434
LIMIT ?= 0
SYNC_TIMEOUT ?= 300
PROMPT_OVERRIDES ?=
LLM_POSTPROCESS ?= 1
TEMPORARY_CONTEXT ?=
ANALYSIS_DIR ?= $(OUTPUT_DIR)
TRAINING_OUTPUT ?= training_records.jsonl
TASK ?= write
WORKERS ?= 2
IMAGE_MAX_DIMENSION ?= 1600
IMAGE_JPEG_QUALITY ?= 88
STUDY_OUTPUT_DIR ?= artifacts/photo_corpus_study
EXPORT_DIR ?= artifacts/site_photo_analyze
PUBLISH_DIR ?= public

.PHONY: help run run-one export export-critique export-write export-interpret study-extract study-report site study package-site full clean

help:
	@printf '%s\n' \
	'Targets:' \
	'  make run               Run the full image pipeline' \
	'  make run-one           Run the pipeline on one image only (LIMIT=1)' \
	'  make study-extract     Build per-image optical metrics from pipeline outputs' \
	'  make study-report      Build corpus figures and reports' \
	'  make site              Build the static study website automatically' \
	'  make study             Run study-extract + study-report + site' \
	'  make package-site      Build the publishable static bundle' \
	'  make full              Run analysis + study + publishable bundle' \
	'  make export            Export training records (TASK=write|critique|interpret)' \
	'  make export-critique   Export critique training records' \
	'  make export-write      Export write training records' \
	'  make export-interpret  Export interpret training records' \
	'  make clean             Remove generated analysis outputs' \
	'' \
	'Variables overrideable:' \
	'  MODEL, VISION_MODEL, REASONING_MODEL, API_HOST, INPUT_DIR, OUTPUT_DIR, LIMIT, SYNC_TIMEOUT, PROMPT_OVERRIDES, LLM_POSTPROCESS, TEMPORARY_CONTEXT, TASK, TRAINING_OUTPUT, WORKERS, IMAGE_MAX_DIMENSION, IMAGE_JPEG_QUALITY, STUDY_OUTPUT_DIR, EXPORT_DIR, PUBLISH_DIR'

run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/ollama_scene_pipeline_v3.py \
		--input-dir $(INPUT_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--model "$(MODEL)" \
		--vision-model "$(VISION_MODEL)" \
		--reasoning-model "$(REASONING_MODEL)" \
		--api-host "$(API_HOST)" \
		--limit $(LIMIT) $(if $(filter 1 true yes on,$(LLM_POSTPROCESS)),--llm-postprocess,) \
		--sync-timeout $(SYNC_TIMEOUT) \
		--workers $(WORKERS) \
		--image-max-dimension $(IMAGE_MAX_DIMENSION) \
		--image-jpeg-quality $(IMAGE_JPEG_QUALITY) \
		$(if $(PROMPT_OVERRIDES),--prompt-overrides "$(PROMPT_OVERRIDES)",) $(if $(TEMPORARY_CONTEXT),--temporary-context "$(TEMPORARY_CONTEXT)",)

run-one:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/ollama_scene_pipeline_v3.py \
		--input-dir $(INPUT_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--model "$(MODEL)" \
		--vision-model "$(VISION_MODEL)" \
		--reasoning-model "$(REASONING_MODEL)" \
		--api-host "$(API_HOST)" \
		--limit 1 $(if $(filter 1 true yes on,$(LLM_POSTPROCESS)),--llm-postprocess,) \
		--sync-timeout $(SYNC_TIMEOUT) \
		--workers 1 \
		--image-max-dimension $(IMAGE_MAX_DIMENSION) \
		--image-jpeg-quality $(IMAGE_JPEG_QUALITY) \
		$(if $(PROMPT_OVERRIDES),--prompt-overrides "$(PROMPT_OVERRIDES)",) $(if $(TEMPORARY_CONTEXT),--temporary-context "$(TEMPORARY_CONTEXT)",)

study-extract:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/extract_photo_metrics.py \
		--input-dir $(INPUT_DIR) \
		--analysis-dir $(ANALYSIS_DIR) \
		--output-dir $(STUDY_OUTPUT_DIR)

study-report:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/build_corpus_report.py \
		--metrics-dir $(STUDY_OUTPUT_DIR)/metrics \
		--output-dir $(STUDY_OUTPUT_DIR)

site:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/build_study_site.py \
		--metrics-dir $(STUDY_OUTPUT_DIR)/metrics \
		--output-dir $(STUDY_OUTPUT_DIR)

study:
	$(MAKE) study-extract
	$(MAKE) study-report
	$(MAKE) site

package-site:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_export_bundle.py \
		--main-output $(OUTPUT_DIR) \
		--study-output $(STUDY_OUTPUT_DIR) \
		--export-dir $(EXPORT_DIR) \
		--publish-dir $(PUBLISH_DIR)

full:
	$(MAKE) run
	$(MAKE) study
	$(MAKE) package-site

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
	rm -rf $(STUDY_OUTPUT_DIR) $(EXPORT_DIR) $(PUBLISH_DIR)
