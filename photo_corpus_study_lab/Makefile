PYTHON ?= python3
PYTHONPATH ?= src
INPUT_DIR ?= ../data/input
ANALYSIS_DIR ?= ../output
OUTPUT_DIR ?= ./artifacts

.PHONY: help extract report site all clean

help:
	@printf '%s\n' \
	'Targets:' \
	'  make extract           Extract optical, color and semantic study metrics' \
	'  make report            Build corpus figures, families and reports' \
	'  make site              Build the static website from collected artifacts' \
	'  make all               Run the full study pipeline' \
	'  make clean             Remove generated study artifacts' \
	'' \
	'Variables:' \
	'  INPUT_DIR, ANALYSIS_DIR, OUTPUT_DIR'

extract:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/extract_photo_metrics.py \
		--input-dir $(INPUT_DIR) \
		--analysis-dir $(ANALYSIS_DIR) \
		--output-dir $(OUTPUT_DIR)

report:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/build_corpus_report.py \
		--metrics-dir $(OUTPUT_DIR)/metrics \
		--output-dir $(OUTPUT_DIR)

site:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/build_study_site.py \
		--metrics-dir $(OUTPUT_DIR)/metrics \
		--output-dir $(OUTPUT_DIR)

all:
	$(MAKE) extract
	$(MAKE) report
	$(MAKE) site

clean:
	rm -rf $(OUTPUT_DIR)
