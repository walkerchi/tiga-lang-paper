TECTONIC ?= tectonic
TECTONIC_BUNDLE ?= https://data1b.fullyjustified.net/tlextras-2022.0r0.tar
PYTHON ?= python3
PROJECT ?= ../graphforgev2
RUN ?= $(PROJECT)/output/paper-20260919
GF_OPT ?= gf-opt

.PHONY: all tectonic evidence figures check
all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex

tectonic:
	mkdir -p build
	$(TECTONIC) --bundle $(TECTONIC_BUNDLE) --keep-logs --keep-intermediates --outdir build main.tex

evidence:
	$(PYTHON) tools/export_evidence.py --project $(PROJECT)

figures:
	$(PYTHON) tools/build_results.py --project $(PROJECT) --run $(RUN) --gf-opt $(GF_OPT)

check:
	$(PYTHON) tools/export_evidence.py --project $(PROJECT) --check
	$(PYTHON) tools/build_results.py --check
	$(PYTHON) tools/build_runtime_results.py --check
	$(PYTHON) tools/build_billion_results.py --check
	$(PYTHON) tools/build_three_questions.py --check
	$(PYTHON) -m unittest discover -s tests
