TECTONIC ?= tectonic
TECTONIC_BUNDLE ?= https://data1b.fullyjustified.net/tlextras-2022.0r0.tar
PYTHON ?= python3
PROJECT ?= ../tiga-lang
RUN ?= $(PROJECT)/output/paper-20260919
GF_OPT ?= gf-opt

.PHONY: all tectonic pdf iclr pdf-iclr pdf-all arxiv evidence figures check
all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex

tectonic:
	mkdir -p build
	$(TECTONIC) --bundle $(TECTONIC_BUNDLE) --keep-logs --keep-intermediates --outdir build main.tex

pdf: tectonic
	cp build/main.pdf tiga-lang.pdf

iclr:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/iclr2027 iclr2027.tex

pdf-iclr:
	mkdir -p build/iclr2027
	$(TECTONIC) --bundle $(TECTONIC_BUNDLE) --keep-logs --keep-intermediates --outdir build/iclr2027 iclr2027.tex
	cp build/iclr2027/iclr2027.pdf tiga-lang-iclr2027.pdf

pdf-all: pdf pdf-iclr

arxiv:
	$(PYTHON) tools/prepare_arxiv.py

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
	$(PYTHON) tools/build_supplement.py --check
	$(PYTHON) tools/build_page_sensitivity.py --check
	$(PYTHON) -m unittest discover -s tests
