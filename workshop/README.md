# ICLR 2027 workshop manuscript

[Read the draft PDF](../tiga-lang-iclr2027.pdf) · [LaTeX entry point](../iclr2027.tex)

This is a separate, shorter manuscript derived from the technical report, not
an ICLR main-conference submission. No workshop has been selected and no paper
has been submitted or accepted. The
[ICLR 2027 workshop call](https://iclr.cc/Conferences/2027/CallForWorkshops)
lists proposal decisions for November 29, 2026 and a suggested contribution
deadline of February 1, 2027; individual workshops set their own requirements.

## Relationship to the report

- `main.tex` and `tiga-lang.pdf` remain the complete, attributed technical report.
- `iclr2027.tex` and `tiga-lang-iclr2027.pdf` build this focused draft.
- Both manuscripts share `references.bib` and the same benchmark figure PDFs.
- The short manuscript focuses on relation-aware execution and memory capacity.
  Distributed execution is retained as a negative result, not a speedup claim.
- No new experiments are implied by reorganizing existing measurements.

The official ICLR 2027 template is preserved under `styles/iclr2027/` with source
URL and checksums. The wrapper replaces the template's submission-status wording
with an explicit draft notice. Template margins, fonts and spacing are unchanged.
The main-conference page limit is not a confirmed workshop limit.

## Build

```bash
make pdf-iclr TECTONIC=/path/to/tectonic
make pdf-all TECTONIC=/path/to/tectonic
```

Alternatively, `make iclr` uses pdfLaTeX through latexmk. Builds use the repository
root as the working directory; no GPU or benchmark execution is required.

## Before submission

1. Select an accepted workshop and check its scope, template, length, anonymization,
   preprint, concurrent-submission and archival policies. Replace the draft status
   only when an actual submission exists and its instructions specify the wording.
2. Prioritize a same-search materialized/fused ablation and a real-data EdgeNN
   forward/backward application. Add first-call JIT and amortization measurements.
   A matched bounded-memory baseline is needed for comparative offload claims.
   The full technical report now contains a fixed-topology consumer comparison,
   a differentiated Bunny-geometry operator, first-call/cache measurements and
   a page-size sweep. They are not yet incorporated in this short manuscript;
   generated-search ablation, an end-to-end learning application and a matched
   bounded-memory peer remain distinct evidence requirements.
3. Review the AI use statement with the human author. It reflects assistance in
   implementation, experiments, literature work and writing; it does not assert
   that a human has already independently reviewed every generated artifact.
4. Prepare and inspect an anonymous artifact package if required. The draft PDF
   omits the author, email, logo and identifying repository links, but the public
   Tiga name and prior report remain discoverable. The full repository and raw
   benchmark archives are not an anonymized submission package.
5. Rebuild, validate claims against archived records, inspect the resulting PDF,
   and confirm the eventual workshop's page limit. Uploading/submitting is a
   separate action; building a PDF does not submit it.

The technical report and workshop manuscript may share material, but subsequent
publication and disclosure must follow the selected venues' policies.
