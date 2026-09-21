# Technical-report submission preparation

This repository contains two manuscripts. The arXiv package contains only the
attributed technical report, not the anonymous ICLR workshop draft.

```bash
make pdf TECTONIC=/path/to/tectonic
make arxiv
```

`build/arxiv/tiga-lang-source.zip` contains the recursive dependencies of
`main.tex`: active sections, code listings, figures and `references.bib`.
`build/arxiv/manifest.json` records the file and archive checksums outside the
upload package. Repeated packaging of unchanged sources is deterministic.
The full report PDF, workshop files, historical sections, raw measurements,
credentials, build logs and unrelated images are excluded.

## Author checklist

- Independently review the scientific claims, references, measured records and
  AI use statement. Automated checks do not replace author responsibility.
- Upload the source ZIP and select `main.tex` with the LaTeX/PDFLaTeX processor.
  Bibliography processing uses standard BibTeX and `plain`; no custom build
  command, internet access or shell escape is needed.
- Inspect the PDF produced by arXiv, including equations, citations, figures
  and both appendices. Local TeX Live 2026 and Tectonic builds are useful checks,
  not a guarantee of identical output on arXiv's available TeX Live versions.
- Use title `Tiga: Compiling Graph Message Passing at Scale`, author
  `Mingyuan Chi`, affiliation `Independent Researcher`, and the abstract from
  `main.tex`. Add the actual page count from the final platform preview.
- Select the subject category based on the compiler/systems contribution;
  `cs.PL` is a candidate, with `cs.DC` a possible cross-list. Final category and
  any endorsement requirement are determined during submission.
- Select the paper's distribution license explicitly. The compiler's Apache
  2.0 license does not choose an arXiv paper license. Consider future venue
  policies before selection; this tool makes no license choice.
- Preserve a commit-specific artifact link in submission metadata. If a later
  workshop version substantially overlaps, check arXiv's replacement/overlap
  rules and the workshop's publication policies instead of creating an
  automatically duplicated arXiv submission.

Building this package does not create an arXiv account, seek endorsement,
accept a license, submit a manuscript, or claim acceptance.

Official guidance: [submission](https://info.arxiv.org/help/submit/index.html),
[TeX sources](https://info.arxiv.org/help/submit_tex.html),
[moderation and AI disclosure](https://info.arxiv.org/help/moderation/index.html),
[licenses](https://info.arxiv.org/help/license/index.html).
