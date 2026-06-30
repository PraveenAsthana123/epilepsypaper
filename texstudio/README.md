# TeXstudio build folder — self-contained

Open any of the three papers in TeXstudio and build (pdfLaTeX → BibTeX → pdfLaTeX ×2):

- `q1_noRGAIG_2col.tex` — **primary submission**: Patient-Independent EEG Seizure Detection (no RGAIG)
- `q1_full_2col.tex` — with-RGAIG / deployment variant
- `review_full_2col.tex` — systematic review (uses `_scopus/_clinical/_reviz/_bibmatrix/_narrative/_sections.tex`)

Everything needed is in this folder: `references.bib`, the review `_*.tex` includes, and `images/`.
No external paths. (`*.aux/*.log/*.pdf` etc. are build artifacts.)
