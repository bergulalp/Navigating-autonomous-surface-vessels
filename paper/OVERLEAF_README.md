# 5.9.1. MARIN — Overleaf project

Upload **MARIN_Overleaf_v4.zip** using Overleaf's **New Project → Upload Project** flow.
The ZIP contains `main.tex` at its root, all included `.tex` files, `references.bib`, and
all required vector PDF figures. It does not require Python, external shell commands,
third-party course slides or the simulation dataset to compile.

Select `main.tex` as the main document and **pdfLaTeX** as compiler. Overleaf normally
runs BibTeX automatically. If citations are unresolved after a first build, recompile
from scratch. The extracted ZIP was also compiled locally with LaTeX/BibTeX.

For a local TeX installation, run:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Or run `pdflatex main`, `bibtex main`, and `pdflatex main` twice.

The two-column article layout is a scientific manuscript format, not a claim that a
particular journal has accepted the paper or that all journal submission requirements
are met. Replace the class/style only after choosing a journal and checking its current
author instructions. The project identity is **5.9.1. MARIN**.

Official [Overleaf upload instructions](https://docs.overleaf.com/managing-projects-and-files/uploading-a-project).
