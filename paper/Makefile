.PHONY: all main appendix markers clean

all: main appendix

main:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

appendix:
	latexmk -pdf -interaction=nonstopmode -halt-on-error appendix.tex

# Everything still to resolve before submission.
markers:
	@grep -n 'TODO\|SLOT\|\[\[FIG\|\[\[DIAG' *.tex sections/*.tex || echo "no markers left"

clean:
	latexmk -C main.tex appendix.tex
