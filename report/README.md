# Report — IEEE 5-Page Format

## Compile

```bash
cd report
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Output: `report/main.pdf` (target: 5 pages including references).

## Dependencies

The skeleton uses the `IEEEtran` document class. Install via either:

- **Arch / Manjaro:** `sudo pacman -S texlive-publishers`
- **Debian / Ubuntu:** `sudo apt-get install texlive-publishers`
- **macOS (BasicTeX):** `sudo tlmgr install ieeetran`
- **Manual drop-in:** place `IEEEtran.cls` from <https://www.ctan.org/pkg/ieeetran> into this directory.

## Figures

Plots are produced by `bench/plot.py` from the JSON cells in `bench/results/`.
Run:

```bash
uv run python -m bench.plot
ln -sf ../../bench/figures report/figures   # if plot.py writes to bench/figures
# or simply: cp -r ../bench/figures-out/* report/figures/
```

The current skeleton has `[placeholder for measured speedup]` and bracketed
figure-insertion notes; replace these with `\includegraphics{figures/<name>}`
in Section V (Results) once the plots exist.

## Section page budget (target = 5 pages incl. references)

| Section | Target |
|---|---|
| Abstract + Introduction | 0.5 |
| Background | 0.75 |
| Design Choices | 1.25 |
| Implementation | 1.0 |
| Results | 1.0 |
| Discussion + Future Work | 0.25 |
| References | 0.25 |
