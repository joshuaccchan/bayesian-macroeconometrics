# Bayesian Macroeconometrics: Methods and Applications

MATLAB code and sample chapters for the book *Bayesian Macroeconometrics: Methods and
Applications* by [Joshua C. C. Chan](https://joshuachan.org).

**Sample PDF** — [`sample/BayesMacroBook_sample.pdf`](sample/BayesMacroBook_sample.pdf)
contains the title page, the complete table of contents, the preface, and Chapters 1-4
in full. Posted with the publisher's permission. The PDF is copyrighted material and is
*not* covered by the code license below.

## Contents of the book

Each chapter links to its MATLAB code; the R and Python ports mirror the same folder
names under `code/R` and `code/python`.

**Part I - Core Bayesian Modeling and Assessment**

1. [Foundations of Bayesian Econometrics](code/matlab/chapter01)
2. [Normal Linear Regression](code/matlab/chapter02)
3. [Linear Regression with General Errors](code/matlab/chapter03)
4. [Mixture Models](code/matlab/chapter04)
5. [Bayesian Model Comparison](code/matlab/chapter05)

**Part II - Bayesian Computation and High-Dimensional Methods**

6. [Foundations of Bayesian Computation](code/matlab/chapter06)
7. [Bayesian Shrinkage Methods](code/matlab/chapter07)
8. [Bayesian Nonparametric Methods](code/matlab/chapter08)

**Part III - Bayesian Time-Series Models**

9. [Linear Gaussian State Space Models](code/matlab/chapter09)
10. [Stochastic Volatility Models](code/matlab/chapter10)
11. [Factor Models](code/matlab/chapter11)
12. [Vector Autoregressions](code/matlab/chapter12)
13. [Time-Varying Vector Autoregressions](code/matlab/chapter13)
14. [Large VARs with Stochastic Volatility](code/matlab/chapter14)

Appendices: Common Probability Distributions; Matrix Algebra.

## Code

### Layout

`code/matlab/chapter01` ... `chapter14` contain the MATLAB code for every chapter's
examples, applications, and figures. Each folder is **self-contained**: it carries the
data and helper functions its scripts need.

### Running it

Set MATLAB's working directory to the chapter folder and run the script — no
installation or path setup is needed.

```matlab
cd code/matlab/chapter02
linreg_NIG_predictive
```

### Requirements

A recent MATLAB with the Statistics and Machine Learning Toolbox.
`chapter14/forecast_largeVAR.m` (an 8-model recursive forecasting exercise with a
runtime of several hours) additionally uses the Parallel Computing Toolbox.

### R and Python

`code/R` and `code/python` mirror the MATLAB folders script-for-script
(`linreg_t.m` → `linreg_t.R` / `linreg_t.py`), and each script's output has been
validated against the MATLAB version — per-script status is in
[`code/PARITY.md`](code/PARITY.md). The Python code needs Python 3.10+ with numpy,
scipy, pandas and matplotlib; the R code needs only R 4.x and the `Matrix` package
that ships with it.

### One data file to download

In `chapter10`, run `get_SP500_data.m` once to create `SP500.csv`. The S&P 500 series
is proprietary to S&P Dow Jones Indices and cannot be redistributed; the script's
header has the details.

## Data sources

All datasets are small CSV extracts constructed from public sources, shipped inside the
chapter folders that use them:

| Data | Source |
|---|---|
| US macro series (PCE deflator, CPI, GDP, unemployment, fed funds, NFCI, ...) | FRED (St. Louis Fed) |
| `FRED-MD.csv` (chapters 11, 14) | FRED-MD monthly database - McCracken and Ng (2016), *JBES* 34(4) |
| `gpr_uncertainty_data.csv` (chapter 8) | Includes the macro uncertainty index of Jurado, Ludvigson and Ng (2015), *AER* 105(3) |
| `oil_SVAR_data.csv` (chapter 12) | Oil-market variables incl. the global real activity index of Kilian (2009), *AER* 99(3) |
| `SP500.csv` (chapter 10) | Not shipped - run `get_SP500_data.m` (FRED, S&P DJI terms) |

## Citation

If you use this code, please cite the book:

> Chan, J. C. C. *Bayesian Macroeconometrics: Methods and Applications*. Chapman & Hall/CRC, forthcoming.

```bibtex
@book{chan-bmar,
  author    = {Chan, Joshua C. C.},
  title     = {Bayesian Macroeconometrics: Methods and Applications},
  publisher = {Chapman \& Hall/CRC},
  note      = {Forthcoming}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The sample PDF and all book
content remain copyrighted and are excluded from that license.
