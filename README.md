# Bayesian Macroeconometrics: Methods and Applications

MATLAB code and sample chapters for the book *Bayesian Macroeconometrics: Methods and
Applications* by [Joshua C. C. Chan](https://joshuachan.org).

**Sample PDF** — [`sample/BayesMacroBook_sample.pdf`](sample/BayesMacroBook_sample.pdf)
contains the title page, the complete table of contents, the preface, and Chapters 1-4
in full. Posted with the publisher's permission. The PDF is copyrighted material and is
*not* covered by the code license below.

## Contents of the book

**Part I - Core Bayesian Modeling and Assessment**

1. Foundations of Bayesian Econometrics
2. Normal Linear Regression
3. Linear Regression with General Errors
4. Mixture Models
5. Bayesian Model Comparison

**Part II - Bayesian Computation and High-Dimensional Methods**

6. Foundations of Bayesian Computation
7. Bayesian Shrinkage Methods
8. Bayesian Nonparametric Methods

**Part III - Bayesian Time-Series Models**

9. Linear Gaussian State Space Models
10. Stochastic Volatility Models
11. Factor Models
12. Vector Autoregressions
13. Time-Varying Vector Autoregressions
14. Large VARs with Stochastic Volatility

Appendices: Common Probability Distributions; Matrix Algebra.

## Code

`code/matlab/chapter1` ... `chapter14` contain the MATLAB code for every chapter's
examples, applications, and figures. Each folder is **self-contained**: it carries the
data and helper functions its scripts need. To run anything, set MATLAB's working
directory to the chapter folder and run the script.

Requirements: a recent MATLAB with the Statistics and Machine Learning Toolbox.
`chapter14/forecast_largeVAR.m` (an 8-model recursive forecasting exercise with a
runtime of several hours) additionally uses the Parallel Computing Toolbox.

One data file must be downloaded rather than shipped: in `chapter10`, run
`get_SP500_data.m` once to create `SP500.csv` (the S&P 500 series is proprietary to
S&P Dow Jones Indices and cannot be redistributed; the script's header has details).

**R and Python ports are planned.** They will mirror the MATLAB folders
script-for-script, starting with Chapters 1-4; see `code/PARITY.md` for status.

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
