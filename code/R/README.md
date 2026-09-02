# R code

R ports of the MATLAB chapter code, one folder per chapter mirroring
`code/matlab/` script-for-script (`linreg_t.m` -> `linreg_t.R`). Each folder is
self-contained: it carries the data and helper functions its scripts need. To run
anything, make the chapter folder the working directory and run the script, e.g.

```
cd chapter03
Rscript linreg_t.R
```

Requirements: R 4.x with the `Matrix` package. Nothing else is needed — `Matrix`
is a recommended package that ships with R, and everything else uses base R.

Notes on the translation:

- R matches MATLAB on the conventions that matter most here: 1-based indexing,
  column-major storage (so `matrix(x, m, n)` is `reshape(x, m, n)`), `var`/`sd`
  with the n-1 denominator, and `chol()` returning the upper factor.
- Quantiles use `type = 5`, which reproduces MATLAB's `quantile` exactly; R's
  default (`type = 7`) does not.
- Gamma draws and densities always name their arguments
  (`rgamma(n, shape = a, scale = b)`), because R's second positional argument is
  the *rate*, whereas MATLAB's `gamrnd`/`gampdf` take the *scale*.
- The precision-based samplers use sparse `Matrix` objects throughout: `chol()`
  on a sparse symmetric matrix returns the upper factor with no fill-reducing
  permutation, so `x_hat + solve(chol(K), rnorm(n))` is a direct transcription of
  MATLAB's `x_hat + chol(K,'lower')'\randn(n,1)`.
- MATLAB's `iwishrnd` (Statistics Toolbox) has no base-R counterpart, so the
  files that need an inverse-Wishart draw define one locally from the Bartlett
  decomposition, with the same parameterization.
- Random draws use R's global RNG with the same seeds as the MATLAB scripts. The
  streams differ from MATLAB's, so results are reproducible on the R side but
  agree with the MATLAB output only up to Monte Carlo error.
- MATLAB `save`/`load` of `.mat` results files becomes `saveRDS`/`readRDS` with
  `.rds` files carrying the same variable names.
- Following the book's notation, scripts use `T` for the sample size and, in
  `TVPVAR_Primiceri.R`, `F` for the companion matrix. These shadow R's `T`/`F`
  abbreviations for `TRUE`/`FALSE`, so the code always spells `TRUE` and `FALSE`
  out; keep that habit if you edit these scripts interactively.
- As in the MATLAB folders, `chapter10/SP500.csv` is not shipped; run
  `get_SP500_data.R` once to create it.
- R is slower than MATLAB on the scalar-loop-heavy samplers — typically 5-10x, so
  the longest chapters (the chapter 13-14 forecasting exercises in particular)
  are best run with reduced settings unless you can leave them overnight.

Per-script validation against the MATLAB output is tracked in `code/PARITY.md`:
deterministic quantities were checked at fixed inputs (agreement to ~1e-8 or
better) and MCMC output was compared at matched settings within Monte Carlo
error. Ports not marked there should be treated as drafts.
