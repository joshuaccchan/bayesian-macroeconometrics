# python code

Python ports of the MATLAB chapter code, one folder per chapter mirroring
`code/matlab/` script-for-script (`linreg_t.m` -> `linreg_t.py`). Each folder is
self-contained: it carries the data and helper modules its scripts need. To run
anything, make the chapter folder the working directory and run the script, e.g.

```
cd chapter03
python linreg_t.py
```

Requirements: Python 3.10+ with numpy, scipy, pandas, and matplotlib.

Notes on the translation:

- Random draws use numpy's global RNG with the same seeds as the MATLAB
  scripts. The streams differ from MATLAB's, so results are reproducible on
  the Python side but agree with the MATLAB output only up to Monte Carlo error.
- MATLAB `save`/`load` of `.mat` results files becomes `np.savez`/`np.load`
  with `.npz` files carrying the same variable names.
- As in the MATLAB folders, `chapter10/SP500.csv` is not shipped; run
  `get_SP500_data.py` once to create it. `chapter14/forecast_largeVAR.py`
  translates the MATLAB `parfor` as a plain loop, so its multi-hour runtime is
  longer still in Python.

Per-script validation against the MATLAB output is tracked in `code/PARITY.md`:
deterministic quantities were checked at fixed inputs (agreement to ~1e-8 or
better), and MCMC output was compared at matched settings within Monte Carlo
error. Ports not marked there should be treated as drafts.
