"""gpr_fit_eb.py
Empirical-Bayes hyperparameters for Gaussian process regression with an
ARD squared exponential kernel. The model is
  y_t = f(x_t) + eps_t,  eps_t ~ N(0, sig2),  f ~ GP(0, k),
  k(x,x') = sigf2 * exp(-0.5 * sum_j (x_j - x'_j)^2 / lam_j^2).
The hyperparameters (sigf2, lam_1..lam_d, sig2) are set to the maximizers
of the log integrated likelihood, found by optimizing on the log scale (so
positivity is automatic) from a median-heuristic start with several
restarts. The objective is the local function gp_se_negloglike below,
minimized with Nelder-Mead (the counterpart of MATLAB's fminsearch).

Inputs:
  y    : length-n response
  X    : n-by-d input matrix
  opts : (optional) dict with keys 'nstart' (restarts, default 5) and
         'verbose' (print per-restart progress, default False)

Outputs:
  hp         : dict with keys 'sigf2' (scalar), 'lam' (length-d), 'sig2' (scalar)
  logintlike : maximized log integrated likelihood
  info       : dict with keys 'logp' (optimizer solution) and 'Dc'
               (per-dimension squared-distance list)
"""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize


def gpr_fit_eb(y, X, opts=None):
    if opts is None:
        opts = {}
    nstart = opts.get('nstart', 5)
    verbose = opts.get('verbose', False)

    n, d = X.shape

    # per-dimension squared distances; median-heuristic starting length scales
    Dc = []
    lam0 = np.zeros(d)
    for j in range(d):
        Dj = (X[:, j][:, None] - X[:, j][None, :])**2
        Dc.append(Dj)
        dv = np.sqrt(Dj[np.triu_indices(n, k=1)])
        lam0[j] = np.median(dv[dv > 0])
        if not (lam0[j] > 0):
            lam0[j] = 1

    vy = np.var(y, ddof=1)
    logp0 = np.concatenate(([np.log(vy)], np.log(lam0), [np.log(vy/4)]))
    optns = {'maxfev': 4000, 'maxiter': 4000, 'xatol': 1e-6, 'fatol': 1e-6}

    # optimize from the median-heuristic start plus perturbed restarts
    best_logp = logp0
    best_nll = np.inf
    for s in range(nstart):
        start = logp0.copy()
        if s > 0:
            start = logp0 + 0.7*np.random.randn(d+2)  # perturb on log scale
        res = minimize(gp_se_negloglike, start, args=(y, Dc),
                       method='Nelder-Mead', options=optns)
        lp, nll = res.x, res.fun
        if nll < best_nll:
            best_nll = nll
            best_logp = lp
        if verbose:
            print(f'  restart {s+1}: log int. lik. = {-nll:.4f}')

    # unpack hyperparameters
    hp = {'sigf2': np.exp(best_logp[0]),
          'lam': np.exp(best_logp[1:d+1]),   # length-d length scales
          'sig2': np.exp(best_logp[d+1])}
    logintlike = -best_nll
    info = {'logp': best_logp, 'Dc': Dc}
    return hp, logintlike, info


def gp_se_negloglike(logp, y, Dc):
    """Negative log integrated likelihood of the ARD squared exponential GP at
    the log-hyperparameters logp = [log(sigf2), log(lam_1..lam_d), log(sig2)].
    Integrating f out gives (y | theta) ~ N(0, Ksig) with Ksig = K + sig2*I, so
    the likelihood is evaluated through the Cholesky factor of Ksig. Returns a
    large value if Ksig stays numerically indefinite (so the optimizer backs
    off).

    Inputs:
      logp : log-hyperparameters [log(sigf2), log(lam_1..lam_d), log(sig2)]
      y    : length-n response
      Dc   : list of d per-dimension squared-distance matrices (each n-by-n)

    Output:
      nll  : negative log integrated likelihood at logp
    """
    n = y.size
    d = len(Dc)
    sigf2 = np.exp(logp[0])
    lam = np.exp(logp[1:d+1])
    sig2 = np.exp(logp[d+1])

    M = np.zeros((n, n))  # scaled squared distances
    for j in range(d):
        M = M + Dc[j]/(lam[j]**2)
    Ksig = sigf2*np.exp(-0.5*M) + sig2*np.eye(n)

    try:
        C = np.linalg.cholesky(Ksig)  # lower triangular
    except np.linalg.LinAlgError:
        C = None
    if C is None:  # escalate jitter if not pos. def.
        jit = 1e-8*(np.trace(Ksig)/n + 1)
        for _ in range(6):
            try:
                C = np.linalg.cholesky(Ksig + jit*np.eye(n))
                break
            except np.linalg.LinAlgError:
                jit = jit*10
        if C is None:
            return 1e10

    a = solve_triangular(C, y, lower=True)  # a'a = y' Ksig^{-1} y
    logdet = 2*np.sum(np.log(np.diag(C)))  # log|Ksig|
    nll = 0.5*(a @ a) + 0.5*logdet + 0.5*n*np.log(2*np.pi)
    return nll
