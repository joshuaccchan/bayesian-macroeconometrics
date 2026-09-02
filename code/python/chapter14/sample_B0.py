"""sample_B0.py
Draws the contemporaneous impact matrix B_0 in the
order-invariant VAR-SV model row by row using the Waggoner-Zha-Villani
scheme. Conditional on residuals E = Y - X*A and log-volatilities h,
the i-th row b_i of B_0 has full conditional density
    p(b_i | y, beta, b_{-i}, h_{i,1:T}) propto
        |B_0|^T * exp(-0.5*(b_i - bhat_i)' Kbi (b_i - bhat_i)),
with Kbi = V_b^{-1} + E' Omega_{h_i}^{-1} E and
bhat_i = Kbi^{-1} V_b^{-1} b_{0,i}. The determinant prefactor is
handled by an orthonormal change of variables: the resulting density
in xi has one absolute-normal component along the cofactor direction
and n-1 independent normals. The absolute-normal draw is generated
inline via the two-component normal mixture approximation of
Appendix C of Villani (2009).

Inputs:
  B0:     n x n current draw of B_0
  E:      T x n residual matrix Y - X*A
  h:      T x n matrix of log-volatilities; h[:, i] is for equation i
  b0_B0:  n x n prior mean of B_0; row i is the prior mean of b_i
  iV_B0:  n x n prior precision of each row b_i (assumed common)

Output:
  B0:     n x n updated draw, with positive diagonal entries
"""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import expit


def sample_B0(B0, E, h, b0_B0, iV_B0):
    T, n = E.shape
    B0 = B0.copy()

    for i in range(n):
        iOmega_i = np.exp(-h[:, i])
        Kbi = iV_B0 + E.T @ (iOmega_i[:, None]*E)
        bhat_i = np.linalg.solve(Kbi, iV_B0 @ b0_B0[i, :])
        Ci = np.linalg.cholesky(Kbi/T)          # so K_bi = T * Ci * Ci'

        # unit vector orthogonal to rows of B_0 except row i
        B0_no_i = np.delete(B0, i, axis=0)
        _, _, Vh_svd = np.linalg.svd(B0_no_i)
        w = Vh_svd[-1, :]

        # orthonormal basis (v_1, v_2, ..., v_n) with v_1 along Ci^{-1}*w
        v1 = solve_triangular(Ci, w, lower=True)
        v1 = v1/np.linalg.norm(v1)
        Q, _ = np.linalg.qr(np.column_stack((v1, np.eye(n))))
        Vmat = Q
        Vmat[:, 0] = v1

        # means xi_hat_j = v_j' * Ci' * bhat_i
        xi_hat = Vmat.T @ (Ci.T @ bhat_i)

        # sample xi: xi_1 ~ AN(xi_hat_1, 1/T); xi_j ~ N(xi_hat_j, 1/T) j>1.
        # The AN draw uses Villani (2009)'s two-component mixture: pick a
        # mode of the density |z|^T * exp(-T(z - mu)^2/2) and propose with
        # the local Laplace variance.
        xi = np.zeros(n)
        mu_an = xi_hat[0]
        disc = np.sqrt(mu_an**2 + 4)
        # expit(-x) = 1/(1 + exp(x)) avoids overflow for large mu_an*T
        if np.random.rand() < expit(-2*mu_an*T):  # weight on the negative mode
            m_an = (mu_an - disc)/2
        else:
            m_an = (mu_an + disc)/2
        xi[0] = m_an + np.sqrt(m_an**2/(T*(1 + m_an**2)))*np.random.randn()
        xi[1:] = xi_hat[1:] + (1/np.sqrt(T))*np.random.randn(n-1)
        bi_new = solve_triangular(Ci.T, Vmat @ xi, lower=False)

        # sign normalization: enforce positive diagonal of B_0
        if bi_new[i] < 0:
            bi_new = -bi_new
        B0[i, :] = bi_new
    return B0
