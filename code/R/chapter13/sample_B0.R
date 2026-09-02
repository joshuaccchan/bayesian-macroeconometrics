# sample_B0.R
# Draws the contemporaneous impact matrix B_0 in the
# order-invariant VAR-SV model row by row using the Waggoner-Zha-Villani
# scheme. Conditional on residuals E = Y - X*A and log-volatilities h,
# the i-th row b_i of B_0 has full conditional density
#     p(b_i | y, beta, b_{-i}, h_{i,1:T}) propto
#         |B_0|^T * exp(-0.5*(b_i - bhat_i)' Kbi (b_i - bhat_i)),
# with Kbi = V_b^{-1} + E' Omega_{h_i}^{-1} E and
# bhat_i = Kbi^{-1} V_b^{-1} b_{0,i}. The determinant prefactor is
# handled by an orthonormal change of variables: the resulting density
# in xi has one absolute-normal component along the cofactor direction
# and n-1 independent normals. The absolute-normal draw is generated
# inline via the two-component normal mixture approximation of
# Appendix C of Villani (2009).
#
# Inputs:
# B0:     n x n current draw of B_0
# E:      T x n residual matrix Y - X*A
# h:      T x n matrix of log-volatilities; h[, i] is for equation i
# b0_B0:  n x n prior mean of B_0; row i is the prior mean of b_i
# iV_B0:  n x n prior precision of each row b_i (assumed common)
#
# Output:
# B0:     n x n updated draw, with positive diagonal entries

sample_B0 <- function(B0, E, h, b0_B0, iV_B0) {

    T <- nrow(E)
    n <- ncol(E)

    for (i in 1:n) {
        # Kbi = iV_B0 + E' Omega_{h_i}^{-1} E, Omega_{h_i} = diag(exp(h[, i]))
        Kbi <- iV_B0 + crossprod(E, E/exp(h[, i]))
        bhat_i <- as.numeric(solve(Kbi, iV_B0 %*% b0_B0[i, ]))
        Ci <- t(chol(Kbi/T))                    # so K_bi = T * Ci * Ci'

        # unit vector orthogonal to rows of B_0 except row i
        # (last left singular vector of B0_no_i', i.e. MATLAB's V_svd(:, end))
        B0_no_i <- B0[-i, , drop = FALSE]
        w <- svd(t(B0_no_i), nu = n, nv = 0)$u[, n]

        # orthonormal basis (v_1, v_2, ..., v_n) with v_1 along Ci^{-1}*w
        v1 <- forwardsolve(Ci, w)
        v1 <- v1/sqrt(sum(v1^2))
        Q <- qr.Q(qr(cbind(v1, diag(n))))
        Vmat <- Q
        Vmat[, 1] <- v1

        # means xi_hat_j = v_j' * Ci' * bhat_i
        xi_hat <- as.numeric(crossprod(Vmat, crossprod(Ci, bhat_i)))

        # sample xi: xi_1 ~ AN(xi_hat_1, 1/T); xi_j ~ N(xi_hat_j, 1/T) j>1.
        # The AN draw uses Villani (2009)'s two-component mixture: pick a
        # mode of the density |z|^T * exp(-T(z - mu)^2/2) and propose with
        # the local Laplace variance.
        xi <- numeric(n)
        mu_an <- xi_hat[1]
        disc <- sqrt(mu_an^2 + 4)
        if (runif(1) < 1/(1 + exp(2*mu_an*T))) {  # weight on the negative mode
            m_an <- (mu_an - disc)/2
        } else {
            m_an <- (mu_an + disc)/2
        }
        xi[1] <- m_an + sqrt(m_an^2/(T*(1 + m_an^2)))*rnorm(1)
        for (j in 2:n) {
            xi[j] <- xi_hat[j] + (1/sqrt(T))*rnorm(1)
        }
        # t(Ci) is upper triangular, so this is MATLAB's (Ci')\(Vmat*xi)
        bi_new <- as.numeric(backsolve(t(Ci), as.numeric(Vmat %*% xi)))

        # sign normalization: enforce positive diagonal of B_0
        if (bi_new[i] < 0) {
            bi_new <- -bi_new
        }
        B0[i, ] <- bi_new
    }
    B0
}
