from P1_FEM import *
from pwlin_prewav_precomputed import *
from helpers import *
from numpy import *
import matplotlib.pyplot as plt

"""
2018 Lukas Herrmann, SAM, ETH Zurich
"""



#resolution of plot
l = 11


for ell0 in range(5, 9):
    #set parameters
    lambdaC = 0.1
    alpha = 2.
    sigma0 = 1.0

    kappa, theta = compute_GRF_params(alpha, sigma0, lambdaC)

    N = 2 ** (l + 2)

    y_1 = zeros(N)
    ell = ell0
    k0 = 2 ** (ell - 1) - ell
    y_1[4 + 2 ** (ell + 1) + k0 - 1] = 1

    #load Lmatrices
    Lmatrices = []
    for k in range(l + 1):
        if k <= 1:
            L = load("L_matrices/L_matrix%d.npy" % (k))
            Lmatrices.append(L)
        if k > 1:
            L = sparse.load_npz("L_matrices/L_matrix%d.npz" % (k))
            Lmatrices.append(L)

    u = GRF_prewav_to_hat_precomputed(Lmatrices, theta, kappa, alpha, l, y_1)

    xx = linspace(0, 1, N + 1)
    str = ['k--', 'k-', 'k:', 'k-.']
    plt.semilogy(xx[:-1], abs(u[:-1]), str[(ell0) % 4], label=r"$\ell=%d,k=%d$" % (ell, k0))
    plt.legend(loc="upper right", fontsize=12)

plt.grid(True)


plt.xlabel(r"$x$", fontsize=15)
plt.ylabel(r"$|\psi^{1}_{\ell,k}(x)|$", fontsize=15)

plt.tight_layout()
plt.savefig("psi_several.pdf")

plt.show()