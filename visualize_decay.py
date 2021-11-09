from numpy import *
import matplotlib.pyplot as plt
import sys

"""
2018 Lukas Herrmann, SAM, ETH Zurich
"""


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("need arguments: Lmax")
        exit(0)

    Lmax = int(sys.argv[1])

    s = 2 ** (Lmax + 2)

    # compute reference

    ref = []

    for j in range(4):
        ref.append(1)

    for l in range(1, Lmax + 1):
        for j in range(2 ** (l + 1)):
            ref.append(2 ** (-1.5 * l))

    b_01 = loadtxt("betalk_L%d_lambda10.txt" % Lmax)
    #b_01 = (10 * b_01) ** (1.5)
    b_005 = loadtxt("betalk_L%d_lambda5.txt" % Lmax)
    #b_005 = (10 * b_005) ** (1.5)
    b_001 = loadtxt("betalk_L%d_lambda1.txt" % Lmax)
    #b_001 = (10 * b_001) ** (1.5)

    ss = linspace(0, s - 1, s)

    ref_str = '$b_{\rm ref}$'
    b_01_str = '$\lambda=0.1$'
    b_005_str = '$\lambda=0.05$'
    b_001_str = '$\lambda=0.01$'

    plt.loglog(ss, ref, 'k-', label=r"$b_{\rm ref}$")
    plt.loglog(ss, b_01, 'k--', label=r"$\lambda=0.1$")
    plt.loglog(ss, b_005, 'k:', label=r"$\lambda=0.05$")
    plt.loglog(ss, b_001, 'k-.', label=r"$\lambda=0.01$")

    plt.grid(True)

    plt.xlabel(r"$j(\ell,k)$", fontsize=15)
    plt.ylabel(r"$\|\psi^{1}_{\ell,k}\|_{L^\infty(\mathbb{T}^1)}$", fontsize=15)

    plt.legend(loc="lower left", fontsize=12)

    plt.tight_layout()

    plt.savefig("decay_psi.pdf")

    plt.show()