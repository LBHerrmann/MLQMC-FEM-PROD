from P1_FEM import *
from pwlin_prewav_precomputed import *
from helpers import *
from numpy import *

"""
2018 Lukas Herrmann, SAM, ETH Zurich

Computes decay of the representation system of the GRF
and saves this as file. This may be loaded to be visualized.
input:
    Lmax : maximal level in wavelet like expansion
    sigma0 : parameter that controls variance of GRF  
    lambdaC : correlation length of GRF
"""

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("need arguments: Lmax sigma0 lambdaC")
        exit(0)

    Lmax = int(sys.argv[1])
    sigma0 = float(sys.argv[2])
    lambdaC = float(sys.argv[3])

    print("Lmax=", Lmax, ", sigma0=", sigma0, ", lambdaC=", lambdaC)

    # hard code order of covariance operator
    alpha = 2.0

    kappa, theta = compute_GRF_params(alpha, sigma0, lambdaC)

    Lmatrices = []

    for k in range(14):
        if k <= 1:
            L = load("L_matrices/L_matrix%d.npy" % (k))
            Lmatrices.append(L)
        if k > 1:
            L = sparse.load_npz("L_matrices/L_matrix%d.npz" % (k))
            Lmatrices.append(L)

    vals = []

    # number of degress of freedom
    N = 2 ** (Lmax + 2)

    mesh = gen_mesh((0, 1), N, "equidistant")
    coeffs = 1
    coeffs2 = lambda x: kappa**2 * (1 + 0.5 * sin(2 * pi * x))

    # assemble the system matrices
    A0 = assemble_stiffness_periodicBC(mesh, coeffs)
    M0 = assemble_mass_periodicBC(mesh, coeffs2)
    S = A0 + M0

    for j in range(4):
        y = zeros(N)
        y[j] = 1
        y_t = block_chol_M_precomputed(Lmatrices, Lmax, y)

        RHS = prewav_to_hat(Lmax, y_t)
        u = zeros(N + 1)
        u[0:N] = spsolve(S, RHS)
        u[N] = u[0]
        u = theta * u
        b = max(u)
        vals.append(b)

    for l in range(1, Lmax + 1):

        for j in range(2 ** (l + 1)):
            y = zeros(N)
            y[2 ** (l + 1) + j] = 1
            y_t = block_chol_M_precomputed(Lmatrices, Lmax, y)

            RHS = prewav_to_hat(Lmax, y_t)
            u = zeros(N + 1)
            u[0:N] = spsolve(S, RHS)
            u[N] = u[0]
            u = theta * u
            b = max(u)
            vals.append(b)

    outfile = "betalk_L%d_lambda%d.txt" % (Lmax, lambdaC * 100)
    print("writing to file:", outfile)
    S0 = "".join(["%1.14f\n" % v for v in vals])
    with open(outfile, "w") as f:
        f.write(S0)
