from P1_FEM import *
from pwlin_prewav_precomputed import *
import numpy as np

from scipy.stats import norm

Phiinv = norm.ppf

"""
2018 Lukas Herrmann, SAM, ETH Zurich

The intgrand classes for the single 
and the difference term in the MLQMC-FEM scheme
"""


class integrand_single:
    """
    Integrand class for lowest level of multilevel sampling
    """

    def __init__(self, kappa, theta, alpha, l, Lmatrices, s_leaf, shift):
        self.kappa = kappa
        self.theta = theta
        self.alpha = alpha
        self.l = l
        self.Lmatrices = Lmatrices
        self.s_leaf = s_leaf
        self.shift = shift
        self.N = self.s_leaf
        self.mesh = gen_mesh((0, 1), self.N, "equidistant")

    def __call__(self, y):
        shift_level = self.shift[: self.s_leaf]
        y = np.mod(y + shift_level, 1)
        coeffs = GRF_prewav_to_hat_precomputed(
            self.Lmatrices,
            self.theta,
            self.kappa,
            self.alpha,
            self.l,
            Phiinv(y[: self.s_leaf]),
        )
        u = solve_pde(self.N, coeffs, self.fct, self.mesh)
        u = u - sum(u) * 2 ** (-self.l - 2)
        return eval_fct_pt(u, self.mesh, 0.7)

    @staticmethod
    def fct(x):
        return np.sin(2 * np.pi * x)


class integrand_difference:
    """
    Integrand class for higher level of multilevel sampling
    """

    def __init__(
        self, kappa, theta, alpha, l, Lmatrices, s_leaf, s_level, Sused, shift
    ):
        self.kappa = kappa
        self.theta = theta
        self.alpha = alpha
        self.l = l
        self.Lmatrices = Lmatrices
        self.s_leaf = s_leaf
        self.s_level = s_level
        self.shift = shift
        self.N = self.s_leaf
        self.N_level = self.s_level
        self.Sused = Sused
        self.mesh = gen_mesh((0, 1), self.N, "equidistant")
        self.mesh_level = gen_mesh((0, 1), self.N_level, "equidistant")

    def __call__(self, y):
        shift_level = self.shift[self.Sused : self.Sused + self.s_leaf]
        assert len(y) == len(shift_level)
        y = mod(y + shift_level, 1)
        # transform from [0,1] to R
        # coeffs are coefficients of constant function on each interval
        coeffs = GRF_prewav_to_hat_precomputed(
            self.Lmatrices,
            self.theta,
            self.kappa,
            self.alpha,
            self.l,
            Phiinv(y[: self.s_leaf]),
        )
        coeffs_level = GRF_prewav_to_hat_precomputed(
            self.Lmatrices,
            self.theta,
            self.kappa,
            self.alpha,
            self.l - 1,
            Phiinv(y[: self.s_level]),
        )
        u = solve_pde(self.N, coeffs, self.fct, self.mesh)
        u = u - sum(u) * 2 ** (-self.l - 2)
        u_level = solve_pde(self.N_level, coeffs_level, self.fct, self.mesh_level)
        u_level = u_level - sum(u_level) * 2 ** (-self.l - 1)
        # evaluate QoI and return
        return eval_fct_pt(u, self.mesh, 0.7) - eval_fct_pt(
            u_level, self.mesh_level, 0.7
        )

    @staticmethod
    def fct(x):
        return np.sin(2 * np.pi * x)
