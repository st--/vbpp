# Copyright (C) PROWLER.io 2017 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import itertools as it

import numpy as np
import scipy.special as ss


# These are references implementations of the psi-matrices
# they contain no broadcasting and only work in low dimensions

def calc_Ψ_matrix(zs, γ, sqrtα, T):
    α = sqrtα**2
    M = len(zs)
    Ψ = np.ones((M, M))
    inv_sqrt_α = 1/sqrtα
    for r in range(len(T)):
        mult = - 0.5 * np.sqrt(np.pi * α[r])
        Tmin, Tmax = T[r]
        for i, j in it.product(range(M), range(M)):
            z1 = zs[i, r]
            z2 = zs[j, r]
            zm = (z1 + z2)*0.5
            zd_sqr = (z1 - z2)**2

            Ψ[i, j] *= mult * np.exp(- zd_sqr / (4*α[r])) * (
                ss.erf((zm - Tmax)*inv_sqrt_α[r]) - ss.erf((zm - Tmin)*inv_sqrt_α[r])
            )
    return γ**2 * Ψ


def calc_Ψ_matrix_2(zs, γ, sqrtα, T):
    α = sqrtα**2
    M = len(zs)
    Ψ = np.zeros((M, M))
    inv_sqrt_α = 1/sqrtα
    for i, j in it.product(range(M), range(M)):
        v = 1.0
        for r in range(len(T)):
            v *= - 0.5 * np.sqrt(np.pi * α[r])
            Tmin, Tmax = T[r]
            z1 = zs[i, r]
            z2 = zs[j, r]
            zm = (z1 + z2)*0.5
            zd_sqr = (z1 - z2)**2

            v *= np.exp(- zd_sqr / (4*α[r])) * (
                ss.erf((zm - Tmax)*inv_sqrt_α[r]) - ss.erf((zm - Tmin)*inv_sqrt_α[r])
            )
        Ψ[i, j] = v
    return γ**2 * Ψ
