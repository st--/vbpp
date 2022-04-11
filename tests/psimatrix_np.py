# Copyright (C) Secondmind Ltd 2017
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools as it

import numpy as np
import scipy.special as ss


# These are references implementations of the psi-matrices
# they contain no broadcasting and only work in low dimensions


def calc_Ψ_matrix(zs, γ, sqrtα, T):
    α = sqrtα**2
    M = len(zs)
    Ψ = np.ones((M, M))
    inv_sqrt_α = 1 / sqrtα
    for r in range(len(T)):
        mult = -0.5 * np.sqrt(np.pi * α[r])
        Tmin, Tmax = T[r]
        for i, j in it.product(range(M), range(M)):
            z1 = zs[i, r]
            z2 = zs[j, r]
            zm = (z1 + z2) * 0.5
            zd_sqr = (z1 - z2) ** 2

            Ψ[i, j] *= (
                mult
                * np.exp(-zd_sqr / (4 * α[r]))
                * (ss.erf((zm - Tmax) * inv_sqrt_α[r]) - ss.erf((zm - Tmin) * inv_sqrt_α[r]))
            )
    return γ**2 * Ψ


def calc_Ψ_matrix_2(zs, γ, sqrtα, T):
    α = sqrtα**2
    M = len(zs)
    Ψ = np.zeros((M, M))
    inv_sqrt_α = 1 / sqrtα
    for i, j in it.product(range(M), range(M)):
        v = 1.0
        for r in range(len(T)):
            v *= -0.5 * np.sqrt(np.pi * α[r])
            Tmin, Tmax = T[r]
            z1 = zs[i, r]
            z2 = zs[j, r]
            zm = (z1 + z2) * 0.5
            zd_sqr = (z1 - z2) ** 2

            v *= np.exp(-zd_sqr / (4 * α[r])) * (
                ss.erf((zm - Tmax) * inv_sqrt_α[r]) - ss.erf((zm - Tmin) * inv_sqrt_α[r])
            )
        Ψ[i, j] = v
    return γ**2 * Ψ
