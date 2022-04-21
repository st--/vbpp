# Copyright (C) 2022 ST John
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

import numpy as np
import pytest
import tensorflow as tf
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import SquaredExponential
import gpflow
from vbpp import VBPP

rng = np.random.RandomState(0)


class Data:
    domain = np.array([[0.0, 10.0]])
    events = rng.uniform(0, 10, size=20)[:, None]
    Z = np.linspace(0, 10, 17)[:, None]
    Xtest = np.linspace(-2, 12, 37)[:, None]


@pytest.mark.parametrize("whiten", [True, False])
def test_elbo_terms_at_initialization(whiten):
    kernel = SquaredExponential()
    feature = InducingPoints(Data.Z)
    M = feature.num_inducing
    m_init = np.zeros(M)
    S_init = np.eye(M) if whiten else kernel(Data.Z, full_cov=True)
    m = VBPP(feature, kernel, Data.domain, m_init, S_init, whiten=whiten)

    Kuu = m.compute_Kuu()
    assert np.allclose(m.prior_kl(Kuu).numpy(), 0.0)
    assert np.allclose(m._elbo_integral_term(Kuu).numpy(), -m.total_area)


def test_equivalence_of_whitening():
    kernel = SquaredExponential()
    feature = InducingPoints(Data.Z)

    M = feature.num_inducing
    np.random.seed(42)
    m_init = np.random.randn(M)
    S_init = (lambda A: A @ A.T)(np.random.randn(M, M))

    Kuu = kernel(Data.Z)
    L = np.linalg.cholesky(Kuu.numpy())

    beta0 = 1.234
    m_whitened = VBPP(feature, kernel, Data.domain, m_init, S_init, whiten=True, beta0=beta0)
    m_unwhitened = VBPP(
        feature, kernel, Data.domain, L @ m_init, L @ S_init @ L.T, whiten=False, beta0=beta0
    )

    Xnew = np.linspace(-3, 13, 17)[:, None]
    f_mean_whitened, f_var_whitened = m_whitened.predict_f(Xnew)
    f_mean_unwhitened, f_var_unwhitened = m_unwhitened.predict_f(Xnew)
    np.testing.assert_allclose(f_mean_whitened, f_mean_unwhitened, rtol=1e-3)
    np.testing.assert_allclose(f_var_whitened, f_var_unwhitened, rtol=2e-3)

    np.testing.assert_allclose(
        m_whitened.elbo(Data.events), m_unwhitened.elbo(Data.events), rtol=1e-6
    )


@pytest.mark.parametrize("whiten", [True, False])
def test_lambda_predictions(whiten):
    kernel = SquaredExponential()
    feature = InducingPoints(Data.Z)

    M = feature.num_inducing
    np.random.seed(42)
    m_init = np.random.randn(M)
    S_init = (lambda A: A @ A.T)(np.random.randn(M, M))
    beta0 = 1.234

    m = VBPP(feature, kernel, Data.domain, m_init, S_init, whiten=whiten, beta0=beta0)

    mean, lower, upper = m.predict_lambda_and_percentiles(Data.Xtest)
    mean_again = m.predict_lambda(Data.Xtest)
    np.testing.assert_allclose(mean, mean_again)
    np.testing.assert_array_less(lower, mean)
    np.testing.assert_array_less(mean, upper)
