# Copyright (C) PROWLER.io 2017 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import gpflow
import numpy as np
import pytest
import tensorflow as tf
from gpflow.test_util import session_tf

from vbpp.model import integrate_log_fn_sqr
from vbpp.tf_utils import tf_len


def quadrature_log_fn_sqr(μn, σn, H=15):
    """
    Integrate log(f^2) N(f; μ, σ^2) df from -infty to infty
    """
    N = tf_len(μn)
    μn = tf.reshape(μn, (N, 1))
    σn = tf.reshape(σn, (N, 1, 1))
    return gpflow.quadrature.mvnquad(lambda f: tf.log(f ** 2), μn, σn, H, Din=1)

@pytest.mark.parametrize('H', [15, 20, 27, 30, 35])
def test_integral(session_tf, H):
    z = 10**np.linspace(1.6, 10.5, 1000)
    s2 = np.r_[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    zz, ss2 = np.meshgrid(z, s2, indexing='ij')
    mu = np.sqrt(2 * zz * ss2).flatten()
    assert np.all(mu > 0)
    sigma2 = ss2.flatten()

    gh_int = quadrature_log_fn_sqr(mu, sigma2, H)
    G_int = integrate_log_fn_sqr(mu, sigma2)
    res_gh = session_tf.run(gh_int).flatten()
    res_G = session_tf.run(G_int).flatten()
    np.testing.assert_allclose(res_gh, res_G, rtol=1e-4, atol=1e-4)
    err = np.abs(res_gh - res_G)
    assert err.max() < 2e-5
