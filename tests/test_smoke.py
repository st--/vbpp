import numpy as np
import tensorflow as tf
from gpflow.features import InducingPoints
from gpflow.kernels import SquaredExponential
from gpflow.test_util import session_tf
import gpflow
from vbpp import VBPP

def test_smoke(session_tf):
    domain = np.array([[0., 10.]])
    kernel = SquaredExponential(1)
    events = np.random.uniform(0, 10, size=20)[:, None]
    feature = InducingPoints(np.linspace(0, 10, 20)[:, None])
    M = len(feature)
    m = VBPP(events, feature, kernel, domain, np.zeros(M), np.eye(M))

    Kuu = m.compute_Kuu()
    m.q_sqrt = np.linalg.cholesky(Kuu)
    assert np.allclose(session_tf.run(m.build_kl(tf.identity(Kuu))), 0.0)

    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m, maxiter=2)
