import numpy as np
import tensorflow as tf
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import SquaredExponential
import gpflow
from vbpp import VBPP

rng = np.random.RandomState(0)

def test_smoke():
    domain = np.array([[0., 10.]])
    kernel = SquaredExponential()
    events = rng.uniform(0, 10, size=20)[:, None]
    feature = InducingPoints(np.linspace(0, 10, 20)[:, None])
    M = len(feature)
    m = VBPP(feature, kernel, domain, np.zeros(M), np.eye(M))

    Kuu = m.compute_Kuu()
    m.q_sqrt.assign(np.linalg.cholesky(Kuu))
    assert np.allclose(m.prior_kl(tf.identity(Kuu)).numpy(), 0.0)

    def objective_closure():
        return - m.elbo(events)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=2))
