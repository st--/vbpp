import numpy as np
import pytest
import gpflow
from vbpp.psimatrix import tf_calc_Psi_matrix
from .psimatrix_np import calc_Ψ_matrix, calc_Ψ_matrix_2

class Data:
    M = 15
    D = 3
    Z = np.random.randn(M, D)
    variance = 0.8
    lengthscale = np.array([0.8] * D)
    domain = np.array([[0, 1]] * D)

@pytest.fixture
def Psi():
    inducing_var = gpflow.inducing_variables.InducingPoints(Data.Z)
    kernel = gpflow.kernels.SquaredExponential(variance=Data.variance, lengthscale=Data.lengthscale)
    Psi_tensor = tf_calc_Psi_matrix(kernel, inducing_var, Data.domain)
    return Psi_tensor.numpy()

def test_Psi_matrix_shape(Psi):
    assert Psi.shape == (Data.M, Data.M)

def test_tf_calc_Psi_matrix_is_symmetric(Psi):
    assert np.all(Psi == Psi.T)

def test_tf_calc_Psi_matrix_matches_np(Psi):
    # tensorflowed equals numpy version:
    # tf_calc_Psi_matrix == calc_Ψ_matrix == calc_Ψ_matrix_2
    np_Psi = calc_Ψ_matrix(Data.Z, Data.variance, Data.lengthscale, Data.domain)
    np_Psi2 = calc_Ψ_matrix_2(Data.Z, Data.variance, Data.lengthscale, Data.domain)
    np.testing.assert_allclose(Psi, np_Psi)
    np.testing.assert_allclose(Psi, np_Psi2)
