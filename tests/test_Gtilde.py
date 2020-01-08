import numpy as np
import pytest
import tensorflow as tf

from vbpp.Gtilde import np_Gtilde_lookup, tf_Gtilde_lookup

class Data:
    z = - np.concatenate([np.random.randn(101)**2,
                          10**np.random.uniform(0, 11, 1000),
                          np.r_[0.0, 0.001, 1.0, 1.001, 1.01, 10.0, 11.0]])
    z.sort()

def test_Gtilde_errors_for_positive_values():
    with pytest.raises(ValueError):
        np_Gtilde_lookup(np.r_[0.1, -0.1, -1.2])

def test_Gtilde_at_zero():
    npG, _ = np_Gtilde_lookup(0.0)
    assert np.allclose(npG, 0.0)

def test_Gtilde_with_scalar():
    z = np.float64(- 12.3)  # give explicit type so np and tf match up
    npG, _ = np_Gtilde_lookup(z)
    tfG = tf_Gtilde_lookup(z).numpy()
    assert npG == tfG

@pytest.mark.parametrize('shape', [(-1,), (-1, 1), (-1, 2), (2, -1)])
def test_Gtilde(shape):
    z = Data.z.reshape(shape)
    npG, _ = np_Gtilde_lookup(z)
    assert npG.shape == z.shape
    tfG = tf_Gtilde_lookup(z).numpy()
    assert tfG.shape == z.shape
    np.testing.assert_equal(npG, tfG, "tensorflowed should equal numpy version")
    if shape == (-1,):
        assert list(npG) == sorted(npG), "Gtilde should be monotonous"

def test_Gtilde_gradient_matches():
    z = Data.z
    _, npgrad = np_Gtilde_lookup(z)
    assert npgrad.shape == z.shape
    z_tensor = tf.identity(z)
    with tf.GradientTape() as tape:
        tape.watch(z_tensor)
        tf_res = tf_Gtilde_lookup(z_tensor)
    tfgrad = tape.gradient(tf_res, z_tensor).numpy()
    assert tfgrad.shape == z.shape
    np.testing.assert_equal(npgrad, tfgrad, "tensorflowed should equal numpy version")
