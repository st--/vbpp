import gpflow
import numpy as np
import tensorflow as tf

def _tf_calc_Phi_vector_SqExp(Z, variance, lengthscales, domain):
    Tmin = domain[:, 0]
    Tmax = domain[:, 1]

    inv_fac_lengthscales = np.sqrt(0.5) / lengthscales

    erf_val = (tf.erf((Tmax - Z) * inv_fac_lengthscales) -
               tf.erf((Tmin - Z) * inv_fac_lengthscales))

    mult = 0.5 * np.sqrt(2. * np.pi) * lengthscales
    product = tf.reduce_prod(mult * erf_val, axis=1)

    return variance * product

def tf_calc_Phi_vector(kernel, feature, domain):
    """
    This method calculates the Phi vector:
        Φ(z) = ∫dx K(x, z)
    The Phi values depend on the kernel and the type of features,
    and this function differentiates between the different configurations.
    """
    if (isinstance(feature, gpflow.features.InducingPoints) and
            isinstance(kernel, gpflow.kernels.SquaredExponential)):
        with gpflow.params_as_tensors_for(feature, kernel):
            return _tf_calc_Phi_vector_SqExp(feature.Z, kernel.variance, kernel.lengthscales, domain)
    else:
        raise NotImplementedError("tf_calc_Phi_vector only implemented for SquaredExponential "
                                  "kernel with InducingPoints")
