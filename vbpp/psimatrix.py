# Copyright (C) PROWLER.io 2017
#
# Licensed under the Apache License, Version 2.0

"""
Prototype Code! This code may not be fully tested, or in other ways fit-for-purpose.
Use at your own risk!
"""

import numpy as np
import tensorflow as tf
import gpflow

def tf_calc_Psi_matrix_SqExp(Z, variance, lengthscales, domain):
    """
    Calculates  Ψ(z,z') = ∫ K(z,x) K(x,z') dx  for the squared-exponential
    RBF kernel with `variance` (scalar) and `lengthscales` vector (length D).

    :param Z:  M x D array containing the positions of the inducing points.
    :param domain:  D x 2 array containing lower and upper bound of each dimension.

    Does not broadcast over leading dimensions.
    """
    variance = tf.cast(variance, Z.dtype)
    lengthscales = tf.cast(lengthscales, Z.dtype)

    mult = tf.cast(0.5 * np.sqrt(np.pi), Z.dtype) * lengthscales
    inv_lengthscales = 1.0 / lengthscales

    Tmin = domain[:, 0]
    Tmax = domain[:, 1]

    z1 = tf.expand_dims(Z, 1)
    z2 = tf.expand_dims(Z, 0)

    zm = (z1 + z2)/2.0

    exp_arg = tf.reduce_sum(
        - tf.square(z1 - z2) / (4.0 * tf.square(lengthscales)),
        axis=2)

    erf_val = (tf.erf((zm - Tmin) * inv_lengthscales) -
               tf.erf((zm - Tmax) * inv_lengthscales))
    product = tf.reduce_prod(mult * erf_val, axis=2)
    Ψ = tf.square(variance) * tf.exp(exp_arg + tf.log(product))
    return Ψ

def tf_calc_Psi_matrix(kernel, feature, domain):
    if (isinstance(feature, gpflow.features.InducingPoints) and
            isinstance(kernel, gpflow.kernels.SquaredExponential)):
        with gpflow.params_as_tensors_for(feature, kernel):
            return tf_calc_Psi_matrix_SqExp(feature.Z, kernel.variance, kernel.lengthscales, domain)
    else:
        raise NotImplementedError("tf_calc_Psi_matrix only implemented for SquaredExponential "
                                  "kernel with InducingPoints")
