# Copyright (C) Secondmind Ltd 2017-2020
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


import gpflow
import numpy as np
import tensorflow as tf


def _tf_calc_Phi_vector_SqExp(Z, variance, lengthscales, domain):
    Tmin = domain[:, 0]
    Tmax = domain[:, 1]

    inv_fac_lengthscales = np.sqrt(0.5) / lengthscales

    erf_val = tf.math.erf((Tmax - Z) * inv_fac_lengthscales) - tf.math.erf(
        (Tmin - Z) * inv_fac_lengthscales
    )

    mult = 0.5 * np.sqrt(2.0 * np.pi) * lengthscales
    product = tf.reduce_prod(mult * erf_val, axis=1)

    return variance * product


def tf_calc_Phi_vector(kernel, inducing_variable, domain):
    """
    This method calculates the Phi vector:
        Φ(z) = ∫dx K(x, z)
    The Phi values depend on the kernel and the type of features,
    and this function differentiates between the different configurations.
    """
    if isinstance(inducing_variable, gpflow.inducing_variables.InducingPoints) and isinstance(
        kernel, gpflow.kernels.SquaredExponential
    ):
        return _tf_calc_Phi_vector_SqExp(
            inducing_variable.Z, kernel.variance, kernel.lengthscales, domain
        )
    else:
        raise NotImplementedError(
            "tf_calc_Phi_vector only implemented for SquaredExponential kernel with InducingPoints"
        )
