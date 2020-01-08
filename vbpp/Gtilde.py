# Copyright (C) PROWLER.io 2017-2019
#
# Licensed under the Apache License, Version 2.0

import logging
import numpy as np
import tensorflow as tf

from .Gtilde_data import __G_lookup_table

LOG = logging.getLogger(__name__)

# lookup table code adapted from Gtilde_lookup.m provided by Chris Lloyd


def np_Gtilde_lookup(z: np.ndarray):
    """
    Calculates G̃(z)

    :param z: the points at which to evaluate G̃
    :returns: values and derivatives of G̃(z)

    See "Variational Inference for Gaussian Process Modulated Poisson Processes"
    by Lloyd et al. (2015), Section 4.3
    """
    # adapted from matlab code by Chris Lloyd

    z = np.array(z)  # copy and do not overwrite input array

    is_scalar = z.ndim == 0
    z = np.atleast_1d(z)  # otherwise array indexing does not work

    # Transform z -> -z, bit easier to think about!
    z = - z

    if np.any(z >= 10**(len(__G_lookup_table)-1)):
        n_overflow = (z >= 10**(len(__G_lookup_table)-1)).sum()
        n_lesszero = (z < 0).sum()
        LOG.warning('Gtilde: z out of range: %d greater-than-zeros, '
                    '%d out-of-ranges (of %d)', n_lesszero, n_overflow, len(z))

    if np.any(z < 0):  # remember, original zs were assumed to be negative
        raise ValueError("Gtilde: invalid z: we require z <= 0")

    REAL_MIN = np.finfo(z.dtype).tiny  # smallest positive number
    z[z == 0] = REAL_MIN

    Gs = np.zeros_like(z)
    dGs = np.zeros_like(z)

    out_of_range = (z >= 10**(len(__G_lookup_table)-1))
    Gs[out_of_range] = dGs[out_of_range] = np.nan

    lower = 0
    upper = 1
    binWidth = 0.001
    # For each region
    for Gi in __G_lookup_table:
        # Work out which z lie in this region
        zR = (lower <= z) & (z < upper)

        # Work out which are the upper (zj) and lower (zi) intervals for each z
        # and the fraction across the bin the point is (zr)
        zi = np.floor(    z[zR] / binWidth).astype(int)  # lower
        zj = np.ceil(     z[zR] / binWidth).astype(int)  # upper
        zr = np.remainder(z[zR] / binWidth, 1)           # remainder
        # If the remainder is zero, increment the second point
        zj[zr == 0.0] += 1
        # Compute the gradient for each point
        dGs[zR] = Gi[zj] - Gi[zi]
        # Interpolate using the gradient to find the function value
        Gs[zR]  = Gi[zi] + dGs[zR] * zr
        # Correct dG for bin width
        dGs[zR] = dGs[zR] / binWidth

        # Adjust binWidth, upper and lower boundaries for next region
        binWidth = binWidth * 10
        lower = upper
        upper = upper * 10

    # Correct dG for z -> -z transformation
    dGs = - dGs

    if is_scalar:  # undo atleast_1d
        Gs = Gs[0]
        dGs = dGs[0]
    return Gs, dGs

def _tf_Gtilde_lookup(z, name=None):
    """
    This function returns the value of the lookup table, as well as the
    derivative of the lookup table (which we get for free). Reverse model
    differentiation w.r.t the derivative will fail: only the first argument is
    meant to be used. Returning both allows us to save some computation.
    """
    with tf.name_scope(name=name) as scope:
        return tf.py_function(np_Gtilde_lookup, [z], [z.dtype, z.dtype], name=scope)

@tf.custom_gradient
def tf_Gtilde_lookup(z, name=None):
    """
    Tensorflowed version of np_Gtilde_lookup with gradient
    """
    if name is None:
        name = "tf_Gtilde_lookup"

    Gs, dGs = _tf_Gtilde_lookup(z, name=name)

    def grad(grad_G):
        return dGs * grad_G

    return Gs, grad
