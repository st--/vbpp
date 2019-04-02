# Copyright (C) PROWLER.io 2017
#
# Licensed under the Apache License, Version 2.0

"""
Prototype Code! This code may not be fully tested, or in other ways fit-for-purpose.
Use at your own risk!
"""

import tensorflow as tf


def tf_squeeze_1d(A):
    return tf.reshape(A, (-1,))  # TODO should check that it's got the same length as before


def tf_len(A):
    return tf.shape(A)[0]


def tf_vec_dot(v1, v2):
    """
    Calculate the dot product between v1 and v2, regardless of shapes, as long
    as there is at most one dimension with a length > 1 in each vector.
    """
    # turn into flat vectors:
    v1 = tf.squeeze(v1)
    v2 = tf.squeeze(v2)
    #XXX assert v1.ndims == 1
    #XXX assert v2.ndims == 1
    return tf.reduce_sum(tf.multiply(v1, v2))


def tf_vec_mat_vec_mul(v1, M, v2):
    """
    Calculate the bilinear form v1^T M v2, where
    v1 and v2 are vectors of length N and M is a N x N matrix.
    """
    #XXX assert tf.squeeze(v1).ndims == 1
    #XXX assert tf.squeeze(v2).ndims == 1
    v2 = tf.reshape(v2, [-1, 1])  # turn into column vector
    M_dot_v2 = tf.matmul(M, v2)
    return tf_vec_dot(v1, M_dot_v2)
