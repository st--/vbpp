# Copyright (C) PROWLER.io 2017-2019
#
# Licensed under the Apache License, Version 2.0

"""
Prototype Code! This code may not be fully tested, or in other ways fit-for-purpose.
Use at your own risk!
"""

import gpflow
import numpy as np
import tensorflow as tf
from functools import reduce
from gpflow import Minibatch, DataHolder, Param
from gpflow import autoflow, params_as_tensors
from gpflow import kullback_leiblers
from gpflow import transforms, settings
from gpflow.conditionals import conditional
from scipy.stats import ncx2

from .tf_utils import tf_len, tf_vec_mat_vec_mul, tf_vec_dot, tf_squeeze_1d
from .Gtilde import tf_Gtilde_lookup
from .psimatrix import tf_calc_Psi_matrix
from .phivector import tf_calc_Phi_vector


def _integrate_log_fn_sqr(mean, var):
    """
    ∫ log(f²) N(f; μ, σ²) df  from -∞ to ∞
    """
    z = - 0.5 * tf.square(mean) / var
    C = 0.57721566  # Euler-Mascheroni constant
    G = tf_Gtilde_lookup(z)
    return - G + tf.log(0.5 * var) - C


def integrate_log_fn_sqr(mean, var):
    # N = tf_len(μn)
    μn = tf_squeeze_1d(mean)
    σ2n = tf_squeeze_1d(var)
    integrated = _integrate_log_fn_sqr(mean, var)
    point_eval = tf.log(mean ** 2)  # TODO use mvnquad instead?
    # TODO explain
    return tf.where(tf.is_nan(integrated), point_eval, integrated)


class VBPP(gpflow.models.GPModel):
    """
    Implementation of the "Variational Bayes for Point Processes" model by
    Lloyd et al. (2015), with capability for multiple observations and the
    constant offset `beta0` from John and Hensman (2018).
    """

    def __init__(self, events: np.ndarray,
                 feature: gpflow.features.InducingFeature,
                 kernel: gpflow.kernels.Kernel,
                 domain: np.ndarray,
                 q_mu: np.ndarray, q_S: np.ndarray,
                 *,
                 beta0: float=0.0,
                 num_observations: int=1,
                 minibatch_size: int=None,
                 name=None):
        """
        N = total number of events observed
        D = number of dimensions
        M = size of inducing features (number of inducing points)

        :param events: observed data: the positions of observed events  (N x D)

        :param feature: inducing features (features.InducingPoints or features.FourierFeature instance)
        :param kernel: the kernel (`gpflow.kernels.Kernel` instance)
        :param q_mu: initial mean vector of the variational distribution q(u)  (length M)
        :param q_S: how to initialise the covariance matrix of the variational distribution q(u)  (M x M)

        :param domain: lower and upper bounds of (hyper-rectangular) domain (D x 2) or multi-window
        integration region in the case of the filtered derived class.

        :param beta0: initial value of prior mean of the GP; should be sufficiently large so that
        the GP does not go negative...

        :param num_observations: number of observations of sets of events under the distribution

        :param minibatch_size: size of minibatches; if None use all data
        """
        gpflow.models.Model.__init__(self, name=name)

        # observation domain  (D x 2)
        self.domain = domain
        if domain.ndim != 2 or domain.shape[1] != 2:
            raise ValueError("domain must be of shape D x 2")
        dim = domain.shape[0]

        # observed data  (N x D)
        if events.ndim != 2 or events.shape[1] != dim:
            raise ValueError("events must be of shape N x D")

        self.num_data = events.shape[0]
        if minibatch_size is None:
            self.events = DataHolder(events)
            self.minibatch_scale = 1.0
        else:
            self.events = Minibatch(events, batch_size=minibatch_size, seed=0)
            self.minibatch_scale = self.num_data / minibatch_size

        self.num_observations = num_observations

        if not (isinstance(kernel, gpflow.kernels.SquaredExponential)
                and isinstance(feature, gpflow.features.InducingPoints)):
            raise NotImplementedError("This VBPP implementation can only handle real-space "
                                      "inducing points together with the SquaredExponential "
                                      "kernel.")
        self.kernel = kernel
        self.feature = feature

        self.beta0 = Param(beta0, transform=transforms.positive)  # constant mean offset

        # variational approximate Gaussian posterior q(u) = N(u; m, S)
        self.q_mu = Param(q_mu)  # mean vector  (length M)

        # covariance:
        L = np.linalg.cholesky(q_S)  # S = L L^T, with L lower-triangular  (M x M)
        self.q_sqrt = Param(L, transform=transforms.LowerTriangular(len(feature), num_matrices=1,
                                                                    squeeze=True))

        self.num_latent = 1
        self.psi_jitter = 0.0

    @params_as_tensors
    def build_Psi_matrix(self):
        Ψ = tf_calc_Psi_matrix(self.kernel, self.feature, self.domain)
        psi_jitter_matrix = self.psi_jitter * tf.eye(len(self.feature), dtype=settings.float_type)
        return Ψ + psi_jitter_matrix 

    @property
    def total_area(self):
        return np.prod(self.domain[:, 1] - self.domain[:, 0])

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, *, Kuu=None):
        """
        VBPP-specific conditional on the approximate posterior q(u), including a
        constant mean function.
        """
        mean, var = conditional(Xnew, self.feature, self.kernel, self.q_mu[:, None],
                                full_cov=full_cov, q_sqrt=self.q_sqrt[None, :, :])
        # TODO make conditional() use Kuu if available

        return mean + self.beta0, var

    @params_as_tensors
    def build_elbo_data_term(self, Kuu=None):
        mean, var = self._build_predict(self.events, full_cov=False, Kuu=Kuu)
        expect_log_fn_sqr = integrate_log_fn_sqr(mean, var)
        return self.minibatch_scale * tf.reduce_sum(expect_log_fn_sqr)

    @params_as_tensors
    def build_var_fx_kxx_term(self):
        if isinstance(self.kernel, gpflow.kernels.Product):
            γ = reduce(lambda a, b: a * b, [k.variance for k in self.kernel.kernels])
        elif isinstance(self.kernel, gpflow.kernels.Sum):
            γ = reduce(lambda a, b: a + b, [k.variance for k in self.kernel.kernels])
        else:
            γ = self.kernel.variance
        kxx_term = γ * self.total_area
        return kxx_term

    @params_as_tensors
    def build_elbo_integral_term(self, Kuu):
        """
        Kuu : dense matrix
        """
        # q(f) = GP(f; μ, Σ)
        Ψ = self.build_Psi_matrix()

        # int_expect_fx_sqr = m^T Kzz⁻¹ Ψ Kzz⁻¹ m
        # = (Kzz⁻¹ m)^T Ψ (Kzz⁻¹ m)

        # Kzz = R R^T
        R = tf.cholesky(Kuu)

        # Kzz⁻¹ m = R^-T R⁻¹ m
        # Rinv_m = R⁻¹ m
        Rinv_m = tf.matrix_triangular_solve(R, self.q_mu[:, None], lower=True)

        # R⁻¹ Ψ R^-T
        # = (R⁻¹ Ψ) R^-T
        Rinv_Ψ = tf.matrix_triangular_solve(R, Ψ, lower=True)
        # = (Rinv_Ψ) R^-T = (R⁻¹ Rinv_Ψ^T)^T
        Rinv_Ψ_RinvT = tf.matrix_triangular_solve(R, tf.transpose(Rinv_Ψ), lower=True)

        int_mean_f_sqr = tf_vec_mat_vec_mul(Rinv_m, Rinv_Ψ_RinvT, Rinv_m)

        Rinv_L = tf.matrix_triangular_solve(R, self.q_sqrt, lower=True)
        Rinv_L_LT_RinvT = tf.matmul(Rinv_L, Rinv_L, transpose_b=True)

        # int_var_fx = γ |T| + trace_terms
        # trace_terms = - Tr(Kzz⁻¹ Ψ) + Tr(Kzz⁻¹ S Kzz⁻¹ Ψ)
        trace_terms = tf.reduce_sum((Rinv_L_LT_RinvT - tf.eye(len(self.feature), dtype=settings.float_type)) *
                                    Rinv_Ψ_RinvT)

        kxx_term = self.build_var_fx_kxx_term()
        int_var_f = kxx_term + trace_terms

        f_term = int_mean_f_sqr + int_var_f

        # λ = E_f{(f + β₀)**2}
        #   = (E_f)^2 + var_f + 2 f β₀ + β₀^2
        #   = f_term + int_cross_terms + betas_term
        Kuu_inv_m = tf.matrix_triangular_solve(tf.transpose(R), Rinv_m, lower=False)

        Phi = tf_calc_Phi_vector(self.kernel, self.feature, self.domain)
        int_cross_term = 2 * self.beta0 * tf_vec_dot(Phi, Kuu_inv_m)

        beta_term = tf.square(self.beta0) * self.total_area

        int_lambda = f_term + int_cross_term + beta_term

        return - int_lambda

    @params_as_tensors
    def build_kl(self, Kuu):
        """
        KL divergence between p(u) = N(0, Kuu) and q(u) = N(μ, S)
        """
        return kullback_leiblers.gauss_kl(self.q_mu[:, None], self.q_sqrt[None, :, :], Kuu)

    @params_as_tensors
    def _build_likelihood(self):
        """
        Evidence Lower Bound (ELBo) for the log likelihood.
        """
        K = gpflow.features.Kuu(self.feature, self.kernel)

        integral_term = self.build_elbo_integral_term(K)
        data_term = self.build_elbo_data_term(K)
        kl_div = self.build_kl(K)

        elbo = self.num_observations * integral_term + data_term - kl_div
        return tf.squeeze(elbo)  # XXX this should be fixed higher up

    @autoflow()
    def compute_Kuu(self):
        return gpflow.features.Kuu(self.feature, self.kernel)

    def build_lambda(self, Xnew):
        """
        Expectation value of the rate function of the Poisson process.

        :param xx: points at which to calculate
        """
        mean_f, var_f = self._build_predict(Xnew)
        λ = tf.square(mean_f) + var_f
        return λ

    @autoflow((settings.float_type, [None, None]))
    def predict_lambda(self, Xnew):
        return self.build_lambda(Xnew)

    def compute_lambda_and_percentiles(self, Xnew, lower=5, upper=95):
        """
        Computes mean value of intensity and lower and upper percentiles.
        `lower` and `upper` must be between 0 and 100.
        """
        # f ~ Normal(mean_f, var_f)
        mean_f, var_f = self.predict_f(Xnew)
        # λ = E[f²] = E[f]² + Var[f]
        lambda_mean = mean_f ** 2 + var_f
        # g = f/√var_f ~ Normal(mean_f/√var_f, 1)
        # g² = f²/var_f ~ χ²(k=1, λ=mean_f²/var_f) non-central chi-squared
        m2ov = mean_f ** 2 / var_f
        if m2ov > 10e3:
            raise ValueError("scipy.stats.ncx2.ppf() flatlines for nc > 10e3")
        f2ov_lower = ncx2_ppf(lower/100, df=1, nc=m2ov)
        f2ov_upper = ncx2_ppf(upper/100, df=1, nc=m2ov)
        # f² = g² * var_f
        lambda_lower = f2ov_lower * var_f
        lambda_upper = f2ov_upper * var_f
        return lambda_mean, lambda_lower, lambda_upper
