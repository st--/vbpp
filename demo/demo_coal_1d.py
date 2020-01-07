# Copyright (C) PROWLER.io 2017-2019
#
# Licensed under the Apache License, Version 2.0

# run from this file's directory

import numpy as np
import matplotlib.pyplot as plt
import gpflow

from vbpp.model import VBPP
from data import coal_dataset

def build_data(dataset):
    dim = 1
    events = dataset.get_events_distributed(rng=np.random.RandomState(42))
    events = np.array(events, float).reshape(-1, dim)
    domain = np.array(dataset.domain, float).reshape(dim, 2)
    return events, domain

def domain_grid(domain, num_points):
    return np.linspace(domain.min(axis=1), domain.max(axis=1), num_points)

def domain_area(domain):
    return np.prod(domain.max(axis=1) - domain.min(axis=1))

def build_model(events, domain, M=20):
    kernel = gpflow.kernels.SquaredExponential(input_dim=domain.shape[0])
    Z = domain_grid(domain, M)
    feature = gpflow.features.InducingPoints(Z)
    q_mu = np.zeros(M)
    q_S = np.eye(M)
    beta0 = np.sqrt(len(events) / domain_area(domain))
    model = VBPP(events, feature, kernel, domain, q_mu, q_S, beta0=beta0)
    return model

def demo():
    events, domain = build_data(coal_dataset)
    model = build_model(events, domain)
    gpflow.training.ScipyOptimizer().minimize(model)
    X = domain_grid(domain, 100)
    lambda_mean, lower, upper = model.predict_lambda_and_percentiles(X)

    plt.xlim(X.min(), X.max())
    plt.plot(X, lambda_mean)
    plt.fill_between(X.flatten(), lower.flatten(), upper.flatten(), alpha=0.3)
    plt.plot(events, np.zeros_like(events), '|')
    plt.show()

if __name__ == "__main__":
    demo()
