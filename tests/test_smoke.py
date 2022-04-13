# Copyright (C) 2022 ST John
# Copyright (C) Secondmind Ltd 2017
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

import pytest
import numpy as np
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import SquaredExponential
import gpflow
from vbpp import VBPP

rng = np.random.RandomState(0)


@pytest.mark.parametrize("whiten", [True, False])
def test_smoke(whiten):
    domain = np.array([[0.0, 10.0]])
    events = rng.uniform(0, 10, size=20)[:, None]

    kernel = SquaredExponential()
    Z = np.linspace(0, 10, 17)[:, None]
    feature = InducingPoints(Z)
    M = len(feature)
    m = VBPP(feature, kernel, domain, np.zeros(M), np.eye(M), whiten=whiten)

    def objective_closure():
        return -m.elbo(events)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=2))
