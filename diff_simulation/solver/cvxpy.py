#
# Copyright 2024 Max-Planck-Gesellschaft
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ========================================================================
# This file originates from
# https://github.com/locuslab/lcp-physics/tree/a85ecfe0fdc427ee016f3d1c2ddb0d0c0f98f21b/lcp_physics, licensed under the
# Apache License, version 2.0 (see LICENSE). For compatibility with newer Pytorch version, the code was merged with code
# from https://github.com/locuslab/qpth/tree/bc121d59802d43fc6cbd685959e9156030dc1baf, licensed under the Apache License,
# Version 2.0 (see LICENSE).
# These updates were performed by Rama Krishna Kandukuri, rama.kandukuri@tuebingen.mpg.de,
# Embodied Vision Group, Max Planck Institute for Intelligent Systems.
#
import cvxpy as cp
import numpy as np


def forward_single_np(Q, p, G, h, A, b):
    nz, neq, nineq = p.shape[0], A.shape[0] if A is not None else 0, G.shape[0]

    z_ = cp.Variable(nz)

    obj = cp.Minimize(0.5 * cp.quad_form(z_, Q) + p.T * z_)
    eqCon = A * z_ == b if neq > 0 else None
    if nineq > 0:
        slacks = cp.Variable(nineq)
        ineqCon = G * z_ + slacks == h
        slacksCon = slacks >= 0
    else:
        ineqCon = slacks = slacksCon = None
    cons = [x for x in [eqCon, ineqCon, slacksCon] if x is not None]
    prob = cp.Problem(obj, cons)
    prob.solve()  # solver=cp.SCS, max_iters=5000, verbose=False)
    # prob.solve(solver=cp.SCS, max_iters=10000, verbose=True)
    assert('optimal' in prob.status)
    zhat = np.array(z_.value).ravel()
    nu = np.array(eqCon.dual_value).ravel() if eqCon is not None else None
    if ineqCon is not None:
        lam = np.array(ineqCon.dual_value).ravel()
        slacks = np.array(slacks.value).ravel()
    else:
        lam = slacks = None

    return prob.value, zhat, nu, lam, slacks