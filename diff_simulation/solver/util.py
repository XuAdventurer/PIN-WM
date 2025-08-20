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
import torch
import numpy as np


def print_header(msg):
    print('===>', msg)


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().numpy()


def bger(x, y):
    return x.unsqueeze(2).bmm(y.unsqueeze(1))


def get_sizes(G, A=None):
    if G.dim() == 2:
        nineq, nz = G.size()
        nBatch = 1
    elif G.dim() == 3:
        nBatch, nineq, nz = G.size()
    if A is not None:
        neq = A.size(1) if A.nelement() > 0 else 0
    else:
        neq = None
    # nBatch = batchedTensor.size(0) if batchedTensor is not None else None
    return nineq, nz, neq, nBatch


def bdiag(d):
    nBatch, sz = d.size()
    D = torch.zeros(nBatch, sz, sz).type_as(d)
    I = torch.eye(sz).repeat(nBatch, 1, 1).type_as(d).bool()
    D[I] = d.squeeze().view(-1)
    return D


def expandParam(X, nBatch, nDim):
    if X.ndimension() in (0, nDim) or X.nelement() == 0:
        return X, False
    elif X.ndimension() == nDim - 1:
        return X.unsqueeze(0).expand(*([nBatch] + list(X.size()))), True
    else:
        raise RuntimeError("Unexpected number of dimensions.")


def extract_nBatch(Q, p, G, h, A, b, F):
    dims = [3, 2, 3, 2, 3, 2, 3]
    params = [Q, p, G, h, A, b, F]
    for param, dim in zip(params, dims):
        if param.ndimension() == dim:
            return param.size(0)
    return 1

def extract_batch_size(Q, p, G, h, A, b):
    dims = [3, 2, 3, 2, 3, 2]
    params = [Q, p, G, h, A, b]
    for param, dim in zip(params, dims):
        if param.ndimension() == dim:
            return param.size(0)
    return 1


def efficient_btriunpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True):
    """More efficient version of torch.btriunpack.
    From https://github.com/pytorch/pytorch/issues/15182
    """
    nBatch, sz = LU_data.shape[:-1]

    if unpack_data:
        I_U = torch.ones(sz, sz, device=LU_data.device, dtype=torch.uint8).triu_().expand_as(LU_data)
        zero = torch.tensor(0.).type_as(LU_data)
        U = torch.where(I_U, LU_data, zero)
        L = torch.where(I_U, zero, LU_data)
        L.diagonal(dim1=-2, dim2=-1).fill_(1)
    else:
        L = U = None

    if unpack_pivots:
        P = torch.eye(sz, device=LU_data.device, dtype=LU_data.dtype).unsqueeze(0).repeat(nBatch, 1, 1)
        LU_pivots = LU_pivots - 1
        for i in range(nBatch):
            final_order = list(range(sz))
            for k, j in enumerate(LU_pivots[i]):
                final_order[k], final_order[j] = final_order[j], final_order[k]
            P[i] = P[i][final_order]
        P = P.transpose(-2, -1)  # This is because we permuted the rows in the previous operation
    else:
        P = None

    return P, L, U

# def orthogonal(v):


def orthogonal(v):
    """Get any orthogonal vector in 3D space"""
    min_index = torch.argmin(v.abs())
    
    base_vector = torch.zeros_like(v)
    base_vector[min_index] = 1.0
    
    return torch.cross(base_vector, v)