import torch
import numpy as np
import math

def quaternion_conjugate(q):
    w, x, y, z = q
    return torch.tensor(([w,-x,-y,-z]),device = q.device)

def rotate_point_by_quaternion(q, p):

    # import pdb;pdb.set_trace() 
    p_q = torch.cat((torch.zeros_like((p[..., :1]),device = p.device), p), dim=-1)

    q_conjugate = torch.cat((q[..., 0:1], -q[..., 1:4]), dim=-1)

    temp = quaternion_multiply(q, p_q)  # q * p_q
    rotated_p_q = quaternion_multiply(temp, q_conjugate)  # (q * p_q) * q^-1

    rotated_p = rotated_p_q[..., 1:]
    return rotated_p

def quaternion_multiply(q1, q2):

    w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    q = torch.cat((w, x, y, z), dim=-1)
    return q

def normalize(quaternion):
    r"""Normalizes a quaternion to unit norm.

    Args:
        quaternion (torch.Tensor): Quaternion to normalize (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).

    Returns:
        (torch.Tensor): Normalized quaternion (shape: :math:`(4)`).
    """
    norm = quaternion.norm(p=2, dim=0) + 1e-5
    return quaternion / norm

def rotmat_to_quaternion(R):

    assert R.shape[-2:] == (3, 3)
    
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    
    q = torch.zeros(R.shape[:-2] + (4,), device=R.device, dtype=R.dtype)
    
    trace_positive = trace > 0
    if trace_positive.any():
        s = torch.sqrt(trace[trace_positive] + 1.0) * 2  # s = 4 * w
        q[trace_positive, 0] = 0.25 * s
        q[trace_positive, 1] = (R[trace_positive, 2, 1] - R[trace_positive, 1, 2]) / s
        q[trace_positive, 2] = (R[trace_positive, 0, 2] - R[trace_positive, 2, 0]) / s
        q[trace_positive, 3] = (R[trace_positive, 1, 0] - R[trace_positive, 0, 1]) / s
    
    cond1 = (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2]) & (~trace_positive)
    if cond1.any():
        s = torch.sqrt(1.0 + R[cond1, 0, 0] - R[cond1, 1, 1] - R[cond1, 2, 2]) * 2  # s = 4 * x
        q[cond1, 0] = (R[cond1, 2, 1] - R[cond1, 1, 2]) / s
        q[cond1, 1] = 0.25 * s
        q[cond1, 2] = (R[cond1, 0, 1] + R[cond1, 1, 0]) / s
        q[cond1, 3] = (R[cond1, 0, 2] + R[cond1, 2, 0]) / s
    
    cond2 = (R[..., 1, 1] > R[..., 2, 2]) & (~cond1) & (~trace_positive)
    if cond2.any():
        s = torch.sqrt(1.0 + R[cond2, 1, 1] - R[cond2, 0, 0] - R[cond2, 2, 2]) * 2  # s = 4 * y
        q[cond2, 0] = (R[cond2, 0, 2] - R[cond2, 2, 0]) / s
        q[cond2, 1] = (R[cond2, 0, 1] + R[cond2, 1, 0]) / s
        q[cond2, 2] = 0.25 * s
        q[cond2, 3] = (R[cond2, 1, 2] + R[cond2, 2, 1]) / s
    
    cond3 = ~cond1 & ~cond2 & ~trace_positive
    if cond3.any():
        s = torch.sqrt(1.0 + R[cond3, 2, 2] - R[cond3, 0, 0] - R[cond3, 1, 1]) * 2  # s = 4 * z
        q[cond3, 0] = (R[cond3, 1, 0] - R[cond3, 0, 1]) / s
        q[cond3, 1] = (R[cond3, 0, 2] + R[cond3, 2, 0]) / s
        q[cond3, 2] = (R[cond3, 1, 2] + R[cond3, 2, 1]) / s
        q[cond3, 3] = 0.25 * s

    return q    

def quaternion_to_rotmat(quaternion):
    r"""Converts a quaternion to a :math:`3 \times 3` rotation matrix.

    Args:
        quaternion (torch.Tensor): Quaternion to convert (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).

    Returns:
        (torch.Tensor): rotation matrix (shape: :math:`(3, 3)`).
    """
    r = quaternion[0]
    i = quaternion[1]
    j = quaternion[2]
    k = quaternion[3]
    rotmat = torch.zeros(3, 3, dtype=quaternion.dtype, device=quaternion.device)
    twoisq = 2 * i * i
    twojsq = 2 * j * j
    twoksq = 2 * k * k
    twoij = 2 * i * j
    twoik = 2 * i * k
    twojk = 2 * j * k
    twori = 2 * r * i
    tworj = 2 * r * j
    twork = 2 * r * k
    rotmat[0, 0] = 1 - twojsq - twoksq
    rotmat[0, 1] = twoij - twork
    rotmat[0, 2] = twoik + tworj
    rotmat[1, 0] = twoij + twork
    rotmat[1, 1] = 1 - twoisq - twoksq
    rotmat[1, 2] = twojk - twori
    rotmat[2, 0] = twoik - tworj
    rotmat[2, 1] = twojk + twori
    rotmat[2, 2] = 1 - twoisq - twojsq
    return rotmat

def xyzw_quaternion_to_rotmat(quaternion):
    r"""Converts a quaternion to a :math:`3 \times 3` rotation matrix.

    Args:
        quaternion (torch.Tensor): Quaternion to convert (shape: :math:`(4)`)
            (Assumes (x, y, z, w) convention, with :math:`r` being the scalar).

    Returns:
        (torch.Tensor): rotation matrix (shape: :math:`(3, 3)`).
    """
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]
    
    rotmat = np.array([
        [1 - 2 * (y ** 2) - 2 * (z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x ** 2) - 2 * (z ** 2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2) - 2 * (y ** 2)]
    ],dtype=np.float32)
    
    return rotmat

def multiply(q1, q2):
    r"""Multiply two quaternions `q1`, `q2`.

    Args:
        q1 (torch.Tensor): First quaternion (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).
        q2 (torch.Tensor): Second quaternion (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).

    Returns:
        (torch.Tensor): Quaternion product (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).
    """
    r1 = q1[0]
    v1 = q1[1:]
    r2 = q2[0]
    v2 = q2[1:]
    return torch.cat(
        (
            r1 * r2 - torch.matmul(v1.view(1, 3), v2.view(3, 1)).view(-1),
            r1 * v2 + r2 * v1 + torch.cross(v1, v2),
        ),
        dim=0,
    )

def wxyz2xyzw(q):
    return torch.tensor(([q[1],q[2],q[3],q[0]]),device = q.device)

def xyzw2wxyz(q):
    return torch.tensor(([q[3],q[0],q[1],q[2]]),device = q.device)

def wxyz2xyzw_np(q):
    return np.array(([q[1],q[2],q[3],q[0]]),dtype=np.float32)

def xyzw2wxyz_np(q):
    return np.array(([q[3],q[0],q[1],q[2]]),dtype=np.float32)

def create_q_x(theta):
    theta_rad = math.radians(theta)
    half_theta = theta_rad / 2
    cos_half_theta = np.cos(half_theta)
    sin_half_theta = np.sin(half_theta)
    q_x = torch.tensor([cos_half_theta, sin_half_theta, 0, 0],dtype=torch.float32)
    return q_x

def create_q_z(theta):
    theta_rad = math.radians(theta)
    half_theta = theta_rad / 2
    cos_half_theta = np.cos(half_theta)
    sin_half_theta = np.sin(half_theta)
    q_z = torch.tensor([cos_half_theta, 0, 0, sin_half_theta],dtype=torch.float32)
    return q_z

def create_q_y(theta):
    theta_rad = math.radians(theta)
    half_theta = theta_rad / 2
    cos_half_theta = np.cos(half_theta)
    sin_half_theta = np.sin(half_theta)
    q_y = torch.tensor([cos_half_theta, 0, sin_half_theta, 0],dtype=torch.float32)
    return q_y

def quaternion_standardize(q):
    return torch.where(q[-1] < 0, -q, q)

def quaternion_standardize_np(q):
    return np.where(q[-1] < 0, -q, q)


def quaternion_multiply_np(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x3 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z3 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w3 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.array([x3, y3, z3, w3])