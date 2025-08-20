import trimesh
import torch
import numpy as np
# import ode

def cal_MassProperties(mesh:trimesh.base.Trimesh,device):
    # Implemented from:
    # http://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf
    # by trimesh
    triangles = torch.tensor(mesh.triangles,dtype=torch.float32,device=device)
    crosses = torch.tensor(mesh.triangles_cross,dtype=torch.float32,device=device)
    # triangles = np.asanyarray(self.mesh.triangles, dtype=np.float32)
    # crosses = self.mesh.triangles_cross

        # these are the subexpressions of the integral
    # this is equvilant but 7x faster than triangles.sum(axis=1)
    f1 = triangles[:, 0, :] + triangles[:, 1, :] + triangles[:, 2, :]
    # for the the first vertex of every triangle:
    # triangles[:,0,:] will give rows like [[x0, y0, z0], ...]

    # for the x coordinates of every triangle
    # triangles[:,:,0] will give rows like [[x0, x1, x2], ...]
    f2 = (
        triangles[:, 0, :] ** 2
        + triangles[:, 1, :] ** 2
        + triangles[:, 0, :] * triangles[:, 1, :]
        + triangles[:, 2, :] * f1
    )
    f3 = (
        (triangles[:, 0, :] ** 3)
        + (triangles[:, 0, :] ** 2) * (triangles[:, 1, :])
        + (triangles[:, 0, :]) * (triangles[:, 1, :] ** 2)
        + (triangles[:, 1, :] ** 3)
        + (triangles[:, 2, :] * f2)
    )
    g0 = f2 + (triangles[:, 0, :] + f1) * triangles[:, 0, :]
    g1 = f2 + (triangles[:, 1, :] + f1) * triangles[:, 1, :]
    g2 = f2 + (triangles[:, 2, :] + f1) * triangles[:, 2, :]
    integral = torch.zeros((10, len(f1)),dtype=torch.float32,device=device)
    # integral = np.zeros((10, len(f1)))
    integral[0] = crosses[:, 0] * f1[:, 0]
    integral[1:4] = (crosses * f2).T
    integral[4:7] = (crosses * f3).T
    for i in range(3):
        triangle_i = torch.fmod(torch.tensor(i + 1, device=device), 3)
        integral[i + 7] = crosses[:, i] * (
            (triangles[:, 0, triangle_i] * g0[:, i])
            + (triangles[:, 1, triangle_i] * g1[:, i])
            + (triangles[:, 2, triangle_i] * g2[:, i])
        )

    coefficients = 1.0 / torch.tensor(
        [6, 24, 24, 24, 60, 60, 60, 120, 120, 120], dtype=torch.float32,device=device
    )
    integrated = integral.sum(axis=1) * coefficients

    volume = integrated[0]
    # density = mass / volume

    center_mass = integrated[1:4] / volume

    inertia_body_unit = torch.zeros((3, 3),dtype=torch.float32,device=device)
    inertia_body_unit[0, 0] = (
        integrated[5] + integrated[6] - (volume * (center_mass[[1, 2]] ** 2).sum())
    )
    inertia_body_unit[1, 1] = (
        integrated[4] + integrated[6] - (volume * (center_mass[[0, 2]] ** 2).sum())
    )
    inertia_body_unit[2, 2] = (
        integrated[4] + integrated[5] - (volume * (center_mass[[0, 1]] ** 2).sum())
    )
    inertia_body_unit[0, 1] = -(integrated[7] - (volume * torch.prod(center_mass[[0, 1]])))
    inertia_body_unit[1, 2] = -(integrated[8] - (volume * torch.prod(center_mass[[1, 2]])))
    inertia_body_unit[0, 2] = -(integrated[9] - (volume * torch.prod(center_mass[[0, 2]])))
    inertia_body_unit[2, 0] = inertia_body_unit[0, 2]
    inertia_body_unit[2, 1] = inertia_body_unit[1, 2]
    inertia_body_unit[1, 0] = inertia_body_unit[0, 1]
    # inertia_body = inertia_body * density
    
    return volume,center_mass,inertia_body_unit

def cal_MassProperties_np(mesh:trimesh.base.Trimesh):
    # Implemented from:
    # http://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf
    # by trimesh
    triangles = np.asanyarray(mesh.triangles, dtype=np.float32)
    crosses = np.asanyarray(mesh.triangles_cross,dtype=np.float32)
    # triangles = np.asanyarray(self.mesh.triangles, dtype=np.float32)
    # crosses = self.mesh.triangles_cross

        # these are the subexpressions of the integral
    # this is equvilant but 7x faster than triangles.sum(axis=1)
    f1 = triangles[:, 0, :] + triangles[:, 1, :] + triangles[:, 2, :]
    # for the the first vertex of every triangle:
    # triangles[:,0,:] will give rows like [[x0, y0, z0], ...]

    # for the x coordinates of every triangle
    # triangles[:,:,0] will give rows like [[x0, x1, x2], ...]
    f2 = (
        triangles[:, 0, :] ** 2
        + triangles[:, 1, :] ** 2
        + triangles[:, 0, :] * triangles[:, 1, :]
        + triangles[:, 2, :] * f1
    )
    f3 = (
        (triangles[:, 0, :] ** 3)
        + (triangles[:, 0, :] ** 2) * (triangles[:, 1, :])
        + (triangles[:, 0, :]) * (triangles[:, 1, :] ** 2)
        + (triangles[:, 1, :] ** 3)
        + (triangles[:, 2, :] * f2)
    )
    g0 = f2 + (triangles[:, 0, :] + f1) * triangles[:, 0, :]
    g1 = f2 + (triangles[:, 1, :] + f1) * triangles[:, 1, :]
    g2 = f2 + (triangles[:, 2, :] + f1) * triangles[:, 2, :]
    integral = np.zeros((10, len(f1)), dtype=np.float32)
    # integral = np.zeros((10, len(f1)))
    integral[0] = crosses[:, 0] * f1[:, 0]
    integral[1:4] = (crosses * f2).T
    integral[4:7] = (crosses * f3).T
    for i in range(3):
        triangle_i = np.mod(i + 1, 3)
        integral[i + 7] = crosses[:, i] * (
            (triangles[:, 0, triangle_i] * g0[:, i])
            + (triangles[:, 1, triangle_i] * g1[:, i])
            + (triangles[:, 2, triangle_i] * g2[:, i])
        )

    coefficients = 1.0 / np.array(
        [6, 24, 24, 24, 60, 60, 60, 120, 120, 120], dtype=np.float32
    )
    integrated = integral.sum(axis=1) * coefficients

    volume = integrated[0]
    # density = mass / volume

    center_mass = integrated[1:4] / volume

    inertia_body_unit = np.zeros((3, 3), dtype=np.float32)
    inertia_body_unit[0, 0] = (
        integrated[5] + integrated[6] - (volume * (center_mass[[1, 2]] ** 2).sum())
    )
    inertia_body_unit[1, 1] = (
        integrated[4] + integrated[6] - (volume * (center_mass[[0, 2]] ** 2).sum())
    )
    inertia_body_unit[2, 2] = (
        integrated[4] + integrated[5] - (volume * (center_mass[[0, 1]] ** 2).sum())
    )
    inertia_body_unit[0, 1] = -(integrated[7] - (volume * np.prod(center_mass[[0, 1]])))
    inertia_body_unit[1, 2] = -(integrated[8] - (volume * np.prod(center_mass[[1, 2]])))
    inertia_body_unit[0, 2] = -(integrated[9] - (volume * np.prod(center_mass[[0, 2]])))
    inertia_body_unit[2, 0] = inertia_body_unit[0, 2]
    inertia_body_unit[2, 1] = inertia_body_unit[1, 2]
    inertia_body_unit[1, 0] = inertia_body_unit[0, 1]
    # inertia_body = inertia_body * density
    
    return volume,center_mass,inertia_body_unit

def cal_transform_matrix(scale_factors,rotation_quaternion,translation_vectors):
    # scale
    scaling_matrix =  np.eye(4, dtype=np.float32)
    scaling_matrix[0, 0] = scale_factors[0]
    scaling_matrix[1, 1] = scale_factors[1]
    scaling_matrix[2, 2] = scale_factors[2]        
    # rotation
    from utils.quaternion_utils import xyzw_quaternion_to_rotmat
    rotation_matrix = xyzw_quaternion_to_rotmat(rotation_quaternion)
    rotation_matrix = np.concatenate((rotation_matrix, [[0,0,0]]), axis=0)
    rotation_matrix = np.concatenate((rotation_matrix, [[0],[0],[0],[1]]), axis=1)
    # translation
    translation_matrix = np.eye(4,dtype=np.float32)
    translation_matrix[:3, 3] = translation_vectors

    transform_matrix = np.dot(translation_matrix, np.dot(rotation_matrix, scaling_matrix))
    return transform_matrix

def get_mesh_world(mesh,position,rotation):
    import copy
    mesh_copy = copy.deepcopy(mesh)
    transform_matrix = cal_transform_matrix([1,1,1],
                                            rotation,
                                            position)
    mesh_copy.apply_transform(transform_matrix)
    return mesh_copy


def create_sample_points(sample_num,mesh_local):
    sample_mesh = (mesh_local.slice_plane(plane_normal=[0,0,1]
                                ,plane_origin=[0,0,-0.0001])).slice_plane(plane_normal=[0,0,-1],
                                                                        plane_origin=[0,0,0.0001])
    points_position,faces_index = trimesh.sample.sample_surface(sample_mesh,count = sample_num,face_weight = None, seed=0)
    points_normal = sample_mesh.face_normals[faces_index]
    sample_points = np.concatenate((points_position, np.ones((sample_num,1),dtype=np.float32),
                                                    points_normal), axis=1)  
    return sample_points

def get_contact_info(action,sample_points,obj_position,obj_rotation):
    contact_point = sample_points[action]
    transform_matrix = cal_transform_matrix([1,1,1],
                                            obj_rotation,
                                            obj_position)
    contact_point_position = np.dot(transform_matrix,contact_point[:4])[:3]
    contact_point_normal = np.dot(transform_matrix[:3,:3],contact_point[4:])
    # self.vis_contact_info(transform_matrix,contact_point_position)
    return contact_point_position,contact_point_normal



def comp_projection_integrals(verts, faces, A, B):
    a0 = verts[faces][torch.arange(faces.shape[0]), :, A]
    b0 = verts[faces][torch.arange(faces.shape[0]), :, B]
    a1 = verts[faces[:, [1, 2, 0]]][torch.arange(faces.shape[0]), :, A]
    b1 = verts[faces[:, [1, 2, 0]]][torch.arange(faces.shape[0]), :, B]
    da = a1 - a0
    db = b1 - b0
    a0_2 = a0 * a0
    a0_3 = a0_2 * a0
    a0_4 = a0_3 * a0
    b0_2 = b0 * b0
    b0_3 = b0_2 * b0
    b0_4 = b0_3 * b0
    a1_2 = a1 * a1
    a1_3 = a1_2 * a1
    b1_2 = b1 * b1
    b1_3 = b1_2 * b1

    C1 = a1 + a0
    Ca = a1 * C1 + a0_2
    Caa = a1 * Ca + a0_3
    Caaa = a1 * Caa + a0_4
    Cb = b1 * (b1 + b0) + b0_2
    Cbb = b1 * Cb + b0_3
    Cbbb = b1 * Cbb + b0_4
    Cab = 3 * a1_2 + 2 * a1 * a0 + a0_2
    Kab = a1_2 + 2 * a1 * a0 + 3 * a0_2
    Caab = a0 * Cab + 4 * a1_3
    Kaab = a1 * Kab + 4 * a0_3
    Cabb = 4 * b1_3 + 3 * b1_2 * b0 + 2 * b1 * b0_2 + b0_3
    Kabb = b1_3 + 2 * b1_2 * b0 + 3 * b1 * b0_2 + 4 * b0_3

    P1 = (db * C1).sum(dim=1) / 2.0
    Pa = (db * Ca).sum(dim=1) / 6.0
    Paa = (db * Caa).sum(dim=1) / 12.0
    Paaa = (db * Caaa).sum(dim=1) / 20.0
    Pb = (da * Cb).sum(dim=1) / -6.0
    Pbb = (da * Cbb).sum(dim=1) / -12.0
    Pbbb = (da * Cbbb).sum(dim=1) / -20.0
    Pab = (db * (b1 * Cab + b0 * Kab)).sum(dim=1) / 24.0
    Paab = (db * (b1 * Caab + b0 * Kaab)).sum(dim=1) / 60.0
    Pabb = (da * (a1 * Cabb + a0 * Kabb)).sum(dim=1) / -60.0

    return P1, Pa, Paa, Paaa, Pb, Pbb, Pbbb, Pab, Paab, Pabb


def comp_face_integrals(verts, faces, normals, w, A, B, C):
    P1, Pa, Paa, Paaa, Pb, Pbb, Pbbb, Pab, Paab, Pabb = comp_projection_integrals(verts, faces, A, B)

    k1 = 1 / normals[torch.arange(normals.shape[0]), C]
    k2 = k1 * k1
    k3 = k2 * k1
    k4 = k3 * k1

    nA = normals[torch.arange(normals.shape[0]), A]
    nB = normals[torch.arange(normals.shape[0]), B]

    Fa = k1 * Pa
    Fb = k1 * Pb
    Fc = -k2 * (nA * Pa + nB * Pb + w * P1)

    Faa = k1 * Paa
    Fbb = k1 * Pbb
    Fcc = k3 * (nA * nA * Paa + 2 * nA * nB * Pab + nB * nB * Pbb
                + w * (2 * (nA * Pa + nB * Pb) + w * P1))

    Faaa = k1 * Paaa
    Fbbb = k1 * Pbbb
    Fccc = -k4 * (nA ** 3 * Paaa + 3 * nA * nA * nB * Paab
                  + 3 * nA * nB * nB * Pabb + nB * nB * nB * Pbbb
                  + 3 * w * (nA * nA * Paa + 2 * nA * nB * Pab + nB * nB * Pbb)
                  + w * w * (3 * (nA * Pa + nB * Pb) + w * P1))

    Faab = k1 * Paab
    Fbbc = -k2 * (nA * Pabb + nB * Pbbb + w * Pbb)
    Fcca = k3 * (nA * nA * Paaa + 2 * nA * nB * Paab + nB * nB * Pabb
                 + w * (2 * (nA * Paa + nB * Pab) + w * Pa))

    return Fa, Fb, Fc, Faa, Fbb, Fcc, Faaa, Fbbb, Fccc, Faab, Fbbc, Fcca


def comp_volume_integrals(verts, faces, normals, w):
    C = torch.argmax(normals.abs(), dim=1)
    A = (C + 1) % 3
    B = (A + 1) % 3

    Fa, Fb, Fc, Faa, Fbb, Fcc, Faaa, Fbbb, Fccc, Faab, Fbbc, Fcca = comp_face_integrals(verts, faces, normals, w, A, B,
                                                                                        C)
    T0 = verts.new_zeros(faces.shape[0])
    T0[A == 0] = normals[A == 0, 0] * Fa[A == 0]
    T0[B == 0] = normals[B == 0, 0] * Fb[B == 0]
    T0[C == 0] = normals[C == 0, 0] * Fc[C == 0]

    T0 = T0.sum()

    normA = normals[torch.arange(normals.shape[0]), A]
    normB = normals[torch.arange(normals.shape[0]), B]
    normC = normals[torch.arange(normals.shape[0]), C]

    T1 = verts.new_zeros(faces.shape[0], 3)
    T1[torch.arange(faces.shape[0]), A] = normA * Faa
    T1[torch.arange(faces.shape[0]), B] = normB * Fbb
    T1[torch.arange(faces.shape[0]), C] = normC * Fcc
    T1 = T1.sum(dim=0) / 2

    T2 = verts.new_zeros(faces.shape[0], 3)
    T2[torch.arange(faces.shape[0]), A] = normA * Faaa
    T2[torch.arange(faces.shape[0]), B] = normB * Fbbb
    T2[torch.arange(faces.shape[0]), C] = normC * Fccc
    T2 = T2.sum(dim=0) / 3

    TP = verts.new_zeros(faces.shape[0], 3)
    TP[torch.arange(faces.shape[0]), A] = normA * Faab
    TP[torch.arange(faces.shape[0]), B] = normB * Fbbc
    TP[torch.arange(faces.shape[0]), C] = normC * Fcca
    TP = TP.sum(dim=0) / 2

    return T0, T1, T2, TP


def get_ang_inertia(verts, faces, mass):
    # https://github.com/OpenFOAM/OpenFOAM-2.1.x/blob/master/src/meshTools/momentOfInertia/volumeIntegration/volInt.c
    normals = torch.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 1]], dim=1)
    normals = normals / normals.norm(dim=1).unsqueeze(1)
    w = (-normals * verts[faces[:, 0]]).sum(dim=1)

    T0, T1, T2, TP = comp_volume_integrals(verts, faces, normals, w)

    density = mass / T0

    J = torch.diag(density * (T2[[1, 2, 0]] + T2[[2, 0, 1]]))
    J[0, 1] = J[1, 0] = -density * TP[0]
    J[1, 2] = J[2, 1] = -density * TP[1]
    J[2, 0] = J[0, 2] = -density * TP[2]

    return J

def inertia_diagonalize(inertia):
    eigenvalues, eigenvectors = torch.linalg.eigh(inertia)

    I1, I2, I3 = eigenvalues

    v1, v2, v3 = eigenvectors[:, 0], eigenvectors[:, 1], eigenvectors[:, 2]

    R = torch.stack([v1, v2, v3], dim=1)

    I_prime = R.T @ inertia @ R
    return I_prime

def inertia_diagonalize_np(inertia):
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)

    I1, I2, I3 = eigenvalues

    v1, v2, v3 = eigenvectors[:, 0], eigenvectors[:, 1], eigenvectors[:, 2]

    R = np.stack([v1, v2, v3], axis=1)

    I_prime = R.T @ inertia @ R
    return I_prime
