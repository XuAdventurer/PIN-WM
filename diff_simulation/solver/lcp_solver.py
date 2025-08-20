
import torch
from torch.autograd import Function

from diff_simulation.constraints.base import Constraint
from torch.nn.functional import normalize
from diff_simulation.solver.util import bger, expandParam, extract_nBatch,orthogonal
import diff_simulation.solver.cvxpy as cvxpy
import diff_simulation.solver.batch as pdipm_b


from enum import Enum

class QPSolvers(Enum):
    PDIPM_BATCHED = 1
    CVXPY = 2

class LCP_Solver:

    def __init__(self,simulator):
        from diff_simulation.simulator import Simulator
        self.simulator : Simulator = simulator
        self.device = simulator.device

    def Je(self):
        sum_constraint_dim = self.simulator.get_sum_constraint_dim()
        Je = torch.zeros((sum_constraint_dim,self.simulator.velocity_dim * len(self.simulator.bodies)),device=self.device)
        row = 0
        for joint in self.simulator.joints:
            J1,J2 = joint.J()
            body1_index = self.simulator.get_body_list_index(joint.body1_id)
            body2_index = self.simulator.get_body_list_index(joint.body2_id)
            Je[row:row + J1.size(0),body1_index * self.simulator.velocity_dim:(body1_index + 1) * 
                                                                self.simulator.velocity_dim] = J1       
            if J2 is not None:
                Je[row:row + J2.size(0),
                body2_index * self.simulator.velocity_dim:(body2_index + 1) * 
                                            self.simulator.velocity_dim] = J2     
            row += J1.size(0)
        return Je            

    def Jc(self,contact_infos):
        Jc = torch.zeros((len(contact_infos), self.simulator.velocity_dim * len(self.simulator.bodies)),device=self.device)
        for i, contact_info in enumerate(contact_infos):
            c = contact_info[0]
            normal = c[0]
            p_a,p_b = c[1],c[2]
            body1_index = self.simulator.get_body_list_index(contact_info[1])
            body2_index = self.simulator.get_body_list_index(contact_info[2])
            J1 = torch.cat([torch.cross(p_a, normal), normal])
            J2 = - torch.cat([torch.cross(p_b, normal), normal])
            Jc[i, body1_index * self.simulator.velocity_dim:(body1_index + 1) * self.simulator.velocity_dim] = J1
            Jc[i, body2_index * self.simulator.velocity_dim:(body2_index + 1) * self.simulator.velocity_dim] = J2        
        return Jc
    
    def Jf(self,contact_infos):
        Jf = torch.zeros((len(contact_infos) * self.simulator.fric_dirs,
                        self.simulator.velocity_dim * len(self.simulator.bodies)),device=self.device)
        for i, contact_info in enumerate(contact_infos):
            c = contact_info[0]
            body1_index = self.simulator.get_body_list_index(contact_info[1])
            body2_index = self.simulator.get_body_list_index(contact_info[2])
            dir1 = normalize(orthogonal(c[0]), dim=0)
            dir2 = normalize(torch.cross(dir1, c[0]), dim=0)
            dirs = torch.stack([dir1, dir2])
            if self.simulator.fric_dirs == 8:
                dir3 = normalize(dir1 + dir2, dim=0)
                dir4 = normalize(torch.cross(dir3, c[0]), dim=0)

                dirs = torch.cat([dirs,
                                  torch.stack([dir3, dir4])
                                  ], dim=0)
            dirs = torch.cat([dirs, -dirs], dim=0)

            J1 = torch.cat([torch.cross(c[1].expand(self.simulator.fric_dirs, -1), dirs), dirs], dim=1)
            J2 = torch.cat([torch.cross(c[2].expand(self.simulator.fric_dirs, -1), dirs), dirs], dim=1)

            Jf[i * self.simulator.fric_dirs:(i + 1) * self.simulator.fric_dirs, body1_index * self.simulator.velocity_dim:(body1_index + 1)
                                                                                     * self.simulator.velocity_dim] = J1
            Jf[i * self.simulator.fric_dirs:(i + 1) * self.simulator.fric_dirs, body2_index * self.simulator.velocity_dim:(body2_index + 1)
                                                                                     * self.simulator.velocity_dim] = -J2
        return Jf
        

    def M(self):
        M = torch.block_diag(*[b.get_M_world() for b in self.simulator.bodies])
        return M
    
    def E(self,contact_infos):
        num_contacts = len(contact_infos)
        n = self.simulator.fric_dirs * num_contacts
        E = torch.zeros((n, num_contacts),device=self.device)
        for i in range(num_contacts):
            E[i * self.simulator.fric_dirs: (i + 1) * self.simulator.fric_dirs, i] += 1
        return E

    def restitutions(self,contact_infos):
        restitutions = torch.zeros((len(contact_infos)),device=self.device)
        for i, contact_info in enumerate(contact_infos):
            body1_index = self.simulator.get_body_list_index(contact_info[1])
            body2_index = self.simulator.get_body_list_index(contact_info[2])
            r1 = self.simulator.bodies[body1_index].restitution
            r2 = self.simulator.bodies[body2_index].restitution
            # restitutions[i] = (r1 + r2) / 2
            restitutions[i] = (r1 * r2)
            # restitutions[i] = math.sqrt(r1 * r2)
        return restitutions

    def mu(self,contact_infos):
        mu = torch.zeros((len(contact_infos)),device=self.device)
        for i, contact_info in enumerate(contact_infos):
            body1_index = self.simulator.get_body_list_index(contact_info[1])
            body2_index = self.simulator.get_body_list_index(contact_info[2])
            # mu[i] = torch.sqrt(self.bodies[i1].fric_coeff * self.bodies[i2].fric_coeff)
            f1 = self.simulator.bodies[body1_index].friction_coefficient
            f2 = self.simulator.bodies[body2_index].friction_coefficient
            mu[i] = f1 * f2
            # mu[i] = 0.5 * (self.simulator.bodies[body1_index].friction_coefficient + 
            #                self.simulator.bodies[body2_index].friction_coefficient)
        return torch.diag(mu)        

    def solve_constraint(self,contact_infos):
        if not contact_infos:
            inv,u = self.create_lcp_no_contact()
            x = self.solve_lcp_no_contact(inv,u)
        else:
            M, u, G, h, Je, b, F = self.create_lcp_with_contact(contact_infos)
            x = - self.solve_lcp_with_contact(max_iter=10, verbose=-1)(M, u, G, h, Je, b, F)
            x = x.to(torch.float32)
        return x
        
    def create_lcp_no_contact(self):
        M = self.M()
        f = self.simulator.apply_external_forces(self.simulator.cur_time)
        u = torch.matmul(M,self.simulator.get_vel_vec()) + self.simulator.dtime * f     
        if len(self.simulator.joints) > 0:
            Je = self.Je()
            sum_constraint_dim = Je.size(0)
            u = torch.cat([u, u.new_zeros(sum_constraint_dim)])
            P = torch.cat([torch.cat([M, -Je.t()], dim=1),
                            torch.cat([Je, Je.new_zeros(sum_constraint_dim,
                                                         sum_constraint_dim)],dim=1)])        
        else :
            P = M
        inv = torch.inverse(P)
        return inv,u

    def solve_lcp_no_contact(self,inv,u):
        return torch.matmul(inv, u)

    def create_lcp_with_contact(self,contact_infos):
        Jc = self.Jc(contact_infos)
        vel_vec = self.simulator.get_vel_vec()
        v = torch.matmul(Jc, vel_vec) * self.restitutions(contact_infos) # c
        M = self.M()
        f = self.simulator.apply_external_forces(self.simulator.cur_time)
        u = torch.matmul(M,vel_vec) + self.simulator.dtime * f  
        if len(self.simulator.joints) > 0:
            Je = self.Je()
            b = Je.new_zeros(Je.size(0)).unsqueeze(0)
            Je = Je.unsqueeze(0)
        else:
            b = torch.tensor([])
            Je = torch.tensor([])         
        Jc = Jc.unsqueeze(0)
        v = v.unsqueeze(0)
        E = self.E(contact_infos).unsqueeze(0)
        mu = self.mu(contact_infos).unsqueeze(0)
        Jf = self.Jf(contact_infos).unsqueeze(0)        
        G = torch.cat([Jc, Jf,
                    Jf.new_zeros(Jf.size(0), mu.size(1), Jf.size(2))], dim=1)
        F = G.new_zeros(G.size(1), G.size(1)).unsqueeze(0)
        F[:, Jc.size(1):-E.size(2), -E.size(2):] = E
        F[:, -mu.size(1):, :mu.size(2)] = mu
        F[:, -mu.size(1):, mu.size(2):mu.size(2) + E.size(1)] = \
            -E.transpose(1, 2)
        h = torch.cat([v, v.new_zeros(v.size(0), Jf.size(1) + mu.size(1))], 1)
        return M, u, G, h, Je, b, F

    def solve_lcp_with_contact(self,eps=1e-12, verbose=0, notImprovedLim=3,
                        max_iter=20, solver=1,
                        check_Q_spd=True):
        class LCPFunctionFn(Function):
            @staticmethod
            def forward(ctx, Q_, p_, G_, h_, A_, b_, F_):
                """Solve a batch of QPs.

                This function solves a batch of QPs, each optimizing over
                `nz` variables and having `nineq` inequality constraints
                and `neq` equality constraints.
                The optimization problem for each instance in the batch
                (dropping indexing from the notation) is of the form

                    \hat z =   argmin_z 1/2 z^T Q z + p^T z
                            subject to Gz <= h
                                        Az  = b

                where Q \in S^{nz,nz},
                    S^{nz,nz} is the set of all positive semi-definite matrices,
                    p \in R^{nz}
                    G \in R^{nineq,nz}
                    h \in R^{nineq}
                    A \in R^{neq,nz}
                    b \in R^{neq}
                    F \in R^{nz}

                These parameters should all be passed to this function as
                Variable- or Parameter-wrapped Tensors.
                (See torch.autograd.Variable and torch.nn.parameter.Parameter)

                If you want to solve a batch of QPs where `nz`, `nineq` and `neq`
                are the same, but some of the contents differ across the
                minibatch, you can pass in tensors in the standard way
                where the first dimension indicates the batch example.
                This can be done with some or all of the coefficients.

                You do not need to add an extra dimension to coefficients
                that will not change across all of the minibatch examples.
                This function is able to infer such cases.

                If you don't want to use any equality or inequality constraints,
                you can set the appropriate values to:

                    e = Variable(torch.Tensor())

                Parameters:
                Q:  A (nBatch, nz, nz) or (nz, nz) Tensor.
                p:  A (nBatch, nz) or (nz) Tensor.
                G:  A (nBatch, nineq, nz) or (nineq, nz) Tensor.
                h:  A (nBatch, nineq) or (nineq) Tensor.
                A:  A (nBatch, neq, nz) or (neq, nz) Tensor.
                b:  A (nBatch, neq) or (neq) Tensor.
                F:  A (nBatch, nz) or (nz) Tensor.

                Returns: \hat z: a (nBatch, nz) Tensor.
                """
                Q_ = Q_.to(torch.float64)
                p_ = p_.to(torch.float64)
                G_ = G_.to(torch.float64)
                h_ = h_.to(torch.float64)
                A_ = A_.to(torch.float64)
                b_ = b_.to(torch.float64)
                F_ = F_.to(torch.float64)
                nBatch = extract_nBatch(Q_, p_, G_, h_, A_, b_, F_)
                Q, _ = expandParam(Q_, nBatch, 3)
                p, _ = expandParam(p_, nBatch, 2)
                G, _ = expandParam(G_, nBatch, 3)
                h, _ = expandParam(h_, nBatch, 2)
                A, _ = expandParam(A_, nBatch, 3)
                b, _ = expandParam(b_, nBatch, 2)
                F, _ = expandParam(F_, nBatch, 3)

                # Q = Q.to_sparse()

                if check_Q_spd:
                    try:
                        torch.linalg.cholesky(Q)
                    except:
                        raise RuntimeError('Q is not SPD.')

                _, nineq, nz = G.size()
                neq = A.size(1) if A.nelement() > 0 else 0
                assert(neq > 0 or nineq > 0)
                ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz
                
                if solver == QPSolvers.PDIPM_BATCHED.value:
                    ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A, F)
                    zhats, ctx.nus, ctx.lams, ctx.slacks = pdipm_b.forward(
                        Q, p, G, h, A, b, F, ctx.Q_LU, ctx.S_LU, ctx.R,
                        eps, verbose, notImprovedLim, max_iter)                        
 
                ctx.save_for_backward(zhats, Q_, p_, G_, h_, A_, b_, F_)
                return zhats

            @staticmethod
            def backward(ctx, dl_dzhat):
                zhats, Q, p, G, h, A, b, F = ctx.saved_tensors
                nBatch = extract_nBatch(Q, p, G, h, A, b, F)
                Q, Q_e = expandParam(Q, nBatch, 3)
                p, p_e = expandParam(p, nBatch, 2)
                G, G_e = expandParam(G, nBatch, 3)
                h, h_e = expandParam(h, nBatch, 2)
                A, A_e = expandParam(A, nBatch, 3)
                b, b_e = expandParam(b, nBatch, 2)
                F, F_e = expandParam(F, nBatch, 3)

                neq, nineq = ctx.neq, ctx.nineq


                # Clamp here to avoid issues coming up when the slacks are too small.
                # TODO: A better fix would be to get lams and slacks from the solver that don't have this issue.
                d = torch.clamp(ctx.lams, min=1e-8) / torch.clamp(ctx.slacks, min=1e-8)
                pdipm_b.factor_kkt(ctx.S_LU, ctx.R, d)

                dx, _, dlam, dnu = pdipm_b.solve_kkt(
                    ctx.Q_LU, d, G, A, ctx.S_LU,
                    dl_dzhat, torch.zeros((nBatch, nineq),device=self.device).type_as(G),
                    torch.zeros((nBatch, nineq),device=self.device).type_as(G),
                    torch.zeros((nBatch, neq),device=self.device).type_as(G) if neq > 0 else torch.Tensor().to(self.device))

                dps = dx
                dFs = bger(dlam, ctx.lams)
                if F_e: 
                    dFs = dFs.mean(0)
                dGs = bger(dlam, zhats) + bger(ctx.lams, dx)
                if G_e: 
                    dGs = dGs.mean(0)
                dhs = -dlam
                if h_e:
                    dhs = dhs.mean(0)
                if neq > 0:
                    dAs = bger(dnu, zhats) + bger(ctx.nus, dx)
                    dbs = -dnu
                    if A_e:
                        dAs = dAs.mean(0)
                    if b_e:
                        dbs = dbs.mean(0)
                else:
                    dAs, dbs = None, None
                dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))
                if Q_e:
                    dQs = dQs.mean(0)
                if p_e:
                    dps = dps.mean(0)


                grads = (dQs, dps, dGs, dhs, dAs, dbs, dFs)

                return grads
        return LCPFunctionFn.apply
