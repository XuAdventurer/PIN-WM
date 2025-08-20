from diff_simulation.constraints.base import Constraint
import torch

class Rot_Constraint(Constraint):

    def __init__(self, body_id):
        super().__init__(body1_id = body_id, body2_id = None, constraint_dim = 3)
        
    def J(self):
        J_rot = torch.eye(3)
        J_trans = torch.zeros([3, 3])
        J = torch.cat([J_rot, J_trans], dim=1)        
        return J,None
