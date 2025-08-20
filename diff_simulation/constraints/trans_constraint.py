from diff_simulation.constraints.base import Constraint
import torch

class TransZ_Constraint(Constraint):

    def __init__(self, body_id):
        super().__init__(body1_id = body_id, body2_id = None, constraint_dim = 1)
        
    def J(self):
        J = torch.tensor([
            [0.0,0.0,0.0,0.0,0.0,1.0],
        ])
        return J,None

