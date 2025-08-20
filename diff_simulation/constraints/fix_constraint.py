from diff_simulation.constraints.base import Constraint
import torch

class Fix_Constraint(Constraint):

    def __init__(self, body_id):
        super().__init__(body1_id = body_id, body2_id = None, constraint_dim = 6)
        
    def J(self):
        return torch.eye(self.constraint_dim),None
