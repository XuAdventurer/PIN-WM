from abc import ABCMeta, abstractmethod

from enum import Enum
class Joint_Type(Enum):
    FIX_CONSTRAINT = 0
    NO_ROT_CONSTRATNT = 1
    NO_TRANS_Z_CONSTRATNT = 2

class Constraint(metaclass=ABCMeta):
    def __init__(self,body1_id,body2_id,constraint_dim):
        self.id = None
        self.body1_id = body1_id
        self.body2_id = body2_id
        self.constraint_dim = constraint_dim

    @abstractmethod
    def J(self, *args, **kwargs):
        pass

    def set_id(self,id):
        self.id = id