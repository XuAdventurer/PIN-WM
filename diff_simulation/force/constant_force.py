from .base import Force

class Constant_Force(Force):
    def __init__(self,direction,magnitude,starttime=0.0,endtime=1e5,):
        super().__init__(direction, magnitude, starttime, endtime)

    def force_function(self):
        return self.direction * self.magnitude    