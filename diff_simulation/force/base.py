from abc import ABCMeta, abstractmethod
import torch

class Force(metaclass=ABCMeta):
    
    def __init__(self,direction,magnitude=10.0,starttime=0.0,endtime=1e5,):
        self.direction = direction
        self.magnitude = magnitude
        self.starttime = starttime
        self.endtime = endtime

    def apply(self, cur_time):
        if cur_time < self.starttime or cur_time > self.endtime:
            return self.direction * 0
        else:
            return self.force_function()
        
    @abstractmethod
    def force_function(self, *args, **kwargs):
        raise NotImplementedError
    