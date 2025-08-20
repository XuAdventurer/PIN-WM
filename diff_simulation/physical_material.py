import torch
import random
import numpy as np
class  Physical_Materials:
    def __init__(self,requires_grad,device):
        self.requires_grad = requires_grad
        self.body_id = None
        self.device = device
        self.all = {}
        self.all["restitution"] = torch.nn.Parameter(torch.tensor(random.uniform(-5,-3),device=self.device),requires_grad=self.requires_grad)
        self.all["friction_coefficient"] = torch.nn.Parameter(torch.tensor(random.uniform(-5,5),device=self.device),requires_grad=self.requires_grad)
        self.all["mass"] = torch.nn.Parameter(torch.tensor(random.uniform(0.001,1),device=self.device),requires_grad=self.requires_grad)
        self.all["inertia"] = torch.nn.Parameter(0.1*torch.ones((3),device=self.device),requires_grad=self.requires_grad)


    def set_material(self,material_name,value):
        if material_name == "restitution":
            self.all[material_name].data = torch.tensor(np.log(value / (1 - value)),device=self.device)
        elif material_name == "friction_coefficient":
            self.all[material_name].data = torch.tensor(np.log(value / (1 - value)),device=self.device)
        elif material_name == "mass":
            self.all[material_name].data = torch.tensor(value,device=self.device)
        elif material_name == "inertia":
            self.all[material_name].data = torch.tensor([value[0][0],value[1][1],value[2][2]],device=self.device)

    def get_material(self,material_name):
        if material_name == "restitution":
            return torch.sigmoid(self.all[material_name])
        elif material_name == "friction_coefficient":
            return torch.sigmoid(self.all[material_name]) 
        elif material_name == "mass":
            return torch.clip(self.all[material_name],0.1)
        elif material_name == 'inertia':

            inertia = torch.zeros((3,3),device=self.device)
            inertia[0,0] = torch.clip(self.all[material_name][0],1e-3)
            inertia[1,1] = torch.clip(self.all[material_name][1],1e-3)
            inertia[2,2] = torch.clip(self.all[material_name][2],1e-3)

            return inertia

        
    def get_material_num(self):
        return len(self.all)
    
    def get_material_names(self):
        return self.all.keys()
    
    def get_original_json_dict(self):
        json_dict = {"body_id":self.body_id}
        for key in self.all.keys():
            json_dict[key] = self.all[key].tolist()
        return json_dict
    
    def get_activate_json_dict(self):
        json_dict = {"body_id":self.body_id}
        for key in self.all.keys():
            json_dict[key] = self.get_material(key).tolist()
        return json_dict
    
    def no_optimize(self,material_name):
        self.all[material_name].requires_grad = False

    def add_noise(self):    
        noise = (2*random.random()-1)
        self.all["mass"] += noise
        noise = (2*random.random()-1)
        self.all["friction_coefficient"] += noise
        noise = (2*random.random()-1)
        self.all["restitution"] += noise








