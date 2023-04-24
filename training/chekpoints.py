import os
import torch
import pyredner
import json
from utils.resources import save_model, load_model

# Manager for storing and loading the checkpoints of the optimization.
class CheckpointManager:
    def __init__(self, mesh_colors = True, specular=False, source_folder="checkpoints"):
        self.source = source_folder
        self.mesh_colors = mesh_colors
        self.specular = specular
        if os.path.exists(os.path.join(self.source, "info.json")):
            with open(os.path.join(self.source, "info.json"), "r") as info_file:
                self.info = json.load(info_file)
                info_file.close()
        else:
            self.info = {'last_id':-1, 'loss':torch.inf, 'shapes':0}
            with open(os.path.join(self.source, "info.json"), "w") as info_file:
                json.dump(self.info, info_file)
                info_file.close()
    
    def save(self, shapes, materials, loss=torch.inf, only_save_best=False):
        if (not only_save_best) or (loss < self.info['loss']):
            for i in range(0, len(shapes)):
                save_model(self.source, "point-s{}-{}".format(i, self.info['last_id']+1), shapes[i], materials[shapes[i].material_id], mesh_colors=self.mesh_colors)

            self.info['last_id'] += 1
            self.info['loss'] = loss
            self.info['shapes'] = len(shapes)
            with open(os.path.join(self.source, "info.json"), "w") as info_file:
                json.dump(self.info, info_file) 
                info_file.close()

    def load(self, id=None):
        shapes = []
        materials = []
        if id is None:
            id = self.info['last_id']
        for s in range(self.info['shapes']):
            shps, mats = load_model(self.source, "point-s{}-{}".format(s, id), mesh_colors=self.mesh_colors, specular=self.specular)
            shapes.append(shps[0])
            materials.append(mats[0])
        return shapes, materials