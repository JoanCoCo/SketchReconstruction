import numpy as np
import torch
import pyredner
from training import losses
from training.remesh import Remesher
from utils.generic import inversed_normals

# Class in charge of the optimization.
class Optimizer:
    def __init__(self, shapes, materials, targets, cameras, param_func, loss="standard", 
                envmap="data/envmap/white_env.exr", lr=5e-3, batch_size=4, use_laplacian=False, 
                remeshing_interval=10, longtail=0, bounces=1):
        self.shapes = shapes
        self.materials = materials
        self.targets = targets
        self.cameras = cameras
        self.envmap = pyredner.imread(envmap)
        if pyredner.get_use_gpu():
            self.envmap = self.envmap.cuda()
        self.envmap = pyredner.EnvironmentMap(self.envmap)
        self.scenes = []
        self.render = pyredner.RenderFunction.apply
        assert (len(targets) == len(cameras))
        self.build_scenes()
        self.param_func = param_func
        self.lr = lr
        self.update_optimizer()
        self.loss = loss
        self.batches = []
        for i in range(0, len(self.targets), batch_size):
            self.batches.append(range(i, min(i + batch_size, len(self.targets))))
        self.laplacian = use_laplacian
        self.remesher = Remesher()
        self.remeshing_interval = remeshing_interval
        self.longtail = longtail
        self.longtail_is_active = False
        self.bounces = bounces

    def build_scenes(self):
        for cam in self.cameras:
            scene = pyredner.Scene(cam, self.shapes, self.materials, area_lights=[], envmap=self.envmap)
            self.scenes.append(scene)
        
    def update_optimizer(self, longtail=False):
        params = self.param_func(self.shapes, self.materials, longtail)
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        if longtail:
            self.scheduler = None
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: max(0.0, 10**((-x)*(1/1000))))
    
    # Optimization step.
    def step(self, iteration, progress, update_normals = True, update_texture = False):
        if self.longtail > 0 and iteration >= self.longtail and not self.longtail_is_active:
            self.update_optimizer(longtail=True)
            self.longtail_is_active = True

        record = {}
        for batch in self.batches:
            self.optimizer.zero_grad()
            loss = torch.tensor(0.0, device=pyredner.get_device())
            for s in batch:
                # Generate renders.
                if self.bounces > 0:
                    scene_args=pyredner.RenderFunction.serialize_scene(scene=self.scenes[s], num_samples=4, max_bounces=self.bounces, channels=[pyredner.channels.radiance, pyredner.channels.alpha])
                    buffer = self.render(iteration, *scene_args)
                    img = torch.cat((buffer[:, :, :3], buffer[:, :, 3:]), dim=2)
                else:
                    img = pyredner.render_albedo(self.scenes[s], alpha=True)
                # Compute the loss from the renders.
                if self.loss == "standard":
                    lap = torch.tensor(0.0, device=pyredner.get_device())
                    norm = torch.tensor(0.0, device=pyredner.get_device())
                    smth = torch.tensor(0.0, device=pyredner.get_device())
                    sprg = torch.tensor(0.0, device=pyredner.get_device())
                    for i in range(len(self.shapes)):
                        if self.laplacian:
                            lap += losses.laplace_regularizer_const(self.shapes[i].vertices, self.shapes[i].indices.long())
                        else:
                            lap += losses.mean_curvature_flow_regularizer(self.shapes[i].vertices, self.shapes[i].indices.long())
                        norm += losses.laplace_regularizer_const(self.shapes[i].normals, self.shapes[i].indices.long())
                        smth += losses.color_smoothness(self.materials[self.shapes[i].material_id].diffuse_reflectance.texels, self.shapes[i].indices.long(), 3)
                        sprg += losses.spring_regularization(self.shapes[i].vertices, self.shapes[i].indices.long(), k=0.0025)
                    if self.laplacian:
                        lap = lap * 1 * (1.0 - max(0.01, (1.0 - 1.5 * progress)))
                    else:
                        lap = lap * 0.02 * (1.0 - max(0.01, (1.0 - 1.5 * progress)))
                    norm = norm * 0.01 * np.clip(np.abs(np.power(np.log(progress + 0.0001), 5.0)), 0.05, 0.4)
                    smth = smth * 0.0002
                    record['shaping'] = record.get('laplace', 0.0) + lap.item()
                    record['norm'] = record.get('norm', 0.0) + norm.item()
                    record['smoothness'] = record.get('smoothness', 0.0) + smth.item()
                    record['spring'] = record.get('spring', 0.0) + sprg.item()
                    sil = losses.silhouette_loss(img, self.targets[s]) * 40 
                    col = losses.color_loss(img, self.targets[s]) * 10 
                    record['silhouette'] = record.get('silhouette', 0.0) + sil.item()
                    record['color'] = record.get('color', 0.0) + col.item()
                    record['total'] = record.get('total', 0.0) + lap.item() + norm.item() + smth.item() + sprg.item() + sil.item() + col.item()
                    loss += lap + norm + smth + sprg + sil + col
                elif self.loss == "color":
                    col = losses.color_loss(img, self.targets[s]) * 10
                    record['color'] = record.get('color', 0.0) + col.item()
                    record['total'] = record.get('total', 0.0) + col.item()
                    loss += col
                elif self.loss == "smooth_color":
                    col = losses.color_loss(img, self.targets[s]) * 10
                    smt = torch.tensor(0.0, device=pyredner.get_device())
                    for i in range(len(self.shapes)):
                        smt += losses.texture_smoothness(self.materials[self.shapes[i].material_id].diffuse_reflectance, self.shapes[i].uvs)
                    smt = smt * 0.002
                    record['color'] = record.get('color', 0.0) + col.item()
                    record['smoothness'] = record.get('smoothness', 0.0) + smt.item()
                    record['total'] = record.get('total', 0.0) + col.item() + smt.item()
                    loss += col + smt
                elif self.loss == "shape":
                    lap = torch.tensor(0.0, device=pyredner.get_device())
                    norm = torch.tensor(0.0, device=pyredner.get_device())
                    for i in range(len(self.shapes)):
                        lap += losses.laplace_regularizer_const(self.shapes[i].vertices, self.shapes[i].indices.long()) 
                        norm += losses.laplace_regularizer_const(self.shapes[i].normals, self.shapes[i].indices.long())
                    lap = lap * 1 * max(0.01, (1.0 - 1.5 * progress))
                    norm = norm * 0.01 * np.clip(np.abs(np.power(np.log(progress + 0.0001), 5.0)), 0.05, 0.4)
                    record['laplace'] = record.get('laplace', 0.0) + lap.item()
                    record['norm'] = record.get('norm', 0.0) + norm.item()
                    sil = losses.silhouette_loss(img, self.targets[s]) * 40
                    record['silhouette'] = record.get('silhouette', 0.0) + sil.item()
                    record['total'] = record.get('total', 0.0) + lap.item() + norm.item() + sil.item()
                    loss += lap + norm + sil
            print("Loss: {}".format(loss.item()))

            previous_vertices = self.shapes[0].vertices.clone().detach()
            previous_vertices.requires_grad = False

            # Update the parameters.
            loss.backward()
            self.optimizer.step()

            # Catch possible infinities that lead to errors.
            if not torch.isfinite(self.shapes[0].vertices).all():
                print("\nWARNINIG: {:.0f} infinite vertices caught.\n".format(torch.sum(1.0 - torch.isfinite(self.shapes[0].vertices).to(torch.float)).item()))
                inft = torch.logical_not(torch.isfinite(self.shapes[0].vertices)).to(torch.float32)
                self.shapes[0].vertices = torch.nan_to_num(self.shapes[0].vertices.clone().detach(), posinf=0.0, neginf=0.0) * (1.0 - inft) + (inft) * previous_vertices
                self.update_optimizer(longtail=(self.longtail > 0 and iteration == self.longtail))
                del inft
            del previous_vertices

            torch.cuda.empty_cache()

            if update_normals:
                for i in range(0, len(self.shapes)):
                    self.shapes[i].normals = pyredner.compute_vertex_normal(self.shapes[i].vertices, self.shapes[i].indices)   
                    self.shapes[i].normals = (-1.0 if inversed_normals(self.shapes[i]) else 1.0) * self.shapes[i].normals 
            for m in range(0, len(self.materials)):
                self.materials[m].diffuse_reflectance.texels.data.clamp_(0.0, 1.0)
                if update_texture:
                    self.materials[m].diffuse_reflectance.generate_mipmap()

        # Apply remeshing, if appropiate.
        if (self.remeshing_interval > 0 and iteration > 0 and iteration % self.remeshing_interval == 0 and (self.longtail == 0 or iteration <= self.longtail)) or (self.remeshing_interval > 0 and self.longtail > 0 and iteration == self.longtail):
            print("Applying remeshing...")
            for i in range(0, len(self.shapes)):
                new_shape, new_material = self.remesher.remesh(self.shapes[i], self.materials[i])
                self.shapes[i] = new_shape
                self.materials[i] = new_material
            for s in range(0, len(self.scenes)):
                self.scenes[s].shapes = self.shapes
                self.scenes[s].materials = self.materials
            self.update_optimizer(longtail=(self.longtail > 0 and iteration == self.longtail))
            if update_normals:
                for i in range(0, len(self.shapes)):
                    self.shapes[i].normals = pyredner.compute_vertex_normal(self.shapes[i].vertices, self.shapes[i].indices)
                    self.shapes[i].normals = (-1.0 if inversed_normals(self.shapes[i]) else 1.0) * self.shapes[i].normals 

        for key in record.keys():
            record[key] = record[key] / float(len(self.scenes))
        return record