import pyredner
import torch
import os
import torch
from PIL import Image
import numpy as np
from utils.generic import fibonacci_sphere

@torch.no_grad()
def save_model(path, name, shape, material=None, mesh_colors=False):
    if not os.path.exists(path):
        os.makedirs(path)
    pyredner.save_obj(shape, os.path.join(path, name + '.obj'), save_material=False)
    if material is not None:
        if mesh_colors:
            torch.save(material.diffuse_reflectance.texels, os.path.join(path, name + ".pt"))
        else:
            pyredner.save_mtl(material, os.path.join(path, name + '.mtl'))

@torch.no_grad()
def load_model(path, name, mesh_colors=False, specular=True):
    shapes = []
    materials = []
    material_map, mesh_list, _ = pyredner.load_obj(os.path.join(path, name + '.obj'))
    for mtl_name, mesh in mesh_list:
        if mesh_colors:
            texels = torch.load(os.path.join(path, name + ".pt")).to(pyredner.get_device())
            diffuse = pyredner.Texture(texels, mesh_colors_resolution=3)
            mat = pyredner.Material(diffuse_reflectance=diffuse, specular_reflectance=material_map[mtl_name].specular_reflectance if specular else None, roughness=None, normal_map=None)
            materials.append(mat)
        else:
            if mtl_name is not None:
                materials.append(material_map[mtl_name])
        shapes.append(pyredner.Shape(
            vertices=mesh.vertices,
            indices=mesh.indices,
            material_id=len(materials)-1,
            uvs = mesh.uvs,
            normals=mesh.normals,
            uv_indices=mesh.uv_indices
        ))
    return (shapes, materials)

@torch.no_grad()
def load_shapes(path, name):
    shapes = []
    _, mesh_list, _ = pyredner.load_obj(os.path.join(path, name + '.obj'))
    i = 0
    for _, mesh in mesh_list:
        shapes.append(pyredner.Shape(
            vertices=mesh.vertices,
            indices=mesh.indices,
            material_id=i,
            uvs = mesh.uvs,
            normals=mesh.normals,
            uv_indices=mesh.uv_indices
        ))
        i += 1
    return shapes

@torch.no_grad()
def load_samples(folder, prefix, number, copy_first=False):
    return load_samples_range(folder, prefix, range(1, number+1), copy_first=copy_first)

@torch.no_grad()
def load_samples_range(folder, prefix, samples_range, copy_first=False):
    targets = []
    if not copy_first:
        for f in samples_range:
            target = np.array(Image.open(os.path.join(folder, '{:s}{:d}.png'.format(prefix, f))).convert('RGBA'), dtype=np.float32) / 255.0
            target = torch.from_numpy(target)
            alpha_map = target[:, :, 3].reshape((target.shape[0], target.shape[1], 1))
            target[:, :, :3] = target[:, :, :3] * alpha_map + torch.ones_like(target[:, :, :3]) * (1 - alpha_map)
            if pyredner.get_use_gpu():
                target = target.cuda()
            targets.append(target)
    else:
        target = np.array(Image.open(os.path.join(folder, '{:s}{:d}.png'.format(prefix, samples_range[0]))).convert('RGBA'), dtype=np.float32) / 255.0
        target = torch.from_numpy(target)
        alpha_map = target[:, :, 3].reshape((target.shape[0], target.shape[1], 1))
        target[:, :, :3] = target[:, :, :3] * alpha_map + torch.ones_like(target[:, :, :3]) * (1 - alpha_map)
        if pyredner.get_use_gpu():
            target = target.cuda()
        for _ in samples_range:
            targets.append(target.clone())
    return targets

@torch.no_grad()
def save_images(folder, name, scenes, samples=4, bounces=1, step=0, overlay=False, targets=[]):
    render = pyredner.RenderFunction.apply
    for s in range(len(scenes)):
        if bounces > 0:
            scene_args=pyredner.RenderFunction.serialize_scene(scene=scenes[s], num_samples=samples, max_bounces=bounces, channels=[pyredner.channels.radiance, pyredner.channels.alpha])
            buffer = render(step, *scene_args)
            img = torch.cat((buffer[:, :, :3], buffer[:, :, 3:]), dim=2)
        else:
            img = pyredner.render_albedo(scenes[s], alpha=True)
        if overlay:
            img[:, :, 0:3] = 0.9 * img[:, :, 0:3] + 0.15 * targets[s][:, :, 0:3]
            img[:, :, 3] = torch.clamp(img[:, :, 3] + 0.15 * targets[s][:, :, 3], 0.0, 1.0)
        pyredner.imwrite(img.cpu(), os.path.join(folder, name + "_s{:d}.png".format(s)))

@torch.no_grad()
def generate_samples(cameras, shapes, materials, envmap="data/envmap/white_env.exr", samples=512, bounces=1):
    scenes = []
    targets = []
    envmap = pyredner.imread(envmap)
    if pyredner.get_use_gpu():
        envmap = envmap.cuda()
    envmap = pyredner.EnvironmentMap(envmap)
    for cam in cameras:
        scene = pyredner.Scene(cam, shapes, materials, area_lights=[], envmap=envmap)
        scenes.append(scene)
    render = pyredner.RenderFunction.apply
    for s in range(len(scenes)):
        if bounces > 0:
            scene_args=pyredner.RenderFunction.serialize_scene(scene=scenes[s], num_samples=samples, max_bounces=bounces, channels=[pyredner.channels.radiance, pyredner.channels.alpha])
            buffer = render(0, *scene_args)
            img = torch.cat((buffer[:, :, :3], buffer[:, :, 3:]), dim=2)
        else:
            img = pyredner.render_albedo(scenes[s], alpha=True)
        targets.append(img.clone().detach())
    return targets
