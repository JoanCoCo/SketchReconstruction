import torch
import random
import math
import matplotlib.pyplot as plt
import pyredner

def round_local(x): 
    return torch.floor(0.5 + x)

def get_mesh_color_index(tri_id, res, i, j):
    res = torch.tensor(res)
    return tri_id * round_local((res + 1) * (res + 2) / 2) + round_local(i * (2 * res - i + 3) / 2) + j

# From shapefromtracing/sft/mesh_colors_to_uv.py
def fibonacci_sphere(samples=1,randomize=True):
    rnd = 1.
    if randomize:
        rnd = random.random() * samples
    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.))
    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y,2))
        phi = ((i + rnd) % samples) * increment
        x = math.cos(phi) * r
        z = math.sin(phi) * r
        points.append(torch.tensor([x,y,z]))
    return points

def plot_values(values, title, xlabel, ylabel, save=None):
    plt.clf()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for key in values:
        plt.plot(values[key])
    plt.legend(values.keys(), loc='upper right')
    if save is None:
        plt.show()
    else:
        plt.savefig(save)

# Determine if the normals have been inversed based on the black level of a white render.
@torch.no_grad()
def inversed_normals(shape, envmap="data/envmap/white_env.exr"):
    shapes = [shape]
    materials = [pyredner.Material(
        diffuse_reflectance=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float, device=pyredner.get_device())
    )]
    envmap = pyredner.imread(envmap)
    if pyredner.get_use_gpu():
        envmap = envmap.cuda()
    envmap = pyredner.EnvironmentMap(envmap)
    camera = pyredner.Camera(
        position=torch.tensor([0.0, 0.0, 6.0], dtype=torch.float32, device=pyredner.get_device()),
        look_at=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=pyredner.get_device()),
        up=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=pyredner.get_device()),
        fov=torch.tensor([50.0]), 
        clip_near=1e-2,
        resolution=(128, 128),
        camera_type=pyredner.camera_type.perspective,
        fisheye=False
    )
    scene = pyredner.Scene(camera, shapes, materials, area_lights=[], envmap=envmap)
    render = pyredner.RenderFunction.apply
    scene_args = pyredner.RenderFunction.serialize_scene(scene=scene, num_samples=4, max_bounces=1, channels=[pyredner.channels.radiance, pyredner.channels.alpha])
    buffer = render(0, *scene_args)
    img = buffer[:, :, :3]
    img = img[buffer[:, :, 3] == 1.0]
    black = img[torch.sum(img, dim=-1) == 0.0]
    print("NORMALS INVERSED TEST: {:d}/{:d}, {}".format(black.shape[0], img.shape[0], float(black.shape[0]) / max(float(img.shape[0]), 1.0) > 0.5))
    return float(black.shape[0]) / max(float(img.shape[0]), 1.0) > 0.5
