import os
import time
import torch
import json
import numpy as np
import pyredner
import argparse
import random
import math
from PIL import Image
import matplotlib.pyplot as plt
from utils.resources import load_model, load_shapes, save_model, load_samples, save_images, generate_samples
from training.chekpoints import CheckpointManager
from training.optimize import Optimizer
from training.status import TrainingStatus
from utils.generic import plot_values, fibonacci_sphere, inversed_normals

torch.manual_seed(98543)
np.random.seed(3251)
random.seed(8732)

MESH_COLORS_RESOLUTION = 3 # DO NOT CHANGE WITHOUT UPDATING OTHER FILES

if __name__ == "__main__":
    #### PARSE ARGUMENTS ####
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('-i', '--iterations', type=int, default=500)
    parser.add_argument('-f', '--frames', type=int, default=1)
    parser.add_argument('--samples-folder', type=str, default="frames")
    parser.add_argument('--samples-prefix', type=str, default="img_")
    parser.add_argument('--checkpoints-folder', type=str, default="checkpoints")
    parser.add_argument('--load-checkpoint', type=bool, default=False)
    parser.add_argument('--mesh', type=str, default="mesh/mesh.obj")
    parser.add_argument('--envmap', type=str, default="data/envmap/white_env.exr")
    parser.add_argument('--steps-folder', type=str, default="steps")
    parser.add_argument('--saving-interval', type=int, default=3)
    parser.add_argument('--output-folder', type=str, default="results")
    parser.add_argument('--output-name', type=str, default="result")
    parser.add_argument('--texture-samples', type=int, default=64)
    parser.add_argument('--texture-refinement-steps', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--use-laplacian', type=bool, default=False)
    parser.add_argument('--remeshing-interval', type=int, default=10)
    parser.add_argument('--longtail', type=int, default=0)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--overlay-target', type=bool, default=False)
    parser.add_argument('--flip-normals', type=bool, default=False)
    parser.add_argument('--free-rotation', type=bool, default=False)
    parser.add_argument('--views', type=str, default=None)
    parser.add_argument('--cameras', type=str, default=None)
    parser.add_argument('--bounces', type=int, default=1)
    ARGS = parser.parse_args()

    if ARGS.config is not None:
        data = json.load(open(ARGS.config, 'r'))
        for key in data:
            ARGS.__dict__[key] = data[key]

    #### LOAD TARGETS ####
    targets = load_samples(ARGS.samples_folder, ARGS.samples_prefix, ARGS.frames, copy_first=ARGS.free_rotation)
    
    #### SET UP CAMERAS ####
    views =[]
    if ARGS.views is not None:
        vdata = json.load(open(ARGS.views, 'r'))
        for key in vdata:
            views.append(torch.tensor(vdata[key]["matrix"], dtype=torch.float32, device=pyredner.get_device()))
    cposition = []
    cups = []
    is_gonna_be_ortho = False
    if ARGS.cameras is not None:
        cdata = json.load(open(ARGS.cameras, 'r'))
        cposition = cdata["positions"]
        cups = cdata["ups"]
        is_gonna_be_ortho = ("type" in cdata) and (cdata["type"] == "ORTHO")
    cameras = []
    for frame in range(ARGS.frames):
        if ARGS.cameras is None:
            rotation = pyredner.gen_rotate_matrix(torch.tensor([0.0, 2.0 * np.pi / ARGS.frames * frame, 0.0], device=pyredner.get_device()))
            if ARGS.views is not None:
                rotation = views[frame]
            elif ARGS.free_rotation:
                rotation = pyredner.gen_rotate_matrix(torch.rand(3, device=pyredner.get_device()) * 2.0 * torch.pi)
            cam_pos = torch.transpose(rotation, 0, 1) @ torch.tensor([0.0, 0.0, 1.0], device=pyredner.get_device()).reshape(3, 1)
            up_vec = torch.transpose(rotation, 0, 1) @ torch.tensor([0.0, 1.0, 1.0], device=pyredner.get_device()).reshape(3, 1)
            cam_pos = torch.reshape(cam_pos, (3,))
            up_vec = torch.reshape(up_vec, (3,)) - cam_pos
            up_vec = up_vec / torch.norm(up_vec, p=2)
            cam_pos = 2.0 * (cam_pos / torch.norm(cam_pos, p=2))
            cam = pyredner.Camera(position=cam_pos, 
                                look_at=torch.tensor([0.0, 0.0, 0.0]), 
                                up=up_vec, 
                                fov=torch.tensor([45.0]), 
                                clip_near=1e-2,
                                resolution=(targets[0].shape[0], targets[0].shape[1]),
                                camera_type=pyredner.camera_type.orthographic,
                                fisheye=False)
            cameras.append(cam)
        else:
            cam = pyredner.Camera(position=torch.tensor(cposition[frame], device=pyredner.get_device()), 
                                look_at=torch.tensor([0.0, 0.0, 0.0]), 
                                up=torch.tensor(cups[frame], device=pyredner.get_device()), 
                                fov=torch.tensor([50.0]), 
                                clip_near=1e-2,
                                resolution=(targets[0].shape[0], targets[0].shape[1]),
                                camera_type=pyredner.camera_type.perspective if not is_gonna_be_ortho else pyredner.camera_type.orthographic,
                                fisheye=False)
            cameras.append(cam)
    
    #### LOAD BASE MODEL AND PREPARE IT ####
    shapes = []
    materials = []
    if not os.path.exists(ARGS.checkpoints_folder):
        os.makedirs(ARGS.checkpoints_folder)
    check_manager = CheckpointManager(mesh_colors=True, source_folder=ARGS.checkpoints_folder)
    if ARGS.load_checkpoint and check_manager.info['last_id'] >= 0:
        print("Loading checkpoint {:d}".format(check_manager.info['last_id']))
        shapes, materials = check_manager.load()
        for i in range(0, len(shapes)):
            shapes[i].normals = pyredner.compute_vertex_normal(shapes[i].vertices, shapes[i].indices)
            shapes[i].normals = (-1.0 if inversed_normals(shapes[i]) else 1.0) * shapes[i].normals
            shapes[i].vertices.requires_grad = True
            materials[i].diffuse_reflectance.texels.requires_grad = True
    else:
        shapes = load_shapes(os.path.dirname(ARGS.mesh), ARGS.mesh.split("/")[-1].split(".")[0])
        for i in range(0, len(shapes)):
            shapes[i].normals = pyredner.compute_vertex_normal(shapes[i].vertices, shapes[i].indices)
            shapes[i].normals = (-1.0 if inversed_normals(shapes[i]) else 1.0) * shapes[i].normals
            texels = torch.randn([shapes[i].indices.shape[0] * int(((MESH_COLORS_RESOLUTION+1) * (MESH_COLORS_RESOLUTION+2)) / 2) * 3], device=pyredner.get_device(), requires_grad=True)
            diffuse = pyredner.Texture(texels, mesh_colors_resolution=MESH_COLORS_RESOLUTION)
            mat = pyredner.Material(diffuse_reflectance=diffuse, specular_reflectance=None, roughness=None, normal_map=None)
            materials.append(mat)
            shapes[i].vertices.requires_grad = True
            materials[i].diffuse_reflectance.texels.requires_grad = True
    
    #### OPTIMIZE ####
    status = TrainingStatus(load=ARGS.load_checkpoint)
    if not os.path.exists(ARGS.steps_folder):
        os.makedirs(ARGS.steps_folder)
    optimizer = None
    def param_fuc(shapes, materials, is_longtail=False):
        if not is_longtail:
            for i in range(len(shapes)):
                shapes[i].vertices.requires_grad = True
            for i in range(len(materials)):
                materials[i].diffuse_reflectance.texels.requires_grad = True
            return  [materials[i].diffuse_reflectance.texels for i in range(len(materials))] + [shapes[i].vertices for i in range(len(shapes))]
        else:
            for i in range(len(shapes)):
                shapes[i].vertices.requires_grad = False
            for i in range(len(materials)):
                materials[i].diffuse_reflectance.texels.requires_grad = True
            return  [materials[i].diffuse_reflectance.texels for i in range(len(materials))]
    optimizer = Optimizer(shapes, materials, targets, cameras, param_fuc, 
                          envmap=ARGS.envmap, batch_size=ARGS.batch_size, use_laplacian=ARGS.use_laplacian, 
                          remeshing_interval=ARGS.remeshing_interval, longtail=ARGS.longtail, bounces=ARGS.bounces)

    if status.status['iteration']+1 == 0:
        save_images(ARGS.steps_folder, "init", optimizer.scenes, bounces=ARGS.bounces)
    if status.status['iteration']+1 < ARGS.iterations:
        start_time = time.time()
        for it in range(status.status['iteration']+1, ARGS.iterations):
            print("Iteration {}".format(it))
            status.status['iteration'] = it
            record = optimizer.step(it, progress=float(it)/float(ARGS.iterations), update_normals=True, update_texture=False)
            for key in record.keys():
                status.status['losses'][key] = status.status['losses'].get(key, []) + [record[key]]
            if it % ARGS.saving_interval == 0:
                check_manager.save(optimizer.shapes, optimizer.materials, record['total'], only_save_best=False)
                save_images(ARGS.steps_folder, "step_{:d}".format(it), optimizer.scenes, overlay=ARGS.overlay_target, targets=targets, bounces=ARGS.bounces)
                status.save()
        end_time = time.time()
    
        #### SAVE RESULTS ####
        if not os.path.exists(os.path.join(ARGS.output_folder, "mesh")):
            os.makedirs(os.path.join(ARGS.output_folder, "mesh"))
        plot_values(status.status['losses'], "Losses", "Iteration", "Loss", save=os.path.join(ARGS.output_folder, "losses.png"))
        save_images(ARGS.output_folder, ARGS.output_name, optimizer.scenes, samples=512, bounces=ARGS.bounces)
        for i in range(len(optimizer.shapes)):
            save_model(os.path.join(ARGS.output_folder, "mesh"), ARGS.output_name + "_{:d}".format(i), optimizer.shapes[i], optimizer.materials[i], mesh_colors=True)

        print("Elapsed time: {:.2f} min".format((end_time - start_time)/60.0))
        print("\n\n")
    
    #### GENERATE SAMPLES FOR TEXTURE GENERATION ####
    camera_positions = fibonacci_sphere(ARGS.texture_samples)
    rnd_cameras = []
    for pos in camera_positions:
        cam = pyredner.Camera(position = 2.5 * pos,
                                 look_at = torch.tensor([0.0, 0.0, 0.0]),
                                 up = torch.tensor([0.0, 1.0, 0.0]),
                                 fov=torch.tensor([45.0]), 
                                 clip_near=1e-2,
                                 resolution=(targets[0].shape[0], targets[0].shape[1]),
                                 camera_type=pyredner.camera_type.perspective,
                                 fisheye=False)
        rnd_cameras.append(cam)
    texture_targets = generate_samples(rnd_cameras, shapes, materials, bounces=ARGS.bounces)
    for i, t in enumerate(texture_targets):
        pyredner.imwrite(t.cpu(), os.path.join(ARGS.steps_folder, "target_{:d}.png".format(i)))

    #### FIX TOPOLOGY AND GENERATE TEXTURES ####
    fixed_shapes = []
    uv_materials = []
    for i in range(len(shapes)):
        texture = torch.randn(2048, 2048, 3, device=pyredner.get_device(), requires_grad=True)
        diffuse_uv = pyredner.Texture(texture, mesh_colors_resolution=0)
        mat2 = pyredner.Material(diffuse_reflectance=diffuse_uv, specular_reflectance=None, roughness=None, normal_map=None)
        mat2.diffuse_reflectance.generate_mipmap()
        uv_materials.append(mat2)

        fixed_shapes.append(pyredner.Shape(
            vertices=shapes[i].vertices.clone().detach(),
            indices=shapes[i].indices.clone().detach(),
            material_id=i,
            uvs = shapes[i].uvs.clone().detach() if shapes[i].uvs is not None else None,
            normals=shapes[i].normals.clone().detach() if shapes[i].normals is not None else None,
            uv_indices=shapes[i].uv_indices.clone().detach()  if shapes[i].uv_indices is not None else None
        ))
        fixed_shapes[-1].uvs, fixed_shapes[-1].uv_indices = pyredner.compute_uvs(fixed_shapes[-1].vertices, fixed_shapes[-1].indices, print_progress=False)
        fixed_shapes[-1].normals = pyredner.compute_vertex_normal(fixed_shapes[-1].vertices, fixed_shapes[-1].indices)
        fixed_shapes[i].normals = (-1.0 if inversed_normals(fixed_shapes[i]) else 1.0) * fixed_shapes[i].normals

    def txt_param_fuc(shapes, materials, is_longtail):
        for i in range(len(materials)):
            materials[i].diffuse_reflectance.texels.requires_grad = True
        return  [materials[i].diffuse_reflectance.texels for i in range(0, len(materials))]
    refiner = Optimizer(fixed_shapes, uv_materials, texture_targets, rnd_cameras, txt_param_fuc, lr=5e-2, loss="smooth_color", remeshing_interval=0, bounces=ARGS.bounces)
    start_time = time.time()
    texture_losses = {}
    for it in range(ARGS.texture_refinement_steps):
        print("Refinement {}".format(it))
        record = refiner.step(it, progress=float(it)/float(ARGS.texture_refinement_steps), update_normals=False, update_texture=True)
        for key in record.keys():
            texture_losses[key] = texture_losses.get(key, []) + [record[key]]
        if it % ARGS.saving_interval == 0:
            save_images(ARGS.steps_folder, "refinement_{:d}".format(it), refiner.scenes, bounces=ARGS.bounces)
    end_time = time.time()
    plot_values(texture_losses, "Texture refinement", "Iteration", "Loss", save=os.path.join(ARGS.output_folder, "texture_losses.png"))
    save_images(ARGS.output_folder, ARGS.output_name + "_texture", refiner.scenes, samples=512, bounces=ARGS.bounces)
    for i in range(len(refiner.shapes)):
        save_model(os.path.join(ARGS.output_folder, "mesh"), ARGS.output_name + "_{}_texture".format(i), refiner.shapes[i], refiner.materials[i], mesh_colors=False)

    status.close()
    print("Elapsed time: {:.2f} min".format((end_time - start_time)/60.0))