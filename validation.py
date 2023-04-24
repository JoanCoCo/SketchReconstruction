import os
import json
import torch
import pyredner
import argparse
import numpy as np
from PIL import Image
from utils.resources import load_model, load_shapes, load_samples_range
from chamferdist import ChamferDistance
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

if __name__ == "__main__":
    #### PARSE ARGUMENTS ####
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('--input', type=str, default="model.obj")
    parser.add_argument('--output', type=str, default="metrics")
    parser.add_argument('--target-images', type=str, default="validation")
    parser.add_argument('--cameras', type=str, default="camera_info.json")
    parser.add_argument('--target-model', type=str, default=None)
    parser.add_argument('--target-prefix', type=str, default="view_")
    parser.add_argument('--envmap', type=str, default="data/envmap/white_env.exr")
    parser.add_argument('--only-mesh', type=bool, default=False)
    ARGS = parser.parse_args()

    #### LOAD CAMERA INFO ####
    cameras_info = []
    val_set = []
    is_gonna_be_ortho = False
    with open(ARGS.cameras, 'r') as cameras_file:
        data = json.load(cameras_file)
        val_set = data["validation"]
        is_gonna_be_ortho = ("type" in data) and (data["type"] == "ORTHO")
        for i in val_set:
            cameras_info.append({
                "position": data["positions"][i-1],
                "up": data["ups"][i-1],
                "rotation": data["rotations"][i-1]
            })
    if len(cameras_info) == 0:
        print("No cameras where found. Verify the provided cameras description file.")
        exit()

    #### LOAD TARGETS ####
    targets = load_samples_range(ARGS.target_images, ARGS.target_prefix, val_set, copy_first=False)

    #### SET UP CAMERAS ####
    cameras = []
    for config in cameras_info:
        cam = pyredner.Camera(position=torch.tensor(config["position"], device=pyredner.get_device()), 
                                look_at=torch.tensor([0.0, 0.0, 0.0]), 
                                up=torch.tensor(config["up"], device=pyredner.get_device()), 
                                fov=torch.tensor([50.0]), 
                                clip_near=1e-2,
                                resolution=(targets[0].shape[0], targets[0].shape[1]),
                                camera_type=pyredner.camera_type.perspective if not is_gonna_be_ortho else pyredner.camera_type.orthographic,
                                fisheye=False)
        cameras.append(cam)
    
    #### LOAD SOURCE MODEL ####
    input_path = os.path.dirname(ARGS.input) 
    input_name = os.path.basename(ARGS.input).split(".")[0]
    shapes = load_shapes(input_path, input_name)
    materials = []

    if not os.path.exists(ARGS.output):
        os.makedirs(ARGS.output)

    if not ARGS.only_mesh:
        for i in range(len(shapes)):
            texture = np.array(Image.open(os.path.join(input_path,"Kd_texels.png")).convert("RGB"), dtype=np.float32) / 255.0
            texels = torch.tensor(texture, dtype=torch.float, device=pyredner.get_device())
            diffuse = pyredner.Texture(texels, mesh_colors_resolution=0)
            mat = pyredner.Material(diffuse_reflectance=diffuse, specular_reflectance=None, roughness=None, normal_map=None)
            mat.diffuse_reflectance.generate_mipmap()
            materials.append(mat)
            shapes[i].material_id = i

        #### LOAD ENVIRONMENT ####
        envmap = pyredner.imread(ARGS.envmap)
        if pyredner.get_use_gpu():
            envmap = envmap.cuda()
        envmap = pyredner.EnvironmentMap(envmap)

        #### CREATE SCENES ####
        scenes = []
        for camera in cameras:
            scenes.append(pyredner.Scene(camera, shapes, materials, area_lights=[], envmap=envmap))
        
        #### GENERATE THE RENDERS ####
        if not os.path.exists(os.path.join(ARGS.output, "renders")):
            os.makedirs(os.path.join(ARGS.output, "renders")) 
        
        render = pyredner.RenderFunction.apply
        renders = []
        for i, scene in enumerate(scenes):
            print("Rendering scene {:d}".format(val_set[i]))
            scene_args=pyredner.RenderFunction.serialize_scene(scene=scene, num_samples=512, max_bounces=1, channels=[pyredner.channels.radiance, pyredner.channels.alpha])
            buffer = render(0, *scene_args)
            img = torch.cat((buffer[:, :, :3], buffer[:, :, 3:]), dim=2)
            uimg = np.array(torch.clamp(img, max=1.0).cpu() * 255.0, dtype=np.uint8)
            Image.fromarray(uimg).save(os.path.join(ARGS.output, "renders", "render_{:d}.png".format(val_set[i])))
            renders.append(img)

        #### EVALUATE IMAGES ####
        with open(os.path.join(ARGS.output, "render_metrics.txt"), 'w') as metrics_file:
            metrics_file.write('ID, MSE(-), PSNR(+), SSIM(+), LPIPS(-)\n')
            mses = []
            psnrs = []
            ssims = []
            lpipss = []
            lpips_func = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(pyredner.get_device())
            for i in range(len(val_set)):
                pred = torch.clamp(renders[i], 0.0, 1.0)
                truth = torch.clamp(targets[i], 0.0, 1.0)
                mse = torch.nn.functional.mse_loss(pred, truth, size_average=None, reduce=None, reduction='mean').item()
                mses.append(mse)
                psnr = -10. / np.log(10.) * np.log(mse)
                psnrs.append(psnr)
                pred = torch.swapaxes(pred, 0, 2)
                truth = torch.swapaxes(truth, 0, 2)
                ssim = structural_similarity_index_measure(pred[None,...], truth[None,...], data_range=1.0).item()
                ssims.append(ssim)
                lpips = lpips_func(pred[None,0:3,...], truth[None,0:3,...]).item()
                lpipss.append(lpips)
                line = "{:d}, {:1.8f}, {:2.8f}, {:1.8f}, {:1.8f}\n".format(val_set[i], mse, psnr, ssim, lpips)
                metrics_file.write(str(line))
            
            avg_mse = np.mean(np.array(mses))
            avg_psnr = np.mean(np.array(psnrs))
            avg_ssim = np.mean(np.array(ssims))
            avg_lpips = np.mean(np.array(lpips))
            line = "AVERAGES: {:1.4f}, {:2.3f}, {:1.4f}, {:1.4f}\n".format(avg_mse, avg_psnr, avg_ssim, avg_lpips)
            metrics_file.write(str(line))
            print("MSE,      PSNR,   SSIM,   LPIPS")
            print("{:1.6f}, {:2.3f}, {:1.4f}, {:1.4f}".format(avg_mse, avg_psnr, avg_ssim, avg_lpips))
            metrics_file.close()
    
    #### LOAD REFERENCE MODEL ####
    _, mesh_list, _ = pyredner.load_obj(ARGS.target_model)
    assert (len(mesh_list) == len(shapes)) 
    reference_model = []
    for i in range(len(mesh_list)):
        reference_model.append(pyredner.Shape(
            vertices=mesh_list[i][1].vertices,
            indices=mesh_list[i][1].indices,
            material_id=0,
            uvs=mesh_list[i][1].uvs,
            normals=mesh_list[i][1].normals,
            uv_indices=mesh_list[i][1].uv_indices
        ))

    #### EVALUATE MODEL ####
    chamfer_metric = ChamferDistance()
    with open(os.path.join(ARGS.output, "model_metrics.txt"), 'w') as metrics_file:
        metrics_file.write("SHAPE, FORWARD(-), BACKWARD(-), SYMMETRIC(-), FACES1, FACES2\n")
        distances = []
        for i in range(len(shapes)):
            c1 = chamfer_metric(reference_model[i].vertices[None,...], shapes[i].vertices[None,...], reverse=False).item() / float(reference_model[i].vertices.shape[0])
            c2 = chamfer_metric(reference_model[i].vertices[None,...], shapes[i].vertices[None,...], reverse=True).item() / float(shapes[i].vertices.shape[0])
            chamfer = c1 + c2
            distances.append(chamfer)
            line = "{:d}, {:.8f}, {:.8f}, {:.8f}, {:d}, {:d}\n".format(i, c1, c2, chamfer, reference_model[i].vertices.shape[0], shapes[i].vertices.shape[0])
            metrics_file.write(str(line))
        avg_chamfer = np.mean(np.array(distances))
        line = "AVERAGE CHAMFER DISTACE: {:1.4f}\n".format(avg_chamfer)
        metrics_file.write(str(line))
        print("CHAMFER")
        print("{:1.4f}".format(avg_chamfer))
        metrics_file.close()
    
    #### EVALUATE MODEL SCALED ####
    with open(os.path.join(ARGS.output, "model_scaled_metrics.txt"), 'w') as metrics_file:
        metrics_file.write("SHAPE, FORWARD(-), BACKWARD(-), SYMMETRIC(-), FACES1, FACES2\n")
        distances = []
        for i in range(len(shapes)):
            reference_vertices = reference_model[i].vertices
            shape_vertices = shapes[i].vertices
            reference_bb = torch.tensor([
                torch.max(reference_vertices[:, 0]) - torch.min(reference_vertices[:, 0]),
                torch.max(reference_vertices[:, 1]) - torch.min(reference_vertices[:, 1]),
                torch.max(reference_vertices[:, 2]) - torch.min(reference_vertices[:, 2])
            ])
            shape_bb = torch.tensor([
                torch.max(shape_vertices[:, 0]) - torch.min(shape_vertices[:, 0]),
                torch.max(shape_vertices[:, 1]) - torch.min(shape_vertices[:, 1]),
                torch.max(shape_vertices[:, 2]) - torch.min(shape_vertices[:, 2])
            ])
            max_dim = torch.argmax(reference_bb)
            scale_factor = reference_bb[max_dim.item()] / shape_bb[max_dim.item()]
            shape_vertices = scale_factor * shape_vertices
            c1 = chamfer_metric(reference_vertices[None,...], shape_vertices[None,...], reverse=False).item() / float(reference_vertices.shape[0])
            c2 = chamfer_metric(reference_vertices[None,...], shape_vertices[None,...], reverse=True).item() / float(shape_vertices.shape[0])
            chamfer = c1 + c2
            distances.append(chamfer)
            line = "{:d}, {:.8f}, {:.8f}, {:.8f}, {:d}, {:d}\n".format(i, c1, c2, chamfer, reference_model[i].vertices.shape[0], shapes[i].vertices.shape[0])
            metrics_file.write(str(line))
        avg_chamfer = np.mean(np.array(distances))
        line = "AVERAGE CHAMFER DISTACE: {:1.4f}\n".format(avg_chamfer)
        metrics_file.write(str(line))
        print("CHAMFER SCALED")
        print("{:1.4f}".format(avg_chamfer))
        metrics_file.close()