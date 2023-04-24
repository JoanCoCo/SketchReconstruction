import os
import json
import torch
import numpy as np
import pyredner
import argparse
import mcubes
from PIL import Image
import pymeshlab
import matplotlib.pyplot as plt

def load_image_as_volume(file, reduction):
    # Load image.
    img = np.array(Image.open(file).reduce(reduction).convert("RGBA"))[:, :, 3]
    # Determine the size of the volume.
    volume_side = np.max(img.shape)
    # First extend the depth.
    volume = np.tile(img, (volume_side, 1)).reshape((volume_side, img.shape[0], img.shape[1]))
    assert((volume[0] == volume[1]).all() and (volume[0] == img).all())
    # Then pad the remaining dimentions.
    d1_pad = (volume_side - volume.shape[1]) / 2.0
    d2_pad = (volume_side - volume.shape[2]) / 2.0
    volume = np.pad(volume, ((0, 0), (np.ceil(d1_pad).astype(int), np.floor(d1_pad).astype(int)), (np.ceil(d2_pad).astype(int), np.floor(d2_pad).astype(int))), mode='constant')
    # Arrange space to X - Y - Z
    volume = np.transpose(volume, (1, 2, 0))
    return volume

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build_initial_mesh")
    parser.add_argument('--cameras', type=str, default="cameras.json")
    parser.add_argument('--output', type=str, default="result.obj")
    parser.add_argument('--reduction', type=int, default=15)
    ARGS = parser.parse_args()

    # Load the views and images.
    views = []
    data = json.load(open(ARGS.cameras, 'r'))
    for i in range(len(data["positions"])):
        print("Loading view {}".format(i))
        rot = data["rotations"][i]
        img = os.path.join(os.path.dirname(ARGS.cameras), "view_{:d}.png".format(i+1))
        volume = load_image_as_volume(img, ARGS.reduction)
        view = np.array(pyredner.gen_rotate_matrix(torch.tensor(rot)).cpu(), dtype=np.float32)
        views.append((volume, view))

    # Compute the voxelated representation.
    total_volume = np.ones_like(views[0][0])
    voxels = {}
    voxel_id = 0
    for volume, matrix in views:
        d0 = np.linspace(0, volume.shape[0]-1, volume.shape[0], dtype=int)
        d1 = np.linspace(0, volume.shape[1]-1, volume.shape[1], dtype=int)
        d2 = np.linspace(0, volume.shape[2]-1, volume.shape[2], dtype=int)
        d0v, d1v, d2v = np.meshgrid(d0, d1, d2)
        indices = np.stack((d0v, d1v, d2v), axis=3)
        center = np.array([indices.shape[0] / 2.0, indices.shape[1] / 2.0, indices.shape[2] / 2.0])
        indices = indices[volume > 0.0].reshape(-1, 3)

        rotated_inds = (np.dot(indices - center, matrix) + center).astype(int)

        dim0_check = np.logical_and(rotated_inds[:, 0] >= 0, rotated_inds[:, 0] < volume.shape[0])
        dim1_check = np.logical_and(rotated_inds[:, 1] >= 0, rotated_inds[:, 1] < volume.shape[1])
        dim2_check = np.logical_and(rotated_inds[:, 2] >= 0, rotated_inds[:, 2] < volume.shape[2])
        indices = indices[np.logical_and(np.logical_and(dim0_check, dim1_check), dim2_check)]
        rotated_inds = rotated_inds[np.logical_and(np.logical_and(dim0_check, dim1_check), dim2_check)]

        """ Uncomment to see the process.
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(indices[:, 0], indices[:, 2], indices[:, 1], marker='o')
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        plt.title("Samples")
        plt.show()
        
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(rotated_inds[:, 0], rotated_inds[:, 2], rotated_inds[:, 1], marker='o')
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        plt.title("Samples")
        plt.show()
        """

        voxels["{:d}_voxels".format(voxel_id)] = rotated_inds.tolist()
        voxels["{:d}_center".format(voxel_id)] = center.tolist()
        voxel_id += 1

        rotated_volume = np.zeros_like(volume)
        rotated_volume[rotated_inds[:, 0], rotated_inds[:, 1], rotated_inds[:, 2]] = 1.0 #volume[indices[:, 0], indices[:, 1], indices[:, 2]]
        total_volume = total_volume * rotated_volume

    d0 = np.linspace(0, total_volume.shape[0]-1, total_volume.shape[0], dtype=int)
    d1 = np.linspace(0, total_volume.shape[1]-1, total_volume.shape[1], dtype=int)
    d2 = np.linspace(0, total_volume.shape[2]-1, total_volume.shape[2], dtype=int)
    d0v, d1v, d2v = np.meshgrid(d0, d1, d2)
    indices = np.stack((d0v, d1v, d2v), axis=3)
    indices = np.transpose(indices, (1, 0, 2, 3))
    center = np.array([indices.shape[0] / 2.0, indices.shape[1] / 2.0, indices.shape[2] / 2.0])
    indices = indices[total_volume > 0.0].reshape(-1, 3)

    voxels["{:d}_voxels".format(voxel_id)] = indices.tolist()
    voxels["{:d}_center".format(voxel_id)] = center.tolist()
    with open(os.path.join(os.path.dirname(ARGS.output), "voxels.json"), 'w') as file:
        json.dump(voxels, file)
        file.close()
    
    """ Uncomment to see the final result.
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(indices[:, 0], indices[:, 2], indices[:, 1], marker='o')
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    plt.title("Samples")
    plt.show()
    """

    # Center the mesh.
    center = np.array([total_volume.shape[0] / 2.0, -total_volume.shape[1] / 2.0, total_volume.shape[2] / 2.0])
    vertices, triangles = mcubes.marching_cubes(total_volume, 0)
    vertices[:, 1] *= -1.0
    vertices -= center
    rot_90 = pyredner.gen_rotate_matrix(torch.tensor([-np.pi/2.0, 0.0, 0.0]))
    for i in range(vertices.shape[0]):
        vertices[i] = rot_90 @ vertices[i]

    # Clean single-edge triangles.
    vertices, inds_map = np.unique(vertices, axis=0, return_inverse=True)
    triangles = inds_map[triangles]
    triangles = triangles[np.logical_and(np.logical_and(triangles[:, 0] != triangles[:, 1], triangles[:, 0] != triangles[:, 2]), triangles[:, 1] != triangles[:, 2])]

    # Scale the model.
    img = np.array(Image.open(os.path.join(os.path.dirname(ARGS.cameras), "view_1.png")).reduce(ARGS.reduction).convert("RGBA"))[:, :, 3]
    img = (img > 0.9).astype(np.float32)
    rot = data["rotations"][0]
    rotation = torch.tensor(np.array(pyredner.gen_rotate_matrix(torch.tensor(rot)).cpu(), dtype=np.float32), device=pyredner.get_device())
        
    ys = (np.sum(img, axis=-1) > 0.0)
    xs = (np.sum(img, axis=0) > 0.0)
    yr = np.array(range(img.shape[0]))
    xr = np.array(range(img.shape[1]))
    yr = yr[ys]
    xr = xr[xs]

    img_bb = ((np.min(yr) / img.shape[0], np.min(xr) / img.shape[1]), (np.max(yr) / img.shape[0], np.max(xr) / img.shape[1]))
    model_bb = ((np.min(vertices[:, 0]), np.min(vertices[:, 1]), np.min(vertices[:, 2])), (np.max(vertices[:, 0]), np.max(vertices[:, 1]), np.max(vertices[:, 2])))
    
    cam_pos = torch.transpose(rotation, 0, 1) @ torch.tensor([0.0, 0.0, 1.0], device=pyredner.get_device()).reshape(3, 1)
    up_vec = torch.transpose(rotation, 0, 1) @ torch.tensor([0.0, 1.0, 1.0], device=pyredner.get_device()).reshape(3, 1)
    cam_pos = torch.reshape(cam_pos, (3,))
    up_vec = torch.reshape(up_vec, (3,)) - cam_pos
    up_vec = up_vec / torch.norm(up_vec, p=2)
    cam_pos = 2.0 * (cam_pos / torch.norm(cam_pos, p=2))

    look = torch.tensor([0.0, 0.0, 0.0], device=pyredner.get_device())
    look_pos = look - cam_pos
    d = look_pos / torch.norm(look_pos, p=2)
    cross_d_up = torch.cross(d, up_vec)
    right = cross_d_up / torch.norm(cross_d_up, p=2)
    cross_right_d = torch.cross(right, d)
    new_up = cross_right_d / torch.norm(cross_right_d, p=2)
    cam_to_world = torch.tensor([
        [right[0], new_up[0],  d[0], cam_pos[0]],
        [right[1], new_up[1],  d[1], cam_pos[1]],
        [right[2], new_up[2],  d[2], cam_pos[2]],
        [      0,          0,     0,          1]
    ], dtype=torch.float32, device=pyredner.get_device())

    aspect_ratio = img.shape[1] / img.shape[0]
    pt_min = torch.tensor([(img_bb[0][1] - 0.5) * 2.0, (img_bb[0][0] - 0.5) * (-2.0) / aspect_ratio, 0.0, 1.0], dtype=torch.float32, device=pyredner.get_device())
    pt_max = torch.tensor([(img_bb[1][1] - 0.5) * 2.0, (img_bb[1][0] - 0.5) * (-2.0) / aspect_ratio, 0.0, 1.0], dtype=torch.float32, device=pyredner.get_device())

    org_min = np.array((cam_to_world @ pt_min).cpu())
    org_max = np.array((cam_to_world @ pt_max).cpu())
    inv_w_min = 1.0 / org_min[3]
    inv_w_max = 1.0 / org_max[3]
    org_min = org_min[0:3] / inv_w_min
    org_max = org_max[0:3] / inv_w_max

    min_point = np.array([model_bb[0][0], model_bb[0][1], model_bb[0][2]])
    max_point = np.array([model_bb[1][0], model_bb[1][1], model_bb[1][2]])

    smp = np.ones(img.shape)
    smp[int(img_bb[0][0] * img.shape[0]), int(img_bb[0][1] * img.shape[1])] = 0.0
    smp[int(img_bb[1][0] * img.shape[0]), int(img_bb[1][1] * img.shape[1])] = 0.0

    scale_factor = np.linalg.norm(org_max[0] - org_min[0]) / np.linalg.norm(max_point[0] - min_point[0])
    print("Scaling model by {:.4f}".format(scale_factor))
    vertices *= scale_factor 

    vertices = np.dot(vertices, np.array(pyredner.gen_rotate_matrix(torch.tensor([-1.5707963705062866, 0.0, 0.0]))))
    
    # Apply Poisson remeshing and save.
    if not os.path.exists(os.path.dirname(ARGS.output)):
        os.makedirs(os.path.dirname(ARGS.output))
    mcubes.export_obj(vertices, triangles, ARGS.output)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(ARGS.output)
    original_size = ms.current_mesh().face_number()
    ms.generate_surface_reconstruction_screened_poisson(preclean=True)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(original_size / 2))
    ms.save_current_mesh(ARGS.output)