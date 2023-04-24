import os
import torch
import pyredner
import pymeshlab
from utils.generic import get_mesh_color_index

class Remesher:
    def __init__(self, working_folder="workshop", sample_radius=3):
        self.folder = working_folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.original_name = "source_mesh.obj"
        self.new_name = "remesh.obj"
        self.msc_samples = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [3, 0]], device=pyredner.get_device())
        self.sample_radius = sample_radius

    def close(self):
        if os.path.exists(self.folder):
            os.rmdir(self.folder)

    @torch.no_grad()
    def remesh(self, shape, material):
        # Save mesh so it an be loaded in meshlab.
        pyredner.save_obj(shape, os.path.join(self.folder, self.original_name), save_material=False)

        # Apply poisson remeshing.
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(os.path.join(self.folder, self.original_name))
        original_size = ms.current_mesh().face_number()
        ms.generate_surface_reconstruction_screened_poisson(preclean=True)
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=original_size)
        ms.save_current_mesh(os.path.join(self.folder, self.new_name))

        # Load the remeshed model.
        _, mesh_list, _ = pyredner.load_obj(os.path.join(self.folder, self.new_name))
        mesh = mesh_list[0][1]
        new_shape = pyredner.Shape(
                        vertices=mesh.vertices,
                        indices=mesh.indices,
                        material_id=0,
                        uvs = mesh.uvs,
                        normals=mesh.normals,
                        uv_indices=mesh.uv_indices
                    )
        new_material = pyredner.Material(
                            diffuse_reflectance=material.diffuse_reflectance,
                            specular_reflectance=material.specular_reflectance,
                            roughness=material.roughness, 
                            normal_map=material.normal_map
                        )
        new_material.diffuse_reflectance.texels = self.repair_texels(
                                                    shape.vertices,
                                                    shape.indices,
                                                    new_shape.vertices,
                                                    new_shape.indices,
                                                    material.diffuse_reflectance.texels
                                                )
        torch.cuda.empty_cache()
        return (new_shape, new_material)

    @torch.no_grad()
    def repair_texels(self, original_vertices, original_indices, new_vertices, new_indices, texels):
        # Generate the sampling points.
        new_samples = self.generate_samples(new_vertices, new_indices)
        # Compute the centers of the original triangles.
        original_centers = torch.sum(original_vertices[original_indices.long()], dim=1) / 3.0
        # Get the sampling candidates.
        candidates = self.closest_triangles(original_centers, new_samples)
        # Obtain the sampled colors.
        sampled_colors = self.sample_colors(original_vertices, original_indices, new_samples, candidates, texels)
        # Update the current texel.
        new_colors = texels.view(-1, 3).clone()
        samples_mc_coords = self.msc_samples.repeat(new_indices.shape[0], 1).view(new_indices.shape[0], -1, 2).view(-1, 2)
        samples_tris_ids = torch.tensor(range(0, new_indices.shape[0]), device=pyredner.get_device()).repeat(self.msc_samples.shape[0], 1).transpose(0,1).flatten()
        samples_color_coords = get_mesh_color_index(samples_tris_ids, 3, samples_mc_coords[:, 0], samples_mc_coords[:, 1]).long()
        new_colors[samples_color_coords] = sampled_colors
        return new_colors.flatten()

    @torch.no_grad()
    def generate_samples(self, new_vertices, new_indices):
        barycentric = torch.zeros(self.msc_samples.shape[0], 3).to(pyredner.get_device())
        barycentric[:, 0] = self.msc_samples[:, 0] / 3.0
        barycentric[:, 1] = self.msc_samples[:, 1] / 3.0
        barycentric[:, 2] = 1.0 - ((self.msc_samples[:, 0] + self.msc_samples[:, 1]) / 3.0)
        barycentric = barycentric.repeat(new_indices.shape[0], 1).view(new_indices.shape[0], -1, 3)
        tris_coords = new_vertices[new_indices.long()]
        tris_coords = tris_coords.repeat(1, self.msc_samples.shape[0], 1).view(-1, self.msc_samples.shape[0], 3, 3)
        barycentric = barycentric.repeat(1, 1, 3).view(-1, self.msc_samples.shape[0], 3, 3).transpose(2,3)
        new_samples = torch.sum(tris_coords * barycentric, dim=2).view(-1, 3)
        return new_samples
    
    @torch.no_grad()
    def closest_triangles(self, original_centers, new_samples):
        vco = new_samples.shape[0]
        original_centers_mtx = original_centers.repeat(vco, 1).view(vco, original_centers.shape[0], 3)
        vcn = original_centers.shape[0]
        new_points = new_samples.repeat(vcn, 1).view(vcn, new_samples.shape[0], 3).transpose(0, 1)
        distances = torch.norm(original_centers_mtx - new_points, p=2, dim=-1)
        candidates = torch.argsort(distances, dim=-1)[:, 0:self.sample_radius]
        assert candidates.shape[0] == new_samples.shape[0]
        return candidates

    @torch.no_grad()
    def sample_colors(self, original_vertices, original_indices, new_samples, candidates, texels):
        # Project the samples onto the candidates by using https://vccimaging.org/Publications/Heidrich2005CBP/Heidrich2005CBP.pdf
        candidate_triangles = original_vertices[original_indices.long()][candidates.long()]
        ct_v0 = candidate_triangles[:, :, 0]
        ct_v1 = candidate_triangles[:, :, 1]
        ct_v2 = candidate_triangles[:, :, 2]
        ct_u = ct_v1 - ct_v0
        ct_v = ct_v2 - ct_v0
        ct_n = torch.cross(ct_u, ct_v, dim=-1)
        ct_denom = 1.0 / torch.einsum("nsd,nsd->ns", ct_n, ct_n)
        ct_w = new_samples.repeat(1, self.sample_radius).view(-1, self.sample_radius, 3) - ct_v0
        projected_bars = torch.zeros(new_samples.shape[0], self.sample_radius, 3, device=pyredner.get_device())
        projected_bars[:, :, 2] = torch.clamp(torch.einsum("nsd,nsd->ns", torch.cross(ct_u, ct_w, dim=-1), ct_n) * ct_denom, 0.0, 1.0)
        projected_bars[:, :, 1] = torch.clamp(torch.einsum("nsd,nsd->ns", torch.cross(ct_w, ct_v, dim=-1), ct_n) * ct_denom, 0.0, 1.0)
        projected_bars[:, :, 0] = 1.0 - projected_bars[:, :, 1] - projected_bars[:, :, 2]

        # Obtain the mesh colors coordinates from the barycentric coordinates.
        projected_mc_coords = torch.zeros(projected_bars.shape[0], projected_bars.shape[1], 2, device=pyredner.get_device())
        projected_mc_coords[:, :, 0] = torch.floor(projected_bars[:, :, 0] * 3.0).long()
        projected_mc_coords[:, :, 1] = torch.floor(projected_bars[:, :, 1] * 3.0).long()

        # Obtain the indexes in the texels vector from the coordinates.
        projected_mc_inds = get_mesh_color_index(candidates, 3, projected_mc_coords[:, :, 0], projected_mc_coords[:, :, 1]).long()

        # Obtain the colors.
        sampled_colors = texels.view(-1, 3)[ projected_mc_inds ]
        sampled_colors = torch.mean(sampled_colors, dim=1)
        return sampled_colors