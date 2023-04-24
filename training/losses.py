import torch
import numpy as np
import pyredner
from utils.generic import get_mesh_color_index

#### Adapted from mtyke/laploss/laploss.py on GitHub

def gauss_kernel(size=5, sigma=1.0):
    grid = np.float32(np.mgrid[0:size,0:size].T)
    gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    return kernel

def conv_gauss(t_input, stride=1, k_size=5, sigma=1.6, repeats=1):
    t_kernel = torch.reshape(torch.tensor(gauss_kernel(size=k_size, sigma=sigma), dtype=torch.float32, device=pyredner.get_device()), [1, 1, k_size, k_size])
    t_kernel3 = torch.concat([t_kernel]*t_input.shape[1], axis=0)
    conv2d = torch.nn.Conv2d(t_input.shape[1], t_input.shape[1], k_size, stride=stride, bias=False, padding="same", groups=t_input.shape[1], device=pyredner.get_device())
    conv2d.weight.data = t_kernel3
    conv2d.weight.requires_grad=True
    t_result = t_input
    for _ in range(repeats):
        t_result = conv2d(t_result)
    return t_result

def get_gaussian_filter(size=5, sigma=1.6, channels=4, stride=1):
    t_kernel = torch.tensor(gauss_kernel(size=size, sigma=sigma), device=pyredner.get_device())
    t_kernel = t_kernel.view(1, 1, size, size)
    t_kernel = t_kernel.repeat(channels, 1, 1, 1).to(pyredner.get_device())
    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, groups=channels, kernel_size=size, bias=False, stride=stride, padding="same")
    gaussian_filter.weight.data = t_kernel
    gaussian_filter.weight.requires_grad = True
    return gaussian_filter 

def make_laplacian_pyramid(t_img, max_levels):
    t_pyr = []
    current = t_img
    avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
    for _ in range(max_levels):
        t_gauss = conv_gauss(current, stride=1, k_size=5, sigma=2.0)
        t_diff = current - t_gauss
        t_pyr.append(t_diff)
        current = avg_pool(t_gauss)
    t_pyr.append(current)
    return t_pyr

def laploss(t_img1, t_img2, max_levels=3):
    t_pyr1 = make_laplacian_pyramid(t_img1, max_levels)
    t_pyr2 = make_laplacian_pyramid(t_img2, max_levels)
    t_losses = torch.tensor([torch.norm(a-b,p=1)/torch.tensor(a.size(), dtype=torch.float32).prod() for a,b in zip(t_pyr1, t_pyr2)], device=pyredner.get_device())
    t_loss = t_losses.sum() * torch.tensor(t_img1.shape[0], dtype=torch.float32)
    return t_loss

##########################################

def color_loss(source, target, use_mask=True, max_levels=3):
    mask = torch.permute(torch.tile(target[:, :, 3], (3, 1, 1)), (1, 2, 0))
    m_source = source[..., 0:3] * mask
    m_target = target[..., 0:3] * mask
    m_source = torch.permute(m_source, (2, 0, 1))
    m_target = torch.permute(m_target, (2, 0, 1))
    m_source = m_source.view((1, m_source.shape[0], m_source.shape[1], m_source.shape[2]))
    m_target = m_target.view((1, m_target.shape[0], m_target.shape[1], m_target.shape[2]))

    avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
    gauss_filter = get_gaussian_filter(size=5, sigma=2.0, channels=m_source.shape[1])
    loss = torch.tensor(0.0, dtype=torch.float32, device=pyredner.get_device())
    for _ in range(max_levels):
        diff_s = m_source - gauss_filter(m_source)
        diff_t = m_target - gauss_filter(m_target)
        loss += torch.norm(diff_s - diff_t, p=1) / torch.tensor(m_source.size(), dtype=torch.float32).prod()
        m_source = avg_pool(m_source)
        m_target = avg_pool(m_target)
    loss += torch.norm(m_source - m_target, p=1) / torch.tensor(m_source.size(), dtype=torch.float32).prod()
    return loss

def silhouette_loss(source, target):
    assert (source.shape[2] == 4 and target.shape[2] == 4)
    return torch.nn.functional.mse_loss(source[..., 3], target[..., 3])

# From nvdiffrec/render/regularizer.py on GitHub
def laplace_regularizer_const(v_pos, t_pos_idx):
    term = torch.zeros_like(v_pos)
    norm = torch.zeros_like(v_pos[..., 0:1])

    v0 = v_pos[t_pos_idx[:, 0], :]
    v1 = v_pos[t_pos_idx[:, 1], :]
    v2 = v_pos[t_pos_idx[:, 2], :]

    term.scatter_add_(0, t_pos_idx[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, t_pos_idx[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, t_pos_idx[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, t_pos_idx[:, 0:1], two)
    norm.scatter_add_(0, t_pos_idx[:, 1:2], two)
    norm.scatter_add_(0, t_pos_idx[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.sum(term**2)

def spring_regularization(v_pos, t_pos_idx, k=0.01):
    v0 = v_pos[t_pos_idx[:, 0], :]
    v1 = v_pos[t_pos_idx[:, 1], :]
    v2 = v_pos[t_pos_idx[:, 2], :]
    
    d01 = torch.norm(v0 - v1, p=2, dim=-1)
    d12 = torch.norm(v1 - v2, p=2, dim=-1)
    d20 = torch.norm(v2 - v0, p=2, dim=-1)

    return k * torch.sum(d01 + d12 + d20)

# Mesh regularization and adaptive smoothing
def mean_curvature_flow_regularizer(v_pos, t_pos_idx):
    # result = torch.zeros_like(v_pos)
    term = torch.zeros_like(v_pos)

    v0 = v_pos[t_pos_idx[:, 0], :]
    v1 = v_pos[t_pos_idx[:, 1], :]
    v2 = v_pos[t_pos_idx[:, 2], :]

    t_areas = torch.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), p=2, dim=-1)/2.0
    v_area = torch.zeros_like(v_pos[:,0])
    v_area.index_add_(0, t_pos_idx.flatten(), t_areas.repeat(3, 1).transpose(0,1).flatten())

    diff01 = (v1 - v0) / torch.norm(v1 - v0, p=2, dim=-1, keepdim=True)
    diff02 = (v2 - v0) / torch.norm(v2 - v0, p=2, dim=-1, keepdim=True)
    cos0 = torch.einsum('nd,nd->n', diff01, diff02)

    diff10 = (v0 - v1) / torch.norm(v0 - v1, p=2, dim=-1, keepdim=True)
    diff12 = (v2 - v1) / torch.norm(v2 - v1, p=2, dim=-1, keepdim=True)
    cos1 = torch.einsum('nd,nd->n', diff10, diff12)

    diff20 = (v0 - v2) / torch.norm(v0 - v2, p=2, dim=-1, keepdim=True)
    diff21 = (v1 - v2) / torch.norm(v1 - v2, p=2, dim=-1, keepdim=True)
    cos2 = torch.einsum('nd,nd->n', diff20, diff21)

    ang0 = torch.acos(cos0)
    cot0 = cos0 / torch.sin(ang0)
    cot0 = cot0.repeat(3,1).transpose(0,1)

    ang1 = torch.acos(cos1)
    cot1 = cos1 / torch.sin(ang1)
    cot1 = cot1.repeat(3,1).transpose(0,1)

    ang2 = torch.acos(cos2)
    cot2 = cos2 / torch.sin(ang2)
    cot2 = cot2.repeat(3,1).transpose(0,1)

    # alpha * (Q-P) + beta * (Q-P)
    term.scatter_add_(0, t_pos_idx[:, 0:1].repeat(1,3), (v1 - v0) * cot2 + (v2 - v0) * cot1)
    term.scatter_add_(0, t_pos_idx[:, 1:2].repeat(1,3), (v0 - v1) * cot2 + (v2 - v1) * cot0)
    term.scatter_add_(0, t_pos_idx[:, 2:3].repeat(1,3), (v0 - v2) * cot1 + (v1 - v2) * cot0)

    # result = (term / (4.0 * v_area.repeat(3,1).transpose(0,1)))

    return torch.sum(term**2)

def color_smoothness(mesh_colors, t_pos_idx, resolution, samples=5):
    result = torch.zeros_like(t_pos_idx)
    samples_idx = torch.randint(0, resolution, (result.shape[0], samples, 2))
    to_clip = samples_idx[:, :, 1] > resolution - samples_idx[:, :, 0]
    samples_idx[ to_clip ][1] =  resolution - samples_idx[to_clip][0]
    tri_ids = torch.tile(torch.tensor(range(0, t_pos_idx.shape[0])), (samples, 1)).transpose(0, 1)
    mshc_idx = get_mesh_color_index(tri_ids, resolution, samples_idx[:, :, 0], samples_idx[:, :, 1]).long()
    colors = torch.clone(mesh_colors).detach().reshape(-1, 3)
    result = result + colors[mshc_idx[:, 0]]
    result = torch.sum(torch.abs(torch.tile(result, (1, samples-1)).reshape(-1, samples-1, 3) - colors[mshc_idx[:, 1:]]), dim=1) / (samples-1)
    return torch.sum(result**2)

# Build based on the texture smoothness regularization from nvdiffrec on GitHub
def texture_smoothness(texture, uvs):
    jitter = torch.normal(mean=0, std=0.005, size=uvs.size(), device=pyredner.get_device())

    s_uvs = uvs * texture.uv_scale
    s_uvs_jit = (uvs + jitter) * texture.uv_scale

    x = s_uvs[:, 0] * texture.texels.shape[1] - 0.5
    y = s_uvs[:, 1] * texture.texels.shape[0] - 0.5
    x_jit = s_uvs_jit[:, 0] * texture.texels.shape[1] - 0.5
    y_jit = s_uvs_jit[:, 1] * texture.texels.shape[0] - 0.5

    xf = torch.floor(x).long()
    yf = torch.floor(y).long()
    xf_jit = torch.floor(x_jit).long()
    yf_jit = torch.floor(y_jit).long()

    xc = xf + 1
    yc = yf + 1
    xc_jit = xf_jit + 1
    yc_jit = yf_jit + 1

    u = x - xf
    v = y - yf
    u_jit = x_jit - xf_jit
    v_jit = y_jit - yf_jit

    xfi = xf % texture.texels.shape[1]
    xfi[xfi < 0] += texture.texels.shape[1]
    xfi_jit = xf_jit % texture.texels.shape[1]
    xfi_jit[xfi_jit < 0] += texture.texels.shape[1]

    yfi = yf % texture.texels.shape[0]
    yfi[yfi < 0] += texture.texels.shape[0]
    yfi_jit = yf_jit % texture.texels.shape[0]
    yfi_jit[yfi_jit < 0] += texture.texels.shape[0]

    xci = xc % texture.texels.shape[1]
    xci[xci < 0] += texture.texels.shape[1]
    xci_jit = xc_jit % texture.texels.shape[1]
    xci_jit[xci_jit < 0] += texture.texels.shape[1]

    yci = yc % texture.texels.shape[0]
    yci[yci < 0] += texture.texels.shape[0]
    yci_jit = yc_jit % texture.texels.shape[0]
    yci_jit[yci_jit < 0] += texture.texels.shape[0]

    colors = torch.zeros(uvs.shape[0], texture.texels.shape[2], device=pyredner.get_device())
    colors_jit = torch.zeros(uvs.shape[0], texture.texels.shape[2], device=pyredner.get_device())
    for i in range(texture.texels.shape[2]):
        value_ff = texture.texels[yfi, xfi, i]
        value_cf = texture.texels[yfi, xci, i]
        value_fc = texture.texels[yci, xfi, i]
        value_cc = texture.texels[yci, xci, i]
        colors[:, i] = value_ff*(1.0-u)*(1.0-v) + value_fc*(1.0-u)*v + value_cf*u*(1.0-v) + value_cc*u*v

        value_ff_jit = texture.texels[yfi_jit, xfi_jit, i]
        value_cf_jit = texture.texels[yfi_jit, xci_jit, i]
        value_fc_jit = texture.texels[yci_jit, xfi_jit, i]
        value_cc_jit = texture.texels[yci_jit, xci_jit, i]
        colors_jit[:, i] = value_ff_jit*(1.0-u_jit)*(1.0-v_jit) + value_fc_jit*(1.0-u_jit)*v_jit + value_cf_jit*u_jit*(1.0-v_jit) + value_cc_jit*u_jit*v_jit

    return (colors - colors_jit).pow(2).sum()