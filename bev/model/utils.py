import torch
import torch.nn.functional as F


def construct_volume_lattice(bev_grid_size, grid_cell_width, vertical_sampling_range, n_samples):
    z = torch.linspace(1+ (bev_grid_size[0] - 0.5) * grid_cell_width, 1 + grid_cell_width/2, bev_grid_size[0])
    x = torch.linspace((-bev_grid_size[1] + 1) * grid_cell_width / 2, (bev_grid_size[1] - 1) * grid_cell_width / 2,
                bev_grid_size[1])
    y = torch.linspace(vertical_sampling_range[0], vertical_sampling_range[1], n_samples)
    xx, yy, zz = torch.meshgrid(x, y, z)
    grid = torch.stack([xx, yy, zz], dim=-1)
    depths = grid[:, :, :, -1].unsqueeze(3).repeat(1, 1, 1, 3)
    norm_grid = torch.div(grid, depths)
    norm_grid.requires_grad = False
    return norm_grid

def construct_ray_attention_grid(bev_grid_size, n_samples):
    # TODO adapt s.t. it works with align_corners = True
    H, W = bev_grid_size

    xx, yy = torch.meshgrid([torch.arange(W), torch.arange(H)])
    xx = xx + 0.5
    yy = yy + 0.5

    orig_p = (W /2, H)
    m = (xx - orig_p[0]) / (orig_p[1] - yy)
    int_x = m * H + W / 2
    int_x[(int_x > W) | (int_x < 0)] = -1
    int_y = torch.zeros((W, H), dtype=torch.float32) - 1
    int_y[int_x == -1] = (H - (1 / m) * torch.sign(m) * (W / 2))[int_x == -1]
    intersections = torch.zeros((W, H, 2), dtype =torch.float32)
    zeros = torch.zeros((W, H), dtype = torch.float32)
    intersections[int_x != -1, :] = torch.stack([int_x[int_x != -1], zeros[int_x != -1]], dim = -1)
    cond_right = (int_y != -1) & (xx > W / 2)
    cond_left = (int_y != -1) & (xx < W / 2)
    intersections[cond_right] = torch.stack([zeros[cond_right] + W, int_y[cond_right]], dim =-1)
    intersections[cond_left] = torch.stack([zeros[cond_left], int_y[cond_left]], dim =-1)

    samples_enum = torch.arange(n_samples) / n_samples

    samples_x = W / 2 + (intersections[:, :, 0].unsqueeze(-1) - W / 2) * samples_enum[None, None, :]
    samples_y = H - (H - intersections[:, :, 1].unsqueeze(-1)) * samples_enum[None, None, :]
    samples = torch.stack([samples_y, samples_x], dim = -1)

    grid = samples - torch.tensor([W/2, H/2]).float()
    grid = torch.div(grid, torch.tensor([W/2, H/2]).float())
    assert ((grid >= -1) & (grid <=1)).all()
    grid = grid.transpose(0, 1).contiguous()
    grid.requires_grad = False
    return grid

def construct_ray_attention_grid_corners_aligned(bev_grid_size, n_samples):
    H, W = bev_grid_size

    xx, zz = torch.meshgrid([torch.arange(W), torch.arange(H)])

    orig_p = ((W - 1) /2, H - 1)
    m = (xx - orig_p[0]) / (orig_p[1] - zz)
    int_x = m * (H - 1) + (W - 1) / 2
    int_x[(int_x > (W - 1)) | (int_x < 0)] = -1
    int_z = torch.zeros((W, H), dtype=torch.float32) - 1
    int_z[int_x == -1] = ((H - 1) - (1 / m) * torch.sign(m) * ((W - 1) / 2))[int_x == -1]
    intersections = torch.zeros((W, H, 2), dtype =torch.float32)
    zeros = torch.zeros((W, H), dtype = torch.float32)
    intersections[int_x != -1, :] = torch.stack([int_x[int_x != -1], zeros[int_x != -1]], dim = -1)
    cond_right = (int_z != -1) & (xx > (W - 1) / 2)
    cond_left = (int_z != -1) & (xx < (W - 1) / 2)
    intersections[cond_right] = torch.stack([zeros[cond_right] + W - 1, int_z[cond_right]], dim =-1)
    intersections[cond_left] = torch.stack([zeros[cond_left], int_z[cond_left]], dim =-1)

    samples_enum = torch.arange(n_samples) / (n_samples - 1)

    samples_x = (W - 1) / 2 + (intersections[:, :, 0].unsqueeze(-1) - (W - 1) / 2) * samples_enum[None, None, :]
    samples_z = H - 1 - (H - 1 - intersections[:, :, 1].unsqueeze(-1)) * samples_enum[None, None, :]
    samples = torch.stack([samples_z, samples_x], dim = -1)

    if (W - 1) % 2 == 0:
        samples_x = torch.tensor([(W - 1) / 2]).repeat(H, n_samples)
        samples_z = ((H - 1) * samples_enum)[None, :].repeat(H, 1)
        samples_vertical_line = torch.stack([samples_z, samples_x], dim = -1)
        samples[int((W - 1) / 2)] = samples_vertical_line

    grid = samples - torch.tensor([(H - 1)/2, (W - 1)/2]).float()
    grid = torch.div(grid, torch.tensor([(H - 1)/2, (W - 1)/2]).float())
    assert ((grid >= -1) & (grid <=1)).all()
    grid = grid.transpose(0, 1).contiguous()
    grid.requires_grad = False
    return grid


def construct_neighbourhood_attention_mask(bev_grid_size, neighbourhood_size):
    H, W = bev_grid_size

    mask = torch.zeros((H * W, H * W), dtype=torch.float32)
    mask.fill_diagonal_(1.)
    kernel = torch.ones((1, 1, neighbourhood_size, neighbourhood_size), dtype=torch.float32)

    mask = mask.unsqueeze(1).reshape((H*W, 1, H, W))
    pad = torch.nn.ZeroPad2d((neighbourhood_size-1) // 2)
    mask = pad(mask)
    mask = F.conv2d(mask, kernel)
    mask = mask.squeeze().reshape((H*W, H*W)).bool().contiguous()
    mask = ~mask
    mask.requires_grad = False

    return mask

def construct_neighbourhood_attention_grid(bev_grid_size, neighbourhood_size):
    H, W = bev_grid_size
    assert neighbourhood_size % 2 == 1

    zz, xx = torch.meshgrid(torch.arange(H), torch.arange(W))
    len_pad = int((neighbourhood_size - 1) / 2)
    pad_value = -neighbourhood_size
    zz_padded = F.pad(zz, (len_pad, len_pad, len_pad, len_pad), mode='constant', value = pad_value)
    xx_padded = F.pad(xx, (len_pad, len_pad, len_pad, len_pad), mode='constant', value = pad_value)
    grid = torch.zeros((H, W, neighbourhood_size * neighbourhood_size, 2), dtype= torch.float32)
    for i in range(H):
        for j in range(W):
            grid[i, j, :, :] = torch.cat([zz_padded[i : i + neighbourhood_size, j: j+neighbourhood_size].flatten().reshape(-1, 1),
                            xx_padded[i : i + neighbourhood_size, j : j + neighbourhood_size].flatten().reshape(-1, 1)], dim =-1)
    mid_H, mid_W = (H - 1) / 2, (W - 1) / 2
    grid = grid - torch.tensor([mid_H, mid_W])
    grid = grid / torch.tensor([mid_H, mid_W])
    grid = grid.reshape(H * W, neighbourhood_size * neighbourhood_size, 2)
    grid.requires_grad = False
    return grid

def construct_density_attention_mask(bev_grid_size, density_grid_size):

    H_bev, W_bev = bev_grid_size
    H_den, W_den = density_grid_size
    assert H_den % H_bev == 0
    assert W_den % W_bev == 0
    h_fac = int(H_den / H_bev)
    w_fac = int(W_den / W_bev)
    
    mask = torch.zeros((H_bev, W_bev, H_den, W_den), dtype = torch.bool)
    for i in range(H_bev):
        for j in range(W_bev):
            mask[i, j, i * h_fac : (i + 1) * h_fac, j * w_fac: (j+1) * w_fac] = True
    mask = mask.reshape(H_bev * W_bev, H_den * W_den).contiguous()
    mask = ~mask
    mask.requires_grad = False

    return mask

def construct_density_attention_grid(bev_grid_size, density_grid_size):
    H_bev, W_bev = bev_grid_size
    H_den, W_den = density_grid_size
    assert H_den % H_bev == 0
    assert W_den % W_bev == 0
    h_fac = int(H_den / H_bev)
    w_fac = int(W_den / W_bev)

    zz, xx = torch.meshgrid(torch.arange(H_den), torch.arange(W_den))
    grid = torch.zeros((H_bev, W_bev, h_fac * w_fac, 2), dtype= torch.float32)
    for i in range(H_bev):
        for j in range(W_bev):
            grid[i, j, :, :] = torch.cat([zz[i * h_fac : (i + 1) * h_fac, j * w_fac: (j+1) * w_fac].flatten().reshape(-1, 1),
                            xx[i * h_fac : (i + 1) * h_fac, j * w_fac: (j+1) * w_fac].flatten().reshape(-1, 1)], dim =-1)
    mid_H, mid_W = (H_den - 1) / 2, (W_den - 1) / 2
    grid = grid - torch.tensor([mid_H, mid_W])
    grid = grid / torch.tensor([mid_H, mid_W])
    grid = grid.reshape(H_bev * W_bev, h_fac * w_fac, 2)
    grid.requires_grad = False

    return grid

def _test():
    #grid = construct_ray_attention_grid((49,50), 50)
    #print(grid.shape)
    grid = construct_volume_lattice((100, 100), 0.5, [-3, 3], 20)
    print(grid.shape)
    #grid = torch.rand((200, 200, 512))
    #masked = grid[ :, :, None, :][mask]
    #print(masked.shape)
    #x = torch.rand((1, 49 * 50, 512))

if __name__ == '__main__':
    _test()
