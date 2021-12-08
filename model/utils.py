import torch
import torch.nn.functional as F


def construct_volume_lattice(bev_grid_size, grid_cell_width, vertical_sampling_range, n_samples):
    z = torch.linspace(1+ (bev_grid_size[0] - 0.5) * grid_cell_width, 1 + grid_cell_width/2, bev_grid_size[0])
    x = torch.linspace((bev_grid_size[1] + 1) * grid_cell_width / 2, (bev_grid_size[1] - 1) * grid_cell_width / 2,
                bev_grid_size[1])
    y = torch.linspace(vertical_sampling_range[0], vertical_sampling_range[1], n_samples)
    xx, yy, zz = torch.meshgrid(x, y, z)
    grid = torch.stack([xx, yy, zz], dim=-1)
    depths = grid[:, :, :, -1].unsqueeze(3).repeat(1, 1, 1, 3)
    norm_grid = torch.div(grid, depths)
    norm_grid.requires_grad = False
    return norm_grid

def construct_ray_attention_grid(bev_grid_size, n_samples):
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
    samples = torch.stack([samples_x, samples_y], dim = -1)

    grid = samples - torch.tensor([W/2, H/2]).float()
    grid = torch.div(grid, torch.tensor([W/2, H/2]).float())
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

def _test():
    grid = construct_neighbourhood_attention_mask((49,50), 17)
    print(grid.shape)

if __name__ == '__main__':
    _test()
