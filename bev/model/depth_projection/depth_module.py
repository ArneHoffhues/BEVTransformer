import torch
import torch.nn as nn
import time

class GridAssembler(nn.Module):
    
    _EPSILON = 1e-10

    def __init__(self,
                grid_dim,
                num_pix,
                with_density = False,
                density_resolution = 0.25,
                max_num_pix_per_cell = 2000,
                num_dim = 512, 
                max_depth = 80.,
                z_range = [51, 1],
                x_range = [-25, 25],
                ):
        super().__init__()
        assert len(grid_dim) == 2 
        self.grid_dim = grid_dim
        self.num_pix = num_pix
        self.max_depth = max_depth
        self.with_density = with_density
        self.density_resolution = density_resolution
        self.max_num_pix_per_cell = max_num_pix_per_cell

        self.z_range = z_range
        self.x_range = x_range

        if num_dim != 512:
            self.dim_reduce = nn.Conv2d(512, num_dim, kernel_size=1, stride=1)
        else:
            self.dim_reduce = None
    
    def _permute_sort(self, x, permutation):
        d1, d2 = x.shape
        ret = x[torch.arange(d1).unsqueeze(1).repeat((1,d2)).flatten(), permutation.flatten()].view(d1,d2)
        return ret
    
    def _density_grid(self, x_coord, z_coord, grid_res):
        B, _  = x_coord.shape
        grid_dim = (int((self.z_range[0] - self.z_range[1]) / grid_res), int((self.x_range[1] - self.x_range[0]) / grid_res))
        density_grid = torch.zeros((B, (grid_dim[0] * grid_dim[1])), device = z_coord.device)
        for i in range(B):
            z = z_coord[i].detach().clone() / grid_res
            z = torch.floor(z)
            z = self.z_range[0] / grid_res - 1 - z
            keep_ind_z  = (z < grid_dim[0]) & (z >= 0)
            z = z[keep_ind_z]
            #z_coord[(z_coord >= grid_dim[0]) | (z_coord < 0)] = -1

            x = x_coord[i].detach().clone() / grid_res
            x = x[keep_ind_z]
            x = torch.floor(x)
            x = x + abs(self.x_range[0] / grid_res)
            keep_ind_x = (x < grid_dim[1]) & (x >= 0)
            x = x[keep_ind_x].long()
            z = z[keep_ind_x].long()
            #x_coord[(x_coord >= grid_dim[1]) | (x_coord < 0)] = -1
        
            #density_grid = torch.zeros(B, ((grid_dim[0] + 1) * grid_dim[1]), device = z_coord.device)
            grid_numbered = torch.arange(grid_dim[0] * grid_dim[1], device=z_coord.device).reshape(grid_dim)
            grid_cells = grid_numbered[z, x]
            cell_counts = grid_cells.unique(return_counts=True)
            #cell_counts = torch.bincount(grid_cells)
            density_grid[i, cell_counts[0]] = cell_counts[1].float()
            density_grid[i] = torch.clamp(density_grid[i], 0., self.max_num_pix_per_cell)
            density_grid[i] /= (self.max_num_pix_per_cell + GridAssembler._EPSILON)

        density_grid.requires_grad = True
        density_grid = density_grid.reshape(B, grid_dim[0], grid_dim[1]).unsqueeze(1).contiguous()
        
        return density_grid


    def forward(self,seg, depth, cam):
        if self.dim_reduce is not None:
            seg = self.dim_reduce(seg)
        B, C, H, W = seg.shape
        seg_flat = seg.flatten(-2, -1)
        depth = depth * self.max_depth
        assert depth.shape[-1] * depth.shape[-2] == self.num_pix

        fx, fy, cx, cy = cam[:,0,0], cam[:,1,1], cam[:,0,2], cam[:,1,2]

        rows, cols = depth.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(rows, device=seg.device), torch.arange(cols, device=seg.device))
        yy = yy[None, :, :].expand(B, H, W)
        xx = xx[None, :, :].expand(B, H, W)
        x = (xx - cx[:, None, None]) / fx[:, None, None]
        y = (yy - cy[:, None, None]) / fy[:, None, None]
        z = depth.squeeze(1)
        x = x * z
        y = y * z
        coord = torch.stack((x,y, z), dim=-1)
        
        x_coord = coord[:, :, :, 0].flatten(1, -1)
        #x_argsort = x_coord.argsort(dim=1)
        #x_sorted = self.permute_sort(x_coord, x_argsort)
        #x_ind_lower = ((x_sorted >= self.x_range[0]) == 0).sum(dim=-1)
        #x_ind_upper = ((x_sorted >= self.x_range[1]) == 0).sum(dim=-1)
        
        #y_coord = coord[:, :, :, 1].flatten(1, -1)
        #y_argsort = y_coord.argsort(dim=1)
        #y_sorted = self.permute_sort(y_coord, y_argsort)
        #y_ind_lower = ((y_sorted >= self.y_range[0]) == 0).sum(dim=-1)
        #y_ind_upper = ((y_sorted >= self.y_range[1]) == 0).sum(dim=-1)

        #y_coord = coord[:, :, :, :, 1].flatten(1, -1)
        #y_coord_ext = y_coord.expand(self.grid_dim[2], -1)
        z_coord = coord[:, :, :, 2].flatten(1, -1)
        
        if self.with_density:
            density = self._density_grid(x_coord, z_coord, self.density_resolution)
        else:
            density = None

        x_coord_ext = x_coord[:, None, :].expand(B, self.grid_dim[1], -1)
        z_coord_ext = z_coord[:, None, :].expand(B, self.grid_dim[0], -1)

        xs = torch.linspace(self.x_range[0], self.x_range[1], self.grid_dim[1] +1, device = x_coord_ext.device, requires_grad = False)
        xs_lower = xs[:-1]
        xs_upper = xs[1:]
        xs_lower = xs_lower[None, :, None].expand(B, -1, self.num_pix)
        xs_upper = xs_upper[None, :, None].expand(B, -1, self.num_pix)

        zs = torch.linspace(self.z_range[0], self.z_range[1], self.grid_dim[0] +1, device = z_coord_ext.device, requires_grad = False)
        zs_upper = zs[:-1]
        zs_lower = zs[1:]
        zs_lower = zs_lower[None, :, None].expand(B, -1, self.num_pix)
        zs_upper = zs_upper[None, :, None].expand(B, -1, self.num_pix)

        #ys = torch.linspace(self.y_range[0], self.y_range[1], self.grid_dim[2] +1, device = y_coord_ext.device)
        #ys_lower = ys[:-1]
        #ys_upper = ys[1:]
        #ys_lower = ys_lower[:, None].expand(-1, self.num_pix)
        #ys_upper = ys_upper[:, None].expand(-1, self.num_pix)

        x_mask = torch.zeros((B, self.grid_dim[1], self.num_pix), device=x_coord_ext.device, requires_grad = False).bool()
        x_mask[((x_coord_ext >= xs_lower) & (x_coord_ext < xs_upper)).nonzero(as_tuple=True)] = True
        z_mask = torch.zeros((B, self.grid_dim[0], self.num_pix), device = z_coord_ext.device, requires_grad = False).bool()
        z_mask[((z_coord_ext >= zs_lower) & (z_coord_ext < zs_upper)).nonzero(as_tuple=True)] = True
        #y_mask = torch.zeros((self.grid_dim[2], self.num_pix), device = y_coord_ext.device).bool()
        #y_mask[((y_coord_ext >= ys_lower) & (y_coord_ext < ys_upper)).nonzero(as_tuple=True)] = True
        
        mask =  (z_mask[:, :, None, :].expand(B, self.grid_dim[0], self.grid_dim[1], self.num_pix)) & \
                (x_mask[:, None, :, :].expand(B, self.grid_dim[0], self.grid_dim[1], self.num_pix))

        mask = mask.float() 
        
        numerator = torch.einsum('bzxi, bni -> bzxn', mask, seg_flat)
        ones = torch.ones((B, self.num_pix), device=seg_flat.device)
        denominator = torch.einsum('bzxi, bi -> bzx', mask, ones) + GridAssembler._EPSILON
        mean = numerator / denominator[:, :, :, None]
        #mean[torch.isnan(mean)] = 0.
        mean = mean.permute(0, 3, 1, 2).contiguous()

        #mean = torch.zeros((B, C, mask.shape[1], mask.shape[2]), dtype=torch.float, device=mask.device)
        #for z in range(mask.shape[1]):
        #    for x in range(mask.shape[2]):
        #        print(seg_flat[*mask[:, z, x, :], :])
        #        mean[:, :, z, x] = seg_flat[mask[:, z, x, :], :].mean(dim=1).squeeze()
        return mean, density


def _test():
    grid_dim = [50, 50]
    num_pix = 384 * 1408
    grid_assembler = GridAssembler(grid_dim, num_pix).to('cuda')
    seg = torch.rand(3, 512, 384, 1408).to('cuda')
    depth = (torch.rand(3, 1, 384, 1408) * 80).to('cuda')
    cam = (torch.Tensor([[552.5, 0, 682.], [0., 552.5, 238.7], [0., 0., 1.]])[None, :, :].repeat(1, 1, 1)).to('cuda')
    out = grid_assembler(seg, depth, cam)
    print(out.shape)

if __name__ == '__main__':
    _test()

