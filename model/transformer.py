import math
import copy

import torch
import torch.nn.functional as F
from torch import nn
from .utils import construct_ray_attention_grid, construct_neighbourhood_attention_mask

class TransformerConfig:

    def __init__(self, n_embed, n_head, num_layers, **kwargs):
        self.n_embed = n_embed
        self.n_head = n_head
        self.num_layers = num_layers
        for k,v in kwargs.items():
            setattr(self, k, v)

class SamplingTransformer(nn.Module):
    
    def __init__(self, bev_grid_size = (49, 50), grid_cell_width = 1, rgb_sampling_range = [-3,3], n_samples_rgb = 50, 
            n_samples_ray = 50, neighbourhood_size = 17, initialization_level = -0.5, 
            n_embed = 512, n_head = 8, num_layers = 6, dim_feedforward=2048, normalize_before = False, activation='gelu',
            embed_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1):
        super().__init__()
        self.config = TransformerConfig(n_embed=n_embed, n_head=n_head, num_layers=num_layers, bev_grid_size=bev_grid_size,
                            grid_cell_width=grid_cell_width, rgb_sampling_range=rgb_sampling_range, n_samples_rgb=n_samples_rgb,
                            n_samples_ray=n_samples_ray, neighbourhood_size=neighbourhood_size,initialization_level=initialization_level,
                            dim_feedforward=dim_feedforward, normalize_before=normalize_before, activation=activation,
                            embed_pdrop=embed_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop)
        self.layer = TransformerLayer(self.config)
        self.layers = _get_clones(self.layer, num_layers)
        self.norm = nn.LayerNorm(n_embed) if normalize_before else None 
    
        self._reset_parameters()
        #TODO add logging of parameters and positional encoding

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, f, cam):
        x = torch.rand((1, 49 * 50, 512), dtype = torch.float32)
        
        for layer in self.layers:
            x = layer(x, f, cam)

        if self.norm is not None:
            x = self.norm(x)
        
        return x


class TransformerLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ray_attn = RayAttentionModule(config)
        self.cross_attn = CrossAttentionModule(config)
        self.neighbourhood_attn = NeighbourhoodAttentionModule(config)
        self.mlp = nn.Sequential(
                nn.Linear(config.n_embed, config.dim_feedforward),
                _get_activation_function(config.activation),
                nn.Dropout(config.resid_pdrop),
                nn.Linear(config.dim_feedforward, config.n_embed),
                nn.Dropout(config.resid_pdrop))

        self.norm1 = nn.LayerNorm(config.n_embed)
        self.norm2 = nn.LayerNorm(config.n_embed)
        self.norm3 = nn.LayerNorm(config.n_embed)
        self.norm4 = nn.LayerNorm(config.n_embed)

        self.normalize_before = config.normalize_before
    
    def forward_post(self, x, f, cam):
        x2 = self.cross_attn(x, f, cam)
        x = self.norm1(x + x2)
        x2 = self.ray_attn(x)
        x = self.norm2(x + x2)
        x2 = self.neighbourhood_attn(x)
        x = self.norm3(x + x2)
        x = x + self.mlp(x)
        x = self.norm4(x)
        return x

    def forward_pre(self, x, f, cam):
        x = self.norm1(x)
        x2 = self.cross_attn(x, f, cam)
        x = self.norm2(x + x2)
        x2 = self.ray_attn(x)
        x = self.norm3(x + x2)
        x2 = self.neighbourhood_attn(x)
        x = self.norm4(x + x2)
        x = x + self.mlp(x)
        return x

    def forward(self, x, f, cam):
        if self.normalize_before:
            return self.forward_pre(x, f, cam)
        return self.forward_post(x, f, cam)

class CrossAttentionModule(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.n_head = config.n_head
        self.bev_grid_size = config.bev_grid_size
        self.grid_cell_width = config.grid_cell_width
        self.rgb_sampling_range = config.rgb_sampling_range
        self.n_samples_rgb = config.n_samples_rgb
        self.n_embed = config.n_embed
        
        self.grid = self.construct_cross_attention_grid()

        self.key = nn.Linear(config.n_embed, config.n_embed)
        self.query = nn.Linear(config.n_embed, config.n_embed)
        self.value = nn.Linear(config.n_embed, config.n_embed)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embed, config.n_embed)
    
    def forward(self, x, f, cam):
        B, T, C = x.shape

        proj_points = torch.einsum('ij, klmj ->klmi', cam, self.grid)
        proj_points = proj_points[:, :, :, :2].unsqueeze(0).repeat(B, 1, 1, 1, 1)
        H, W = f.shape[-2:]
        norm_points = proj_points - torch.tensor([W/2, H/2]).float()
        norm_points = torch.div(norm_points, torch.tensor([W/2, H/2]).float())
        B, X, Y, D, V = norm_points.shape
        points = norm_points.permute((0, 3, 1, 2, 4)).contiguous().view(B, D * X, Y, V)
        sampled_values = F.grid_sample(f, points, padding_mode='zeros')
        
        B, C, T, N = sampled_values.shape
        sampled_values = sampled_values.view(B, T, N, C).contiguous()
        
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(sampled_values).view(B, T, N, self.n_head, C // self.n_head).permute(0, 3, 1, 2, 4)
        v = self.value(sampled_values).view(B, T, N, self.n_head, C // self.n_head).permute(0, 3, 1, 2, 4)

        att = torch.einsum('bhij, bhikj->bhik', q, k) * (1.0 /math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = torch.einsum('bhij, bhijk-> bhik', att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

    def construct_cross_attention_grid(self):
        z = torch.linspace(1+ (self.bev_grid_size[0] - 0.5) * self.grid_cell_width, 1 + self.grid_cell_width/2, self.bev_grid_size[0])
        x = torch.linspace((-self.bev_grid_size[1] + 1) * self.grid_cell_width / 2, (self.bev_grid_size[1] - 1) * self.grid_cell_width / 2,
                self.bev_grid_size[1])
        y = torch.linspace(self.rgb_sampling_range[0], self.rgb_sampling_range[1], self.n_samples_rgb)
        xx, yy, zz = torch.meshgrid(x, y, z)
        grid = torch.stack([xx, yy, zz], dim=-1)
        depths = grid[:, :, :, -1].unsqueeze(3).repeat(1, 1, 1, 3)
        norm_grid = torch.div(grid, depths)
        norm_grid.requires_grad = False
        return norm_grid

class RayAttentionModule(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.n_head = config.n_head
        self.bev_grid_size = config.bev_grid_size
        self.n_samples_ray = config.n_samples_ray
        self.n_embed = config.n_embed
        
        self.grid = construct_ray_attention_grid()
        #self.grid = self.construct_self_attention_grid()

        self.key = nn.Linear(config.n_embed, config.n_embed)
        self.query = nn.Linear(config.n_embed, config.n_embed)
        self.value = nn.Linear(config.n_embed, config.n_embed)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embed, config.n_embed)

    def forward(self, x):
        B, T, C = x.shape

        H, W = self.bev_grid_size
        sample_inp = x.transpose(1, 2).reshape((B, C, H, W)).contiguous()
        points = self.grid.reshape((H * W, self.n_samples_ray, 2)).unsqueeze(0).repeat((B, 1, 1, 1)).contiguous()
        sampled_values = F.grid_sample(sample_inp, points, padding_mode='zeros')

        B, C, T, N = sampled_values.shape
        sampled_values = sampled_values.view(B, T, N, C).contiguous()

        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(sampled_values).view(B, T, N, self.n_head, C // self.n_head).permute(0, 3, 1, 2, 4)
        v = self.value(sampled_values).view(B, T, N, self.n_head, C // self.n_head).permute(0, 3, 1, 2, 4)

        att = torch.einsum('bhij, bhikj->bhik', q, k) * (1.0 /math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = torch.einsum('bhij, bhijk-> bhik', att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

    def construct_self_attention_grid(self):
        H, W = self.bev_grid_size

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

        samples_enum = torch.arange(self.n_samples_ray) / self.n_samples_ray

        samples_x = W / 2 + (intersections[:, :, 0].unsqueeze(-1) - W / 2) * samples_enum[None, None, :]
        samples_y = H - (H - intersections[:, :, 1].unsqueeze(-1)) * samples_enum[None, None, :]
        samples = torch.stack([samples_x, samples_y], dim = -1)

        grid = samples - torch.tensor([W/2, H/2]).float()
        grid = torch.div(grid, torch.tensor([W/2, H/2]).float())
        assert ((grid >= -1) & (grid <=1)).all()
        grid = grid.transpose(0, 1).contiguous()
        grid.requires_grad = False
        return grid

class NeighbourhoodAttentionModule(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.n_head = config.n_head
        self.bev_grid_size = config.bev_grid_size
        self.n_embed = config.n_embed
        self.neighbourhood_size = config.neighbourhood_size

        self.attention_mask = self.construct_attention_mask()

        self.self_attn = nn.MultiheadAttention(config.n_embed, config.n_head, dropout=config.attn_pdrop, batch_first=True)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        y = self.self_attn(x, x, x, attn_mask = self.attention_mask)[0]
        y = self.resid_drop(y)
        return y

    def construct_attention_mask(self):
        H, W = self.bev_grid_size
        
        mask = torch.zeros((H * W, H * W), dtype=torch.float32)
        mask.fill_diagonal_(1.)
        kernel = torch.ones((1, 1, self.neighbourhood_size, self.neighbourhood_size), dtype=torch.float32)

        mask = mask.unsqueeze(1).reshape((H*W, 1, H, W))
        pad = torch.nn.ZeroPad2d((self.neighbourhood_size-1) // 2)
        mask = pad(mask)
        mask = F.conv2d(mask, kernel)
        mask = mask.squeeze().reshape((H*W, H*W)).bool().contiguous()
        mask = ~mask
        mask.requires_grad = False

        return mask

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_function(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _test():
    transformer = SamplingTransformer()
    #module = SelfAttentionModule()
    x = torch.rand((1, 49 * 50, 512), dtype = torch.float32)
    f = torch.rand((1, 512, 75, 300), dtype = torch.float32)
    cam = torch.tensor([[175, 0, 150], [0, 175, 37.5], [0, 0, 1]], dtype = torch.float32)
    out = transformer(f, cam)
    print(out.shape)
    #print(grid.shape)

if __name__ == '__main__':
    _test()

