import math
import copy

import torch
import torch.nn.functional as F
from torch import nn
from bev.model.utils import construct_ray_attention_grid, construct_neighbourhood_attention_mask, construct_density_attention_mask, \
        construct_density_attention_grid, construct_neighbourhood_attention_grid, construct_ray_attention_grid_corners_aligned
#from bev.utils import compute_mem_size
#from utils import construct_ray_attention_grid, construct_neighbourhood_attention_mask, construct_density_attention_mask, construct_density_attention_grid, \
#        construct_neighbourhood_attention_grid
#from resnet import ResNetLayer
from bev.model.resnet import ResNetLayer

class TransformerConfig:

    def __init__(self, n_embed, n_head, num_layers, **kwargs):
        self.n_embed = n_embed
        self.n_head = n_head
        self.num_layers = num_layers
        for k,v in kwargs.items():
            setattr(self, k, v)

class SamplingTransformer(nn.Module):
    
    def __init__(self, bev_grid_size = (49, 50), grid_cell_width = 1, vertical_sampling_range = [-3,3], n_samples_vertical = 50, 
            n_samples_ray = 50, neighbourhood_size = 17, initialization_level = -0.5, 
            n_embed = 512, n_head = 8, num_layers = 6, dim_feedforward=2048, normalize_before = False, activation='relu',
            embed_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1):
        super().__init__()
        self.config = TransformerConfig(n_embed=n_embed, n_head=n_head, num_layers=num_layers, bev_grid_size=bev_grid_size,
                            grid_cell_width=grid_cell_width, vertical_sampling_range=vertical_sampling_range, n_samples_vertical=n_samples_vertical,
                            n_samples_ray=n_samples_ray, neighbourhood_size=neighbourhood_size,initialization_level=initialization_level,
                            dim_feedforward=dim_feedforward, normalize_before=normalize_before, activation=activation,
                            embed_pdrop=embed_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop)

        self.layer = TransformerLayer(self.config)
        self.layers = _get_clones(self.layer, num_layers)
        self.norm = nn.LayerNorm(n_embed) if normalize_before else None 

        #TODO add positional encoding

    def forward(self, f, init, init_pos, height_pos):
        
        B, C, D, X, Y = f.shape
        assert (D, X) == self.config.bev_grid_size
        f = f.permute(0, 2, 3, 4, 1).contiguous().view(B, D * X, Y, C)
        if height_pos is not None:
            height_pos = height_pos.permute(0, 2, 3, 4, 1).view(B, D * X, Y, C)

        B, C, D, X = init.shape
        assert (D, X) == self.config.bev_grid_size
        init = init.permute(0, 2, 3, 1).contiguous().view(B, D * X, C)
        if init_pos is not None:
            init_pos = init_pos.permute(0, 2, 3, 1).contiguous().view(B, D * X, C)

        for layer in self.layers:
            x = layer(init, f, init_pos, height_pos)

        if self.norm is not None:
            x = self.norm(x)
        
        H, W = self.config.bev_grid_size
        x = x.view(B, H, W, self.config.n_embed).permute(0, 3, 1, 2).contiguous()
        return x

class TransformerLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        #self.register_buffer('neighbourhood_mask', construct_neighbourhood_attention_mask(config.bev_grid_size,
        #    config.neighbourhood_size))
        neighbourhood_attention_grid = construct_neighbourhood_attention_grid(config.bev_grid_size, config.neighbourhood_size).\
                view((config.bev_grid_size[0]*config.bev_grid_size[1], config.neighbourhood_size * config.neighbourhood_size, 2))
        ray_grid =construct_ray_attention_grid_corners_aligned(config.bev_grid_size, 
                config.n_samples_ray).view((config.bev_grid_size[0]*config.bev_grid_size[1], config.n_samples_ray, 2))

        self.ray_attn = SampleAttentionModule(config, sample_grid = ray_grid, samp_input_dim = config.bev_grid_size)
        self.cross_attn = SampleAttentionModule(config)
        #self.neighbourhood_attn = nn.MultiheadAttention(config.n_embed, config.n_head, dropout=config.attn_pdrop, batch_first=True)
        self.neighbourhood_attn = SampleAttentionModule(config, sample_grid = neighbourhood_attention_grid, samp_input_dim = config.bev_grid_size)
        self.mlp = nn.Sequential(
                nn.Linear(config.n_embed, config.dim_feedforward),
                _get_activation_function(config.activation),
                nn.Dropout(config.resid_pdrop),
                nn.Linear(config.dim_feedforward, config.n_embed),
                nn.Dropout(config.resid_pdrop))

        if config.normalize_before:
            self.norm1 = nn.LayerNorm(config.n_embed)
        else:
            self.norm1 = nn.Identity()
        self.norm2 = nn.LayerNorm(config.n_embed)
        self.norm3 = nn.LayerNorm(config.n_embed)
        self.norm4 = nn.LayerNorm(config.n_embed)
        if config.normalize_before:
            self.norm5 = nn.Identity()
        else:
            self.norm5 = nn.LayerNorm(config.n_embed)

        self.dropout1 = nn.Dropout(config.resid_pdrop)
        self.dropout2 = nn.Dropout(config.resid_pdrop)
        self.dropout3 = nn.Dropout(config.resid_pdrop)
    
    def with_pos_embed(self, x, pos):
        return x if pos is None else x + pos

    def forward(self, x, f, x_pos, f_pos):
        x = self.norm1(x)
        res = self.cross_attn(self.with_pos_embed(x, x_pos), self.with_pos_embed(f, f_pos), f)
        x = self.norm2(x + self.dropout1(res))
        res = self.ray_attn(self.with_pos_embed(x, x_pos), self.with_pos_embed(x, x_pos), x)
        x = self.norm3(x + self.dropout2(res))
        #res = self.neighbourhood_attn(x, x, x, attn_mask = self.neighbourhood_mask)[0]
        res = self.neighbourhood_attn(self.with_pos_embed(x, x_pos), self.with_pos_embed(x, x_pos), x)
        x = self.norm4(x + self.dropout3(res))
        x = self.norm5(x + self.mlp(x))
        return x

class SampleAttentionModule(nn.Module):

    def __init__(self, config, sample_grid=None, samp_input_dim=None, key_need_reshape=True, value_need_reshape=True):
        super().__init__()
        self.n_head = config.n_head
        self.key_need_reshape = key_need_reshape
        self.value_need_reshape = value_need_reshape

        if sample_grid is not None:
            self.register_buffer('grid', sample_grid)
        else:
            self.grid = None
        self.samp_input_dim = samp_input_dim

        self.key = nn.Linear(config.n_embed, config.n_embed)
        self.query = nn.Linear(config.n_embed, config.n_embed)
        self.value = nn.Linear(config.n_embed, config.n_embed)
        
        self.attn_drop = nn.Dropout(config.attn_pdrop)

    def forward(self, query, key, value):
        B, T, C = query.shape
        
        #S = key.shape[1]
        #assert key.shape == value.shape, f"Shape mismatch: {key.shape}, {value.shape}"
        #assert key.shape[0] == query.shape[0] and key.shape[-1] == query.shape[-1], f"Shape mismatch: {key.shape}, {value.shape}"

        if self.grid is not None:
            if self.key_need_reshape:
                assert self.samp_input_dim is not None
                H, W = self.samp_input_dim
                sample_inp = key.transpose(1, 2).contiguous().view((B, C, H, W))
            else:
                sample_inp = key
            points = self.grid.unsqueeze(0).repeat((B, 1, 1, 1))
            sampled_key = F.grid_sample(sample_inp, points, padding_mode='zeros', align_corners = True)

            sampled_key = sampled_key.permute(0, 2, 3, 1).contiguous()
            
            if torch.all(key.eq(value)):
                sampled_value = sampled_key
            else:
                if self.value_need_reshape:
                    sample_inp = value.transpose(1,2).contiguous().view((B, C, H, W))
                else:
                    sample_inp = value
                sampled_value = F.grid_sample(sample_inp, points, padding_mode='zeros')
                sampled_value = sampled_value.permute(0, 2, 3, 1).contiguous()
        else:
            sampled_key = key
            sampled_value = value
        
        B, T, N, C = sampled_key.shape
        q = self.query(query).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(sampled_key).view(B, T, N, self.n_head, C // self.n_head).permute(0, 3, 1, 2, 4)
        v = self.value(sampled_value).view(B, T, N, self.n_head, C // self.n_head).permute(0, 3, 1, 2, 4)

        att = torch.einsum('bhij, bhikj->bhik', q, k) * (1.0 /math.sqrt(k.size(-1)))
        #att = []
        #for i in range(T):
        #    att.append(torch.einsum('bhj, bhkj->bhk', q[:, :, i, :], k[:, :, i, :, :]))
        #att = torch.stack(att, dim = 2)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = torch.einsum('bhij, bhijk-> bhik', att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return y

class SegAndDensityTransformer(nn.Module):

    def __init__(self, bev_grid_size = (50, 50), density_grid_size = (200, 200), neighbourhood_size = 5,
            n_embed = 256, n_head = 8, num_layers = 6, dim_feedforward=1024, normalize_before = False, activation='relu',
            embed_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1):
        super().__init__()
        self.config = TransformerConfig(n_embed=n_embed, n_head=n_head, num_layers=num_layers, bev_grid_size=bev_grid_size,
                            density_grid_size=density_grid_size, neighbourhood_size=neighbourhood_size,
                            dim_feedforward=dim_feedforward, normalize_before=normalize_before, activation=activation,
                            embed_pdrop=embed_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop)
        
        #neighbourhood_mask = construct_neighbourhood_attention_mask(bev_grid_size, neighbourhood_size)
        neighbourhood_grid = construct_neighbourhood_attention_grid(bev_grid_size, neighbourhood_size)
        #density_attention_mask = construct_density_attention_mask(bev_grid_size,density_grid_size)
        density_attention_grid = construct_density_attention_grid(bev_grid_size, density_grid_size)

        #self.layer = SegAndDensityTransformerLayer(self.config, neighbourhood_mask, density_attention_mask)
        self.layer = SegAndDensityTransformerLayer(self.config, neighbourhood_grid, density_attention_grid)
        self.layers = _get_clones(self.layer, num_layers)
        self.norm = nn.LayerNorm(n_embed) if normalize_before else None

        self.density_encoder = DensityEncoder(n_embed)

    def forward(self, x, density, x_pos, density_pos):
        density = self.density_encoder(density)

        B, C, D, X = x.shape
        assert (D, X) == self.config.bev_grid_size
        x = x.permute(0, 2, 3, 1).contiguous().view(B, D * X, C)
        if x_pos is not None:
            x_pos = x_pos.permute(0, 2, 3, 1).contiguous().view(B, D * X, C)

        for layer in self.layers:
            x, density = layer(x, density, x_pos, density_pos)

        if self.norm is not None:
            x = self.norm(x)

        H, W = self.config.bev_grid_size
        x = x.view(B, H, W, self.config.n_embed).permute(0, 3, 1, 2)
        return x, density

class DensityEncoder(nn.Module):
    
    def __init__(self, out_channels, layers = [2, 2, 2, 2], dilation = 2, blocktype = 'basic'):
        super().__init__()
        modules = list()
        channels = [1]
        channels += [out_channels // (2 **i) for i in reversed(range(len(layers)))]
        for i in range(len(layers)):

            # Add a new residual layer
            module = ResNetLayer(channels[i],
                    channels[i+1], layers[i], dilation=dilation, blocktype=blocktype)
            modules.append(module)
        
        self.encoder = nn.Sequential(*modules)

    def forward(self, density_grid):

        return self.encoder(density_grid)


class SegAndDensityTransformerLayer(nn.Module):

    def __init__(self, config, neighbourhood_grid, density_attention_grid):
        super().__init__()

        #self.register_buffer('neighbourhood_mask', neighbourhood_mask)
        #self.register_buffer('density_attention_mask', density_attention_mask)
        
        self.cross_attn = SampleAttentionModule(config, sample_grid = density_attention_grid, key_need_reshape = False, value_need_reshape = False)
        #self.cross_attn = nn.MultiheadAttention(config.n_embed, config.n_head, dropout = config.attn_pdrop, batch_first=True)
        #self.neighbourhood_attn = nn.MultiheadAttention(config.n_embed, config.n_head, dropout=config.attn_pdrop, batch_first=True)
        self.neighbourhood_attn = SampleAttentionModule(config, sample_grid = neighbourhood_grid, samp_input_dim = config.bev_grid_size)
        self.mlp = nn.Sequential(
                nn.Linear(config.n_embed, config.dim_feedforward),
                _get_activation_function(config.activation),
                nn.Dropout(config.resid_pdrop),
                nn.Linear(config.dim_feedforward, config.n_embed),
                nn.Dropout(config.resid_pdrop))
        self.resnet_layer = ResNetLayer(config.n_embed, config.n_embed, 2, dilation = 2, blocktype= 'basic')

        if config.normalize_before:
            self.norm1 = nn.LayerNorm(config.n_embed)
        else:
            self.norm1 = nn.Identity()
        self.norm2 = nn.LayerNorm(config.n_embed)
        self.norm3 = nn.LayerNorm(config.n_embed)
        if config.normalize_before:
            self.norm4 = nn.Identity()
        else:
            self.norm4 = nn.LayerNorm(config.n_embed)

        self.dropout1 = nn.Dropout(config.resid_pdrop)
        self.dropout2 = nn.Dropout(config.resid_pdrop)
    
    def with_pos_embed(self, x, pos):
        return x if pos is None else x + pos

    def forward(self, x, density, x_pos, density_pos):
        x = self.norm1(x)
        #B, C, H, W = density.shape
        #dens_flattened = density.reshape(B, C, H * W).permute(0, 2, 1)
        res = self.cross_attn(self.with_pos_embed(x, x_pos), self.with_pos_embed(density, density_pos), density)
        x = self.norm2(x + self.dropout1(res))
        res = self.neighbourhood_attn(self.with_pos_embed(x, x_pos), self.with_pos_embed(x, x_pos), x)
        x = self.norm3(x + self.dropout2(res))
        x = self.norm4(x + self.mlp(x))
        density = self.resnet_layer(density)
        return x, density

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
    transformer = SegAndDensityTransformer().cuda()
    #module = SampleAttentionModule()
    #x = torch.rand((1, 49 * 50, 512), dtype = torch.float32)
    #grid = construct_ray_attention_grid()
    for _ in range(10000):
        x = torch.rand((1, 256, 50, 50), dtype = torch.float32).cuda()
        density = torch.rand((1, 1, 200, 200), dtype = torch.float32).cuda()
    #cam = torch.tensor([[175, 0, 150], [0, 175, 37.5], [0, 0, 1]], dtype = torch.float32)
        x, density = transformer(x, density)
        print(x.shape)
        print(density.shape)
    #print(grid.shape)

if __name__ == '__main__':
    _test()

