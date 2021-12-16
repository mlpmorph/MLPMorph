import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops.layers.torch import Rearrange
from . import layers
from .modelio import LoadableModel, store_config_args
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiheadMLP(nn.Module):
    def __init__(self, d_model, nhead, bias = True):
        super(MultiheadMLP, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.bias = bias
        self.in_proj_weight = nn.parameter.Parameter(torch.empty((d_model, d_model)))
        if bias:
            self.in_proj_bias = nn.parameter.Parameter(torch.empty(d_model))
        else:
            self.in_proj_bias = None
        self.out_proj = nn.modules.linear.NonDynamicallyQuantizableLinear(d_model, d_model, bias=bias)
        if isinstance(d_model, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            self.head_dim = d_model.div(nhead, rounding_mode='trunc')
        else:
            self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, f"embed_dim {d_model} not divisible by num_heads {nhead}"
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.bias:
            nn.init.constant_(self.in_proj_bias, 0.)

    def forward(self, src):
        # 1st projection
        bsize, plen, Ev = src.shape
        assert self.in_proj_weight.shape == (Ev, Ev), f"expecting value weights shape of {(Ev, Ev)}, but got {self.in_proj_weight}"
        src1 = torch._C._nn.linear(src, self.in_proj_weight, self.in_proj_bias)
        src2 = torch._C._nn.linear(src1, self.out_proj.weight, self.out_proj.bias)
        return src2


class MLPEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward = 1024, activation = 'relu', layer_norm_eps = 1e-5, dropout = 0.1):
        super(MLPEncoderLayer, self).__init__()
        self.self_proj = MultiheadMLP(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps = layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps = layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask = None):
        src1 = self.self_proj(src)
        src = src + self.dropout1(src1)
        src2 = self.norm1(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src2 + self.dropout2(src)
        src = self.norm2(src)
        return src


class MLPDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward = 1024, activation = 'relu', layer_norm_eps = 1e-5, dropout = 0.1):
        super(MLPDecoderLayer, self).__init__()
        self.self_proj = MultiheadMLP(d_model, nhead)
        self.cross_proj = MultiheadMLP(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None):
        tgt2 = self.self_proj(tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_proj(memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class MLPEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm_layer = None):
        super(MLPEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm_layer

    def forward(self, src, mask = None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask = mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class MLPDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm_layer = None):
        super(MLPDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm_layer

    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask = tgt_mask, memory_mask = memory_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, out_dim, dim, depth, heads, MLP= False, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.patch_sz = patch_size
        self.img_sz = image_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )
        self.mask_to_patch = nn.Sequential(
            nn.MaxPool2d(patch_size),
            Rearrange('b c (h 1) (w 1) -> b (h w) (1 1 c)'),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if MLP:
            encoder_layer = MLPEncoderLayer(d_model = dim, nhead=heads, dropout=dropout)
            self.transformer_encoder = MLPEncoder(encoder_layer, num_layers=depth)
            decoder_layer = MLPDecoderLayer(d_model=dim, nhead=heads, dropout=dropout)
            self.transformer_decoder = MLPDecoder(decoder_layer, num_layers=depth)
        else:
            encoder_layer =nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dropout=dropout, batch_first = True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
            decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=heads, dropout=dropout, batch_first = True)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(in_features=dim,  out_features=out_dim),
            nn.Tanh()
        )
        self.sin_embedding = PositionalEncoding(d_model=dim)

    def forward(self, img_fix, img_mov, tgt_mask=None, memory_mask=None):

        x_fix = self.to_patch_embedding(img_fix)
        b, n, pp = x_fix.shape
        x_fix += self.pos_embedding[:, :(n + 1)]
        # x_fix -- encoder -- memory
        # x_mov -- decoder -- tgt

        x_mov = self.to_patch_embedding(img_mov)
        b, n, pp = x_mov.shape
        x_mov += self.pos_embedding[:, :(n + 1)]

        memory = self.transformer_encoder(x_fix, mask=memory_mask)
        vf1 = self.transformer_decoder(x_mov, memory)#, tgt_mask=tgt_mask, memory_mask=memory_mask) # Get the feature map pf velocity field
        vf = self.to_latent(vf1)
        vf = self.mlp_head(vf)
        vf = torch.reshape(torch.transpose(vf, -2, -1), (b, 2, np.int(self.img_sz/self.patch_sz), np.int(self.img_sz/self.patch_sz)))  # _,_,image_size/patch_size
        return vf


class DIRNet(LoadableModel):
    @store_config_args
    def __init__(self, inshape, int_steps, patch_size= [4,8,16], out_dim = [2,2,2], dim = [128,128,128],
                 depth = [1,1,1], heads = [128, 64, 32], int_downsize = 2, bidir=True, MLP =False):
        super().__init__()
        self.bidir = bidir
        self.ViT_1 = ViT(image_size = inshape[0], patch_size = patch_size[0], out_dim = out_dim[0], dim = dim[0], depth = depth[0], heads = heads[0], dropout = 0.0, emb_dropout = 0.1, MLP = MLP)
        self.ViT_4 = ViT(image_size = inshape[0], patch_size = patch_size[1], out_dim = out_dim[1], dim = dim[1], depth = depth[1], heads = heads[1], dropout = 0.0, emb_dropout = 0.1, MLP = MLP)
        self.ViT_8 = ViT(image_size = inshape[0], patch_size = patch_size[2], out_dim = out_dim[2], dim = dim[2], depth = depth[2], heads = heads[2], dropout = 0.0, emb_dropout = 0.1, MLP = MLP)

        self.loss2 = torch.nn.MSELoss()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample = nn.Upsample(scale_factor=int(patch_size[0]/int_downsize), mode='bilinear')
        self.register_parameter(name='w1', param=torch.nn.Parameter(torch.tensor([0.7])))
        self.register_parameter(name='w2', param=torch.nn.Parameter(torch.tensor([0.2])))
        self.register_parameter(name='w3', param=torch.nn.Parameter(torch.tensor([0.1])))

        vecshape = [int(sh/int_downsize) for sh in inshape]

        self.integrate = layers.VecInt(vecshape, int_steps) if int_steps > 0 else None
        self.transformer = layers.SpatialTransformer(inshape)

        ndims = len(inshape)
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)

    # Dim = 4 bu shou lian; Dim  = 16 shou lian

    def forward(self, x, y, tgt_mask = None, memory_mask = None, registration = False):
        # tgt_mask: y
        # memory mask: x
        v1 = self.ViT_1(x, y, tgt_mask=tgt_mask, memory_mask=memory_mask)
        v4 = self.ViT_4(x, y, tgt_mask=tgt_mask, memory_mask=memory_mask)
        v8 = self.ViT_8(x, y, tgt_mask=tgt_mask, memory_mask=memory_mask)
        v4 = self.upsample2(v4)
        v8 = self.upsample4(v8)

        # velocity
        v = (self.w1 * v1 + self.w2 * v4 + (1- self.w1 - self.w2) * v8)

        preint_flow = self.upsample(v)
        pos_flow = preint_flow
        neg_flow = -preint_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

        # rescale the flow
        pos_flow = self.fullsize(pos_flow)
        neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        trans_x = self.transformer(x, pos_flow)
        trans_y = self.transformer(y, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (trans_x, trans_y, preint_flow) if self.bidir else (trans_x, preint_flow)
        else:
            return (trans_x, trans_y, pos_flow, neg_flow) if self.bidir else(trans_x, pos_flow)
