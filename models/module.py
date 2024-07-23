import torch
from torch import nn, einsum
from einops import rearrange

class SepConv3d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,):
        super(SepConv3d, self).__init__()
        self.depthwise = torch.nn.Conv3d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.bn = torch.nn.BatchNorm3d(in_channels)
        self.pointwise = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class ConvAttention_conv5(nn.Module):
    def __init__(self, dim, img_size, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False):

        super().__init__()
        self.last_stage = last_stage
        self.img_size = img_size 
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv3d(dim, inner_dim, kernel_size, q_stride, pad) 
        self.to_k = SepConv3d(dim, inner_dim, kernel_size, k_stride, pad) 
        self.to_v = SepConv3d(dim, inner_dim, kernel_size, v_stride, pad) 

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, a, h = *x.shape, self.heads

        x = rearrange(x, 'b (z l w) n -> b n z l w', z=4, l=8, w=8) 
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) z l w -> b h (z l w) d', h=h) 

        v = self.to_v(x)
        v = rearrange(v, 'b (h d) z l w -> b h (z l w) d', h=h)

        k = self.to_k(x)
        k = rearrange(k, 'b (h d) z l w -> b h (z l w) d', h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class ConvAttention_conv4(nn.Module):
    def __init__(self, dim, img_size, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False):

        super().__init__()
        self.last_stage = last_stage
        self.img_size = img_size # h w? 
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv3d(dim, inner_dim, kernel_size, q_stride, pad) 
        self.to_k = SepConv3d(dim, inner_dim, kernel_size, k_stride, pad) 
        self.to_v = SepConv3d(dim, inner_dim, kernel_size, v_stride, pad) 

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, a, h = *x.shape, self.heads
        
        x = rearrange(x, 'b (z l w) n -> b n z l w', z=8, l=16, w=16) 
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) z l w -> b h (z l w) d', h=h) 

        v = self.to_v(x)
        v = rearrange(v, 'b (h d) z l w -> b h (z l w) d', h=h)

        k = self.to_k(x)
        k = rearrange(k, 'b (h d) z l w -> b h (z l w) d', h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out