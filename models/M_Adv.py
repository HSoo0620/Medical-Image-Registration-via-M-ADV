import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import numpy as np
import copy, utils
### for CvT ###
from einops import repeat
from einops.layers.torch import Rearrange
from models.module import ConvAttention_conv4, ConvAttention_conv5, PreNorm, FeedForward
from einops.einops import rearrange
from .loftr_module import LocalFeatureTransformer
import models.configs_M_Adv as configs
from models.M_Adv_layers import EncoderBlock_first, EncoderBlock, UpBlock, DecoderBlock_conv, DecoderBlock_conv1, ResizeTransform, RegistrationHead, SpatialTransformer, SpatialTransformer2, ComposeTransform
from models.M_Adv_layers import Transformer4, Transformer5, PositionalEncoding3DTorch
### for CVT ###
    

class M_Adv_model(nn.Module):
    def __init__(self, config):
        '''
        TransMorph Model
        '''
        super(M_Adv_model, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = 32

        ######### Encoder
        self.en_conv1 = EncoderBlock_first(1, embed_dim) 
        self.en_conv2 = EncoderBlock(embed_dim, embed_dim*2)
        self.en_conv3 = EncoderBlock(embed_dim*2, embed_dim*4) 

        cvt_in_channels_32 = embed_dim*4
        cvt_in_channels_64 = embed_dim*8
        cvt_scale = 2
        cvt_dim64 = embed_dim*8
        self.en_conv4_conv_embed = nn.Sequential(
            nn.Conv3d(cvt_in_channels_32, cvt_dim64, 3, 2, 1),
            Rearrange('b c z h w -> b (z h w) c', z = 8, h = 16, w = 16),
            nn.LayerNorm(cvt_dim64)
        )
        self.en_conv4_transformer = nn.Sequential(
            Transformer4(dim=cvt_dim64, img_size=(8, 16, 16), depth=10, heads=6, dim_head=64,
                                              mlp_dim=cvt_dim64 * cvt_scale, dropout=0.),
            Rearrange('b (z h w) c -> b c z h w',z = 8, h = 16, w = 16)
        )
        self.en_conv5_conv_embed = nn.Sequential(
            nn.Conv3d(cvt_in_channels_64, cvt_dim64, 3, 2, 1),
            Rearrange('b c z h w -> b (z h w) c', z = 4, h = 8, w = 8),
            nn.LayerNorm(cvt_dim64)
        )
        self.en_conv5_transformer = nn.Sequential(
            Transformer5(dim=cvt_dim64, img_size=(4, 8, 8), depth=10, heads=6, dim_head=64,
                                              mlp_dim=cvt_dim64 * cvt_scale, dropout=0.),
            Rearrange('b (z h w) c -> b c z h w',z = 4, h = 8, w = 8)
        )

        ########## LoFTR               
        self.feat5_lofter = PositionalEncoding3DTorch(channels=embed_dim*8)
        self.feat4_lofter = PositionalEncoding3DTorch(channels=embed_dim*8)
        self.loftr_coarse = LocalFeatureTransformer()

        ########## Decoder
        self.conv0 = DecoderBlock_conv(embed_dim*8, embed_dim*8, skip_channels=embed_dim*8 if if_transskip else 0, use_batchnorm=False)
        self.conv1 = DecoderBlock_conv(embed_dim*8, embed_dim*8, skip_channels=embed_dim*8 if if_transskip else 0, use_batchnorm=False)
        self.conv2 = DecoderBlock_conv(embed_dim*4, embed_dim*4, skip_channels=embed_dim*4 if if_transskip else 0, use_batchnorm=False)
        self.conv3 = DecoderBlock_conv(embed_dim*2, embed_dim*2, skip_channels=embed_dim*2 if if_transskip else 0, use_batchnorm=False)
        self.conv4 = DecoderBlock_conv(embed_dim, embed_dim, skip_channels=embed_dim if if_transskip else 0, use_batchnorm=False)

        self.conv1_prev = DecoderBlock_conv1(embed_dim*8, embed_dim*8, skip_channels=embed_dim*8 if if_transskip else 0, use_batchnorm=False)
        self.conv2_prev = DecoderBlock_conv1(embed_dim*8, embed_dim*4, skip_channels=embed_dim*4 if if_transskip else 0, use_batchnorm=False)
        self.conv3_prev = DecoderBlock_conv1(embed_dim*4, embed_dim*2, skip_channels=embed_dim*2 if if_transskip else 0, use_batchnorm=False)
        self.conv4_prev = DecoderBlock_conv1(embed_dim*2, embed_dim, skip_channels=embed_dim if if_transskip else 0, use_batchnorm=False)
        
        self.reg_head0 = RegistrationHead(in_channels=embed_dim*8, out_channels=3, kernel_size=3, )
        self.reg_head1 = RegistrationHead(in_channels=embed_dim*8, out_channels=3, kernel_size=3, )
        self.reg_head2 = RegistrationHead(in_channels=embed_dim*4, out_channels=3, kernel_size=3, )
        self.reg_head3 = RegistrationHead(in_channels=embed_dim*2, out_channels=3, kernel_size=3, )
        self.reg_head4 = RegistrationHead(in_channels=embed_dim, out_channels=3, kernel_size=3, )

        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )

        self.spatial_trans = SpatialTransformer(config.img_size)
        res = 1
        self.spatial_trans1 = SpatialTransformer((4*res, 8*res, 8*res))
        self.spatial_trans2 = SpatialTransformer((8*res, 16*res, 16*res))
        self.spatial_trans3 = SpatialTransformer((16*res, 32*res, 32*res))
        self.spatial_trans4 = SpatialTransformer((32*res, 64*res, 64*res))
        self.spatial_trans5 = SpatialTransformer((64*res, 128*res, 128*res))
        self.resize_feature = UpBlock()
        self.resize_flow = ResizeTransform(0.5, 3)

    def forward(self, x):
        moving = x[:, 0:1, :, :, :]
        fixed = x[:, 1:2, :, :, :]

        ###### Encoder #####
        m_f4 = self.en_conv1(moving)
        m_f3 = self.en_conv2(m_f4) 
        m_f2 = self.en_conv3(m_f3)
        m_f1 = self.en_conv4_conv_embed(m_f2)  
        m_f1 = self.en_conv4_transformer(m_f1)
        m_f0 = self.en_conv5_conv_embed(m_f1) 
        m_f0 = self.en_conv5_transformer(m_f0) 

        f_f4 = self.en_conv1(fixed)
        f_f3 = self.en_conv2(f_f4)
        f_f2 = self.en_conv3(f_f3)
        f_f1 = self.en_conv4_conv_embed(f_f2)
        f_f1 = self.en_conv4_transformer(f_f1)
        f_f0 = self.en_conv5_conv_embed(f_f1) 
        f_f0 = self.en_conv5_transformer(f_f0) 

        ##### Decoder #####
        m_d0 = rearrange(m_f0+self.feat5_lofter(m_f0), 'n c d h w -> n (d h w) c', d=4, h=8, w=8)
        f_d0 = rearrange(f_f0+self.feat5_lofter(f_f0), 'n c d h w -> n (d h w) c', d=4, h=8, w=8)
        m_d0, f_d0 = self.loftr_coarse(m_d0, f_d0, None, None)
        m_d0 = rearrange(m_d0, 'n (d h w) c -> n c d h w', d=4, h=8, w=8)
        f_d0 = rearrange(f_d0, 'n (d h w) c -> n c d h w', d=4, h=8, w=8)

        ##### DVF 0 #####
        x0 = self.conv0(m_d0, f_d0)
        flow0 = self.reg_head0(x0) # 4, 8, 8
        flow0_up = self.resize_flow(flow0) #
        
        m_f1_moved = self.spatial_trans2(m_f1, flow0_up)
        m_d0_moved = self.spatial_trans1(m_d0, flow0)
        m_d0_moved = self.resize_feature(m_d0_moved)
        m_concat_d0_f1 = self.conv1_prev(m_d0_moved, m_f1_moved)

        f_d0_up = self.resize_feature(f_d0)
        f_concat_d0_f1 = self.conv1_prev(f_d0_up, f_f1) 

        m_d1 = rearrange(m_concat_d0_f1+self.feat5_lofter(m_concat_d0_f1), 'n c d h w -> n (d h w) c', d=8, h=16, w=16)
        f_d1 = rearrange(f_concat_d0_f1+self.feat5_lofter(f_concat_d0_f1), 'n c d h w -> n (d h w) c', d=8, h=16, w=16)
        m_d1, f_d1 = self.loftr_coarse(m_d1, f_d1, None, None)
        m_d1 = rearrange(m_d1, 'n (d h w) c -> n c d h w', d=8, h=16, w=16)
        f_d1 = rearrange(f_d1, 'n (d h w) c -> n c d h w', d=8, h=16, w=16)

        ##### DVF 1 ######
        x1 = self.conv1(m_d1, f_d1)
        flow1 = self.reg_head1(x1)
        flow_composed1 = ComposeTransform(flow1.shape[2:])(flow0_up, flow1)
        flow1_up = self.resize_flow(flow_composed1) # 

        m_f2_moved = self.spatial_trans3(m_f2, flow1_up)
        m_d1_moved = self.spatial_trans2(m_d1, flow1)
        m_d1_moved = self.resize_feature(m_d1_moved)
        m_concat_d1_f2 = self.conv2_prev(m_d1_moved, m_f2_moved)

        f_d1_up = self.resize_feature(f_d1)
        f_concat_d1_f2 = self.conv2_prev(f_d1_up, f_f2) 

        ###### DVF 2 #####
        x2 = self.conv2(m_concat_d1_f2, f_concat_d1_f2)
        flow2 = self.reg_head2(x2)
        flow_composed2 = ComposeTransform(flow2.shape[2:])(flow1_up, flow2)
        flow_composed2_up = self.resize_flow(flow_composed2)

        m_f3_moved = self.spatial_trans4(m_f3, flow_composed2_up)
        m_d2_moved = self.spatial_trans3(m_concat_d1_f2, flow2)
        m_d2_moved = self.resize_feature(m_d2_moved)
        m_concat_d2_f3 = self.conv3_prev(m_d2_moved, m_f3_moved)

        f_d2_up = self.resize_feature(f_concat_d1_f2)
        f_concat_d2_f3 = self.conv3_prev(f_d2_up, f_f3)
        
        ##### DVF 3 ######     
        x3 = self.conv3(m_concat_d2_f3, f_concat_d2_f3)
        flow3 = self.reg_head3(x3) 
        flow_composed3 = ComposeTransform(flow3.shape[2:])(flow_composed2_up, flow3)
        flow_composed3_up = self.resize_flow(flow_composed3)

        m_f4_moved = self.spatial_trans5(m_f4, flow_composed3_up)
        m_d3_moved = self.spatial_trans4(m_concat_d2_f3, flow3)
        m_d3_moved = self.resize_feature(m_d3_moved)
        m_concat_d3_f4 = self.conv4_prev(m_d3_moved, m_f4_moved)
        
        ##### Final DVF #####
        f_d3_up = self.resize_feature(f_concat_d2_f3)
        f_concat_d3_f4 = self.conv4_prev(f_d3_up, f_f4) 

        x4 = self.conv4(m_concat_d3_f4, f_concat_d3_f4)
        flow4 = self.reg_head4(x4) 
        flow = ComposeTransform(flow4.shape[2:])(flow_composed3_up, flow4)
        out = self.spatial_trans(moving, flow)

        return  out, flow, flow0, flow_composed1, flow_composed2, flow_composed3

class Sagital_Discriminator(nn.Module):
    def __init__(self, input_channels=1, image_size=128, num_features=32, num_classes=1):
        super(Sagital_Discriminator, self).__init__()
        
        self.input_channels = input_channels
        self.image_size = image_size
        self.num_features = num_features
        self.num_classes = num_classes
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, num_features, kernel_size=4, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_features, num_features, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_features, num_features *2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_features * 2, num_features * 2, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_features * 2, num_features * 2, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_features * 2, num_features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_features * 4, num_features * 4, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_features * 4, num_features * 4, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_features * 4, num_features * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_features * 8, num_features * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv2d(num_features * 8, 1, kernel_size=1, stride=1, padding=1, bias=False),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(105, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class Discriminator3D(nn.Module):
    def __init__(self, input_channels=1, image_size=128, num_features=32, num_classes=1):
        super(Discriminator3D, self).__init__()
        
        self.input_channels = input_channels
        self.image_size = image_size
        self.num_features = num_features
        self.num_classes = num_classes
        
        self.conv_layers = nn.Sequential(
            nn.Conv3d(input_channels, num_features, kernel_size=4, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(num_features, num_features, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(num_features),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(num_features, num_features *2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(num_features * 2, num_features * 2, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(num_features * 2, num_features * 2, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(num_features * 2, num_features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(num_features * 4, num_features * 4, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(num_features * 4, num_features * 4, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(num_features * 4, num_features * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(num_features * 8, num_features * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(num_features * 8, 1, kernel_size=1, stride=2, padding=1, bias=False),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

CONFIGS = {
    'M-Adv': configs.get_M_Adv_config(),
}
