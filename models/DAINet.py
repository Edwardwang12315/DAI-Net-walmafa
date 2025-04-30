# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable, Function

from layers import *
from data.config import cfg

import numpy as np
import matplotlib.pyplot as plt

from models import blocks
from mamba_ssm import Mamba
import numbers

import einops
from einops import rearrange
from visualizer import get_local

m = None

def data_transform( X ) :
    return 2 * X - 1.0


def inverse_data_transform( X ) :
    return torch.clamp( (X + 1.0) / 2.0 , 0.0 , 1.0 )


class Conv2d_BN( torch.nn.Sequential ) :
    def __init__( self , a , b , ks = 1 , stride = 1 , pad = 0 , dilation = 1 ,
                  groups = 1 , bn_weight_init = 1 , resolution = -10000 ) :
        super().__init__()
        self.add_module( 'c' , torch.nn.Conv2d(
                a , b , ks , stride , pad , dilation , groups , bias = False ) )
        self.add_module( 'bn' , torch.nn.BatchNorm2d( b ) )
        torch.nn.init.constant_( self.bn.weight , bn_weight_init )
        torch.nn.init.constant_( self.bn.bias , 0 )
    
    @torch.no_grad()
    def fuse( self ) :
        c , bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[ : , None , None , None ]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d( w.size( 1 ) * self.c.groups , w.size(
                0 ) , w.shape[ 2 : ] , stride = self.c.stride , padding = self.c.padding , dilation = self.c.dilation ,
                             groups = self.c.groups ,
                             device = c.weight.device )
        m.weight.data.copy_( w )
        m.bias.data.copy_( b )
        return m


##########################################################################
## Layer Norm

def to_3d( x ) :
    return rearrange( x , 'b c h w -> b (h w) c' )


def to_4d( x , h , w ) :
    return rearrange( x , 'b (h w) c -> b c h w' , h = h , w = w )


class BiasFree_LayerNorm( nn.Module ) :
    def __init__( self , normalized_shape ) :
        super( BiasFree_LayerNorm , self ).__init__()
        if isinstance( normalized_shape , numbers.Integral ) :
            normalized_shape = (normalized_shape ,)
        normalized_shape = torch.Size( normalized_shape )
        
        assert len( normalized_shape ) == 1
        
        self.weight = nn.Parameter( torch.ones( normalized_shape ) )
        self.normalized_shape = normalized_shape
    
    def forward( self , x ) :
        sigma = x.var( -1 , keepdim = True , unbiased = False )  ##返回所有元素的方差
        return x / torch.sqrt( sigma + 1e-5 ) * self.weight


class WithBias_LayerNorm( nn.Module ) :
    def __init__( self , normalized_shape ) :
        super( WithBias_LayerNorm , self ).__init__()
        if isinstance( normalized_shape , numbers.Integral ) :
            normalized_shape = (normalized_shape ,)
        normalized_shape = torch.Size( normalized_shape )
        
        assert len( normalized_shape ) == 1
        
        self.weight = nn.Parameter( torch.ones( normalized_shape ) )
        self.bias = nn.Parameter( torch.zeros( normalized_shape ) )
        self.normalized_shape = normalized_shape
    
    def forward( self , x ) :
        mu = x.mean( -1 , keepdim = True )
        sigma = x.var( -1 , keepdim = True , unbiased = False )
        return (x - mu) / torch.sqrt( sigma + 1e-5 ) * self.weight + self.bias


class LayerNorm( nn.Module ) :
    def __init__( self , dim , LayerNorm_type ) :
        super( LayerNorm , self ).__init__()
        if LayerNorm_type == 'BiasFree' :
            self.body = BiasFree_LayerNorm( dim )
        else :
            self.body = WithBias_LayerNorm( dim )
    
    def forward( self , x ) :
        h , w = x.shape[ -2 : ]
        return to_4d( self.body( to_3d( x ) ) , h , w )


class NextAttentionImplZ( nn.Module ) :
    def __init__( self , num_dims , num_heads , bias ) -> None :
        super().__init__()
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.q1 = nn.Conv2d( num_dims , num_dims * 3 , kernel_size = 1 , bias = bias )
        self.q2 = nn.Conv2d( num_dims * 3 , num_dims * 3 , kernel_size = 3 , padding = 1 , groups = num_dims * 3 , bias = bias )
        self.q3 = nn.Conv2d( num_dims * 3 , num_dims * 3 , kernel_size = 3 , padding = 1 , groups = num_dims * 3 , bias = bias )
        
        self.fac = nn.Parameter( torch.ones( 1 ) )
        self.fin = nn.Conv2d( num_dims , num_dims , kernel_size = 1 , bias = bias )
        return
    
    def forward( self , x , ill_map , mask = None ) :
        # x: [n, c, h, w]
        n , c , h , w = x.size()
        n_heads , dim_head = self.num_heads , c // self.num_heads
        reshape = lambda x : einops.rearrange( x , "n (nh dh) h w -> (n nh w) h dh" , nh = n_heads , dh = dim_head )
        
        qkv = self.q3( self.q2( self.q1( x ) ) )
        q , k , v = map( reshape , qkv.chunk( 3 , dim = 1 ) )
        ill = reshape( ill_map )
        v = v * ill
        q = F.normalize( q , dim = -1 )
        k = F.normalize( k , dim = -1 )
        # fac = dim_head ** -0.5
        res = k.transpose( -2 , -1 )
        res = torch.matmul( q , res ) * self.fac
        
        # 置为-1e9 因为softmax中0输出为1，我们想要输出为0就是负无穷大
        if mask is not None :
            mask = reshape( mask )
            mask = torch.matmul( mask , mask.transpose( -2 , -1 ) ) * self.fac
            res = res.masked_fill( mask == 0 , -1e9 )
        res = torch.softmax( res , dim = -1 )
        
        res = torch.matmul( res , v )
        res = einops.rearrange( res , "(n nh w) h dh -> n (nh dh) h w" , nh = n_heads , dh = dim_head , n = n , h = h )
        res = self.fin( res )
        
        return res


class NextAttentionZ( nn.Module ) :
    def __init__( self , num_dims , num_heads = 1 , bias = True ) -> None :
        super().__init__()
        assert num_dims % num_heads == 0
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.row_att = NextAttentionImplZ( num_dims , num_heads , bias )
        self.col_att = NextAttentionImplZ( num_dims , num_heads , bias )
        return
    
    def forward( self , x: torch.Tensor , ill_map_height , ill_map_width , mask = None ) :
        assert len( x.size() ) == 4
        x = self.row_att( x , ill_map_width , mask = mask )
        x = x.transpose( -2 , -1 )
        if mask is not None :
            x = self.col_att( x , ill_map_height.transpose( -2 , -1 ) , mask = mask.transpose( -2 , -1 ) , )
        else :
            x = self.col_att( x , ill_map_height.transpose( -2 , -1 ) , mask = mask )
        x = x.transpose( -2 , -1 )
        
        return x


class FeedForward( nn.Module ) :
    def __init__( self , dim , ffn_expansion_factor , bias ) :
        super( FeedForward , self ).__init__()
        
        hidden_features = int( dim * ffn_expansion_factor )
        
        self.rep_conv1 = Conv2d_BN( hidden_features , hidden_features , 3 , 1 , 1 , groups = hidden_features )
        self.rep_conv2 = Conv2d_BN( hidden_features , hidden_features , 1 , 1 , 0 , groups = hidden_features )
        
        self.project_in = nn.Conv2d( dim , hidden_features , kernel_size = 1 , bias = bias )
        
        self.dwconv = nn.Conv2d( hidden_features , hidden_features , kernel_size = 3 , stride = 1 , padding = 1 ,
                                 groups = hidden_features , bias = bias )
        
        self.project_out = nn.Conv2d( hidden_features , dim , kernel_size = 1 , bias = bias )
    
    def forward( self , x ) :
        identity = x
        x = self.project_in( x )
        x1 = x + self.rep_conv1( x ) + self.rep_conv2( x )
        x2 = self.dwconv( x )
        x = F.gelu( x2 ) * x1 + F.gelu( x1 ) * x2
        x = self.project_out( x )
        return x + identity
    
    @torch.no_grad()
    def fuse( self ) :
        conv = self.rep_conv1.fuse()  ##Conv_BN
        conv1 = self.rep_conv2.fuse()  ##Conv_BN
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad( conv1_w , [ 1 , 1 , 1 , 1 ] )
        
        identity = torch.nn.functional.pad( torch.ones( conv1_w.shape[ 0 ] , conv1_w.shape[ 1 ] , 1 , 1 , device = conv1_w.device ) ,
                                            [ 1 , 1 , 1 , 1 ] )
        
        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b
        
        conv.weight.data.copy_( final_conv_w )
        conv.bias.data.copy_( final_conv_b )
        return conv


class WM( nn.Module ) :
    def __init__( self , c = 3 ) :
        super().__init__()
        self.convb = nn.Sequential(
                nn.Conv2d( in_channels = c , out_channels = c * 2 , kernel_size = 3 , stride = 1 , padding = 1 ) ,
                nn.ReLU() ,
                nn.Conv2d( in_channels = c * 2 , out_channels = c , kernel_size = 3 , stride = 1 , padding = 1 )
        )
        self.model1 = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model = c ,  # Model dimension d_model
                d_state = 32 ,  # SSM state expansion factor
                d_conv = 4 ,  # Local convolution width
                expand = 2 ,  # Block expansion factor
        )
        
        self.model2 = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model = c ,  # Model dimension d_model
                d_state = 32 ,  # SSM state expansion factor
                d_conv = 4 ,  # Local convolution width
                expand = 9 ,  # Block expansion factor
        )
        self.smooth = nn.Conv2d( in_channels = c , out_channels = c , kernel_size = 3 , stride = 1 , padding = 1 )
        self.ln = nn.LayerNorm( normalized_shape = c )
        self.softmax = nn.Softmax()
    
    def forward( self , x ) :
        b , c , h , w = x.shape
        x = self.convb( x ) + x
        x = self.ln( x.reshape( b , -1 , c ) )
        
        y = self.model1( x ).permute( 0 , 2 , 1 )
        output = y.reshape( b , c , h , w )
        return self.smooth( output )


class WMB( nn.Module ) :
    def __init__( self , dim , num_heads = 1 , ffn_expansion_factor = 2.66 , bias = True , LayerNorm_type = 'WithBias' ) :
        super( WMB , self ).__init__()
        self.DWT = blocks.DWT()
        self.IWT = blocks.IWT()
        self.norm1 = LayerNorm( dim , LayerNorm_type )
        self.illu = Illumination_Estimator( dim , n_fea_in = dim + 1 , n_fea_out = dim )
        self.norm2 = LayerNorm( dim , LayerNorm_type )
        self.ffn = FeedForward( dim , ffn_expansion_factor , bias )
        self.mb = WM( dim )
    
    def forward( self , input_ ) :
        global m
        x = input_
        n , c , h , w = x.shape
        x = self.norm1( x )
        x = data_transform( x )
        input_dwt = self.DWT( x )
        # input_LL=A [B,C,H/2,W/2]   input_high0={V,H,D} [3B,C,H/2,W/2]
        input_LL , input_high = input_dwt[ :n , ... ] , input_dwt[ n : , ... ]
        input_LL , input_image = self.illu( input_LL )
        input_high = self.mb( input_high )
        
        output = self.IWT( torch.cat( (input_LL , input_high) , dim = 0 ) )
        output = inverse_data_transform( output )
        
        x = x + output
        x = x + self.ffn( self.norm2( x ) )
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed( nn.Module ) :
    def __init__( self , in_c = 3 , embed_dim = 48 , bias = False ) :
        super( OverlapPatchEmbed , self ).__init__()
        
        self.proj = nn.Conv2d( in_c , embed_dim , kernel_size = 3 , stride = 1 , padding = 1 , bias = bias )
    
    def forward( self , x ) :
        x = self.proj( x )
        
        return x


class Downsample( nn.Module ) :
    def __init__( self , n_feat ) :
        super( Downsample , self ).__init__()
        
        self.body = nn.Sequential( nn.Conv2d( n_feat , n_feat // 2 , kernel_size = 3 , stride = 1 , padding = 1 , bias = False ) ,
                                   nn.PixelUnshuffle( 2 ) )
    
    def forward( self , x ) :
        return self.body( x )


class Upsample( nn.Module ) :
    def __init__( self , n_feat ) :
        super( Upsample , self ).__init__()
        
        self.body = nn.Sequential( nn.Conv2d( n_feat , n_feat * 2 , kernel_size = 3 , stride = 1 , padding = 1 , bias = False ) ,
                                   nn.PixelShuffle( 2 ) )
    
    def forward( self , x ) :
        return self.body( x )


class Illumination_Estimator( nn.Module ) :
    def __init__(
            self , n_fea_middle , n_fea_in = 4 , n_fea_out = 3 ) :  # __init__部分是内部属性，而forward的输入才是外部输入
        super( Illumination_Estimator , self ).__init__()
        
        self.conv1 = nn.Conv2d( n_fea_in , n_fea_middle , kernel_size = 1 , bias = True )
        
        self.depth_conv = nn.Conv2d(
                n_fea_middle , n_fea_middle , kernel_size = 5 , padding = 2 , bias = True , groups = n_fea_middle )
        
        self.conv2 = nn.Conv2d( n_fea_middle , n_fea_out , kernel_size = 1 , bias = True )
    
    def forward( self , img ) :
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        mean_c = img.mean( dim = 1 ).unsqueeze( 1 )
        # stx()
        input = torch.cat( [ img , mean_c ] , dim = 1 )
        
        x_1 = self.conv1( input )
        illu_fea = self.depth_conv( x_1 )
        illu_map = self.conv2( illu_fea )
        return illu_fea , illu_map

class Interpolate(nn.Module):
    # 插值的方法对张量进行上采样或下采样
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return x


class FEM(nn.Module):
    """docstring for FEM"""

    def __init__(self, in_planes):
        super(FEM, self).__init__()
        inter_planes = in_planes // 3
        inter_planes1 = in_planes - 2 * inter_planes
        self.branch1 = nn.Conv2d(
            in_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_planes, inter_planes, kernel_size=3,
                      stride=1, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_planes, inter_planes, kernel_size=3,
                      stride=1, padding=3, dilation=3)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_planes, inter_planes1, kernel_size=3,
                      stride=1, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_planes1, inter_planes1, kernel_size=3,
                      stride=1, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_planes1, inter_planes1, kernel_size=3,
                      stride=1, padding=3, dilation=3)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = F.relu(out, inplace=True)
        return out


class DSFD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, fem, head1, head2, num_classes):
        super(DSFD, self).__init__()
        self.enh = True
        self.det = True
        
        if self.det:
            self.phase = phase
            self.num_classes = num_classes
            self.vgg = nn.ModuleList(base)
    
            self.L2Normof1 = L2Norm(256, 10)
            self.L2Normof2 = L2Norm(512, 8)
            self.L2Normof3 = L2Norm(512, 5)
    
            self.extras = nn.ModuleList(extras)
            self.fpn_topdown = nn.ModuleList(fem[0])
            self.fpn_latlayer = nn.ModuleList(fem[1])
    
            self.fpn_fem = nn.ModuleList(fem[2])
    
            self.L2Normef1 = L2Norm(256, 10)
            self.L2Normef2 = L2Norm(512, 8)
            self.L2Normef3 = L2Norm(512, 5)
    
            self.loc_pal1 = nn.ModuleList(head1[0])#nn.ModuleList是一种存储子模块的工具
            self.conf_pal1 = nn.ModuleList(head1[1])
    
            self.loc_pal2 = nn.ModuleList(head2[0])
            self.conf_pal2 = nn.ModuleList(head2[1])
    
            if self.phase == 'test':
                self.softmax = nn.Softmax(dim=-1)
                self.detect = Detect(cfg)
        
        if self.enh :
            inp_channels = 3
            out_channels = 3
            dim = 16
            num_blocks = [ 2 , 3 , 4 ]
            heads = [ 1 , 2 , 4 , 8 ]
            ffn_expansion_factor = 2.66
            bias = False
            LayerNorm_type = 'WithBias'
            attention = True
            skip = False
            
            self.estimator = Illumination_Estimator( dim )
            
            self.coefficient = nn.Parameter( torch.Tensor( np.ones( (4 , 2 , int( int( dim * 2 * 4 ) )) ) ) ,
                                             requires_grad = attention )
            
            self.patch_embed = OverlapPatchEmbed( inp_channels , dim )
            self.patch_embed_mask = OverlapPatchEmbed( 1 , dim )
            
            self.sim = nn.Sequential( *[
                    WMB( dim = int( dim ) , num_heads = heads[ 0 ] , ffn_expansion_factor = ffn_expansion_factor ,
                         bias = bias , LayerNorm_type = LayerNorm_type ) for i in range( num_blocks[ 0 ] ) ] )
            
            self.down_1 = Downsample( int( dim ) )  # From Level 0 to Level 1
            
            self.decoder_level1_0 = nn.Sequential( *[ WMB( dim = int( int( dim * 2 ) ) , num_heads = heads[ 1 ] ,
                                                           ffn_expansion_factor = ffn_expansion_factor ,
                                                           bias = bias , LayerNorm_type = LayerNorm_type )
                                                      for i in range( num_blocks[ 0 ] ) ] )
            
            self.down_2 = Downsample( int( dim * 2 ) )  # From Level 1 to Level 2
            self.decoder_level2_0 = nn.Sequential( *[
                    WMB( dim = int( int( dim * 2 * 2 ) ) , num_heads = heads[ 2 ] ,
                         ffn_expansion_factor = ffn_expansion_factor , bias = bias ,
                         LayerNorm_type = LayerNorm_type ) for i in range( num_blocks[ 1 ] ) ] )
            
            self.down_3 = Downsample( int( dim * 2 * 2 ) )  # From Level 2 to Level 3
            self.decoder_level3_0 = nn.Sequential( *[
                    WMB( dim = int( int( dim * 2 * 4 ) ) , num_heads = heads[ 3 ] ,
                         ffn_expansion_factor = ffn_expansion_factor , bias = bias ,
                         LayerNorm_type = LayerNorm_type ) for i in range( num_blocks[ 2 ] ) ] )
            self.latent = blocks.FFAB( int( int( dim * 2 * 4 ) ) )
            
            self.up3_2 = Upsample( int( dim * 2 * 4 ) )  # From Level 3 to Level 2
            self.decoder_level2_1 = nn.Sequential( *[
                    WMB( dim = int( int( dim * 2 * 2 ) ) , num_heads = heads[ 2 ] ,
                         ffn_expansion_factor = ffn_expansion_factor , bias = bias ,
                         LayerNorm_type = LayerNorm_type ) for i in range( num_blocks[ 1 ] ) ] )
            
            self.up2_1 = Upsample( int( dim * 2 * 2 ) )  # From Level 2 to Level 1
            self.decoder_level1_1 = nn.Sequential( *[ WMB( dim = int( int( dim * 2 ) ) , num_heads = heads[ 1 ] ,
                                                           ffn_expansion_factor = ffn_expansion_factor ,
                                                           bias = bias , LayerNorm_type = LayerNorm_type )
                                                      for i in range( num_blocks[ 0 ] ) ] )
            self.up2_0 = Upsample( int( dim * 2 ) )  # From Level 1 to Level 0
            # skip connection wit weights
            self.coefficient_3_2 = nn.Parameter( torch.Tensor( np.ones( (2 , int( int( dim * 2 * 2 ) )) ) ) , requires_grad = attention )
            self.coefficient_2_1 = nn.Parameter( torch.Tensor( np.ones( (2 , int( int( dim * 2 ) )) ) ) , requires_grad = attention )
            self.coefficient_1_0 = nn.Parameter( torch.Tensor( np.ones( (2 , int( int( dim ) )) ) ) , requires_grad = attention )
            
            # skip then conv 1x1
            self.skip_3_2 = nn.Conv2d( int( int( dim * 2 * 2 ) ) , int( int( dim * 2 * 2 ) ) , kernel_size = 1 , bias = bias )
            self.skip_2_1 = nn.Conv2d( int( int( dim * 2 ) ) , int( int( dim * 2 ) ) , kernel_size = 1 , bias = bias )
            self.skip_1_0 = nn.Conv2d( int( int( dim * 2 ) ) , int( int( dim * 2 ) ) , kernel_size = 1 , bias = bias )
            
            self.sim = nn.Sequential( *[
                    WMB( dim = int( dim ) , num_heads = heads[ 0 ] , ffn_expansion_factor = ffn_expansion_factor ,
                         bias = bias , LayerNorm_type = LayerNorm_type ) for i in range( num_blocks[ 0 ] ) ] )
            
            self.output = nn.Conv2d( int( dim ) , out_channels , kernel_size = 3 , stride = 1 , padding = 1 , bias = bias )
            self.skip = skip
            
    def _upsample_prod(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') * y
    # 反射图解码通路
    def enh_forward(self, x):

        x = x[:1]
        for k in range(5):
            x = self.vgg[k](x)

        R = self.ref(x)

        return R

    def test_forward(self, x):
        size = x.size()[2:]
        pal1_sources = list()
        pal2_sources = list()
        loc_pal1 = list()
        conf_pal1 = list()
        loc_pal2 = list()
        conf_pal2 = list()
        
        for k in range(16):
            x = self.vgg[k](x)
            if k == 4:
                x_ = x
        R = self.ref(x_[0:1])
        
        # print(f'x.shape = {x.shape}')
        # exit()
        
        # print( '暗图' )
        # image = np.transpose( R[ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
        # image = (image * 255).astype( np.uint8 )
        # plt.imshow( image )
        # plt.axis( 'off' )
        # # 保存图像到文件
        # plt.savefig( f'test_暗图.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )
        # exit()
        
        of1 = x
        s = self.L2Normof1(of1)
        pal1_sources.append(s)
        # apply vgg up to fc7
        for k in range(16, 23):
            x = self.vgg[k](x)
        of2 = x
        s = self.L2Normof2(of2)
        pal1_sources.append(s)

        for k in range(23, 30):
            x = self.vgg[k](x)
        of3 = x
        s = self.L2Normof3(of3)
        pal1_sources.append(s)

        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        of4 = x
        pal1_sources.append(of4)
        # apply extra layers and cache source layer outputs

        for k in range(2):
            x = F.relu(self.extras[k](x), inplace=True)
        of5 = x
        pal1_sources.append(of5)
        for k in range(2, 4):
            x = F.relu(self.extras[k](x), inplace=True)
        of6 = x
        pal1_sources.append(of6)

        conv7 = F.relu(self.fpn_topdown[0](of6), inplace=True)

        x = F.relu(self.fpn_topdown[1](conv7), inplace=True)
        conv6 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[0](of5)), inplace=True)

        x = F.relu(self.fpn_topdown[2](conv6), inplace=True)
        convfc7_2 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[1](of4)), inplace=True)

        x = F.relu(self.fpn_topdown[3](convfc7_2), inplace=True)
        conv5 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[2](of3)), inplace=True)

        x = F.relu(self.fpn_topdown[4](conv5), inplace=True)
        conv4 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[3](of2)), inplace=True)

        x = F.relu(self.fpn_topdown[5](conv4), inplace=True)
        conv3 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[4](of1)), inplace=True)

        ef1 = self.fpn_fem[0](conv3)
        ef1 = self.L2Normef1(ef1)
        ef2 = self.fpn_fem[1](conv4)
        ef2 = self.L2Normef2(ef2)
        ef3 = self.fpn_fem[2](conv5)
        ef3 = self.L2Normef3(ef3)
        ef4 = self.fpn_fem[3](convfc7_2)
        ef5 = self.fpn_fem[4](conv6)
        ef6 = self.fpn_fem[5](conv7)

        pal2_sources = (ef1, ef2, ef3, ef4, ef5, ef6)
        for (x, l, c) in zip(pal1_sources, self.loc_pal1, self.conf_pal1):
            loc_pal1.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal1.append(c(x).permute(0, 2, 3, 1).contiguous())

        for (x, l, c) in zip(pal2_sources, self.loc_pal2, self.conf_pal2):
            loc_pal2.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal2.append(c(x).permute(0, 2, 3, 1).contiguous())

        features_maps = []
        for i in range(len(loc_pal1)):
            feat = []
            feat += [loc_pal1[i].size(1), loc_pal1[i].size(2)]
            features_maps += [feat]

        loc_pal1 = torch.cat([o.view(o.size(0), -1)
                              for o in loc_pal1], 1)
        conf_pal1 = torch.cat([o.view(o.size(0), -1)
                               for o in conf_pal1], 1)

        loc_pal2 = torch.cat([o.view(o.size(0), -1)
                              for o in loc_pal2], 1)
        conf_pal2 = torch.cat([o.view(o.size(0), -1)
                               for o in conf_pal2], 1)

        priorbox = PriorBox(size, features_maps, cfg, pal=1)
        self.priors_pal1 = Variable(priorbox.forward(), volatile=True)

        priorbox = PriorBox(size, features_maps, cfg, pal=2)
        self.priors_pal2 = Variable(priorbox.forward(), volatile=True)

        if self.phase == 'test':
            output = self.detect.forward(
                loc_pal2.view(loc_pal2.size(0), -1, 4),
                self.softmax(conf_pal2.view(conf_pal2.size(0), -1,
                                            self.num_classes)),  # conf preds
                self.priors_pal2.type(type(x.data))
            )

        else:
            output = (
                loc_pal1.view(loc_pal1.size(0), -1, 4),
                conf_pal1.view(conf_pal1.size(0), -1, self.num_classes),
                self.priors_pal1,
                loc_pal2.view(loc_pal2.size(0), -1, 4),
                conf_pal2.view(conf_pal2.size(0), -1, self.num_classes),
                self.priors_pal2)
        # print( f'out.shape = {output.shape}' )
        # exit()
        return output, R

    # during training, the model takes the paired images, and their pseudo GT illumination maps from the Retinex Decom Net
    def forward(self, x):
        if self.enh:
            print('执行增强')
            print(f'输入数据形状{x.shape}')
            # exit()
            inp_img = x.clone()
            illu_fea , illu_map = self.estimator( inp_img )
            inp_img = inp_img * illu_map + inp_img
            inp_enc_encoder1 = self.patch_embed( inp_img )
            
            inp_enc_level1_0 = self.down_1( inp_enc_encoder1 )
            out_enc_level1_0 = self.decoder_level1_0( inp_enc_level1_0 )
            
            inp_enc_level2_0 = self.down_2( out_enc_level1_0 )
            out_enc_level2_0 = self.decoder_level2_0( inp_enc_level2_0 )
            
            inp_enc_level3_0 = self.down_3( out_enc_level2_0 )
            out_enc_level3_0 = self.decoder_level3_0( inp_enc_level3_0 )
            
            out_enc_level3_0 = self.latent( out_enc_level3_0 )
            
            out_enc_level3_0 = self.up3_2( out_enc_level3_0 )
            inp_enc_level2_1 = self.coefficient_3_2[ 0 , : ][ None , : , None , None ] * out_enc_level2_0 + self.coefficient_3_2[ 1 ,
                                                                                                            : ][ None , : , None ,
                                                                                                            None ] * out_enc_level3_0
            inp_enc_level2_1 = self.skip_3_2( inp_enc_level2_1 )  ### conv 1x1
            out_enc_level2_1 = self.decoder_level2_1( inp_enc_level2_1 )
            
            out_enc_level2_1 = self.up2_1( out_enc_level2_1 )
            
            inp_enc_level1_1 = self.coefficient_2_1[ 0 , : ][ None , : , None , None ] * out_enc_level1_0 + self.coefficient_2_1[ 1 ,
                                                                                                            : ][ None , : , None ,
                                                                                                            None ] * out_enc_level2_1
            
            inp_enc_level1_1 = self.skip_1_0( inp_enc_level1_1 )  ### conv 1x1
            out_enc_level1_1 = self.decoder_level1_1( inp_enc_level1_1 )
            out_enc_level1_1 = self.up2_0( out_enc_level1_1 )
            out_fusion_123 = self.sim( out_enc_level1_1 )
            
            img_enh = self.coefficient_1_0[ 0 , : ][ None , : , None , None ] * out_fusion_123 + self.coefficient_1_0[ 1 , : ][ None , : ,
                                                                                             None , None ] * out_enc_level1_1
            
            if self.skip :
                img_enh = self.output( img_enh ) + inp_img
            else :
                img_enh = self.output( img_enh )
                
        if self.det:
            print('执行检测')
            x=img_enh.clone()
            
            size = x.size()[2:]
            pal1_sources = list()
            pal2_sources = list()
            loc_pal1 = list()
            conf_pal1 = list()
            loc_pal2 = list()
            conf_pal2 = list()
    
            for k in range(16):
                x = self.vgg[k](x)
    
            # the following is the rest of the original detection pipeline
            of1 = x
            s = self.L2Normof1(of1)
            pal1_sources.append(s)
            # apply vgg up to fc7
            for k in range(16, 23):
                x = self.vgg[k](x)
            of2 = x
            s = self.L2Normof2(of2)
            pal1_sources.append(s)
    
            for k in range(23, 30):
                x = self.vgg[k](x)
            of3 = x
            s = self.L2Normof3(of3)
            pal1_sources.append(s)
    
            for k in range(30, len(self.vgg)):
                x = self.vgg[k](x)
            of4 = x
            pal1_sources.append(of4)
            # apply extra layers and cache source layer outputs
    
            for k in range(2):
                x = F.relu(self.extras[k](x), inplace=True)
            of5 = x
            pal1_sources.append(of5)
            for k in range(2, 4):
                x = F.relu(self.extras[k](x), inplace=True)
            of6 = x
            pal1_sources.append(of6)
    
            conv7 = F.relu(self.fpn_topdown[0](of6), inplace=True)
    
            x = F.relu(self.fpn_topdown[1](conv7), inplace=True)
            conv6 = F.relu(self._upsample_prod(
                x, self.fpn_latlayer[0](of5)), inplace=True)
    
            x = F.relu(self.fpn_topdown[2](conv6), inplace=True)
            convfc7_2 = F.relu(self._upsample_prod(
                x, self.fpn_latlayer[1](of4)), inplace=True)
    
            x = F.relu(self.fpn_topdown[3](convfc7_2), inplace=True)
            conv5 = F.relu(self._upsample_prod(
                x, self.fpn_latlayer[2](of3)), inplace=True)
    
            x = F.relu(self.fpn_topdown[4](conv5), inplace=True)
            conv4 = F.relu(self._upsample_prod(
                x, self.fpn_latlayer[3](of2)), inplace=True)
    
            x = F.relu(self.fpn_topdown[5](conv4), inplace=True)
            conv3 = F.relu(self._upsample_prod(
                x, self.fpn_latlayer[4](of1)), inplace=True)
    
            ef1 = self.fpn_fem[0](conv3)
            ef1 = self.L2Normef1(ef1)
            ef2 = self.fpn_fem[1](conv4)
            ef2 = self.L2Normef2(ef2)
            ef3 = self.fpn_fem[2](conv5)
            ef3 = self.L2Normef3(ef3)
            ef4 = self.fpn_fem[3](convfc7_2)
            ef5 = self.fpn_fem[4](conv6)
            ef6 = self.fpn_fem[5](conv7)
    
            pal2_sources = (ef1, ef2, ef3, ef4, ef5, ef6)
            for (x, l, c) in zip(pal1_sources, self.loc_pal1, self.conf_pal1):
                loc_pal1.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf_pal1.append(c(x).permute(0, 2, 3, 1).contiguous())
    
            for (x, l, c) in zip(pal2_sources, self.loc_pal2, self.conf_pal2):
                loc_pal2.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf_pal2.append(c(x).permute(0, 2, 3, 1).contiguous())
    
            features_maps = []
            for i in range(len(loc_pal1)):
                feat = []
                feat += [loc_pal1[i].size(1), loc_pal1[i].size(2)]
                features_maps += [feat]
    
            loc_pal1 = torch.cat([o.view(o.size(0), -1)
                                  for o in loc_pal1], 1)
            conf_pal1 = torch.cat([o.view(o.size(0), -1)
                                   for o in conf_pal1], 1)
    
            loc_pal2 = torch.cat([o.view(o.size(0), -1)
                                  for o in loc_pal2], 1)
            conf_pal2 = torch.cat([o.view(o.size(0), -1)
                                   for o in conf_pal2], 1)
    
            priorbox = PriorBox(size, features_maps, cfg, pal=1)
            with torch.no_grad():
                self.priors_pal1 = priorbox.forward()
    
            priorbox = PriorBox(size, features_maps, cfg, pal=2)
            with torch.no_grad():
                self.priors_pal2 = priorbox.forward()
    
            if self.phase == 'test':
                out_det = self.detect.forward(
                    loc_pal2.view(loc_pal2.size(0), -1, 4),
                    self.softmax(conf_pal2.view(conf_pal2.size(0), -1,
                                                self.num_classes)),  # conf preds
                    self.priors_pal2.type(type(x.data))
                )
    
            else:
                out_det = (
                    loc_pal1.view(loc_pal1.size(0), -1, 4),
                    conf_pal1.view(conf_pal1.size(0), -1, self.num_classes),
                    self.priors_pal1,
                    loc_pal2.view(loc_pal2.size(0), -1, 4),
                    conf_pal2.view(conf_pal2.size(0), -1, self.num_classes),
                    self.priors_pal2)
                
        # packing the outputs from the reflectance decoder:
        return out_det,img_enh

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            mdata = torch.load(base_file,
                               map_location=lambda storage, loc: storage)

            epoch = 0
            self.load_state_dict(mdata)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        return epoch
    
    
    def xavier(self, param):
        init.xavier_uniform_(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            m.bias.data.zero_()

        if isinstance(m, nn.ConvTranspose2d):
            self.xavier(m.weight.data)
            if 'bias' in m.state_dict().keys():
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data[...] = 1
            m.bias.data.zero_()


vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
           512, 512, 512, 'M']

extras_cfg = [256, 'S', 512, 128, 'S', 256]

fem_cfg = [256, 512, 512, 1024, 512, 256]


def fem_module(cfg):
    topdown_layers = []
    lat_layers = []
    fem_layers = []

    topdown_layers += [nn.Conv2d(cfg[-1], cfg[-1],
                                 kernel_size=1, stride=1, padding=0)]
    for k, v in enumerate(cfg):
        fem_layers += [FEM(v)]
        cur_channel = cfg[len(cfg) - 1 - k]
        if len(cfg) - 1 - k > 0:
            last_channel = cfg[len(cfg) - 2 - k]
            topdown_layers += [nn.Conv2d(cur_channel, last_channel,
                                         kernel_size=1, stride=1, padding=0)]
            lat_layers += [nn.Conv2d(last_channel, last_channel,
                                     kernel_size=1, stride=1, padding=0)]
    return (topdown_layers, lat_layers, fem_layers)


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [14, 21, 28, -2]

    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels,
                                 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels,
                                  num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)


def build_net_dark(phase, num_classes=2):
    base = vgg(vgg_cfg, 3)
    extras = add_extras(extras_cfg, 1024)
    head1 = multibox(base, extras, num_classes)
    head2 = multibox(base, extras, num_classes)
    fem = fem_module(fem_cfg)
    return DSFD(phase, base, extras, fem, head1, head2, num_classes)


class DistillKL(nn.Module):
    """KL divergence for distillation"""
    # 知识蒸馏模块，处理KL散度
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        # y_s学生模型的输出，y_t 教师模型的输出
        p_s = F.log_softmax(y_s / self.T, dim=1)#对数概率分布
        p_t = F.softmax(y_t / self.T, dim=1)#概率分布
        # 计算KL散度
        # size_average不使用平均损失，而是返回总损失，(self.T ** 2)补偿温度缩放，/ y_s.shape[0]计算平均损失
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss

