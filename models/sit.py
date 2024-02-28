# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Your name
# @Last Modified time: 2022-02-22 11:35:57
#
# Created on Fri Oct 01 2021
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2021 MeTrICS Lab
#

'''
This file contains our implementation of the ViT model: https://arxiv.org/abs/2010.11929
'''

import torch
from torch import nn
import numpy as np

from einops import repeat
from einops.layers.torch import Rearrange

from vit_pytorch.vit import Transformer

from timm.models.layers import trunc_normal_

from models.pos_emb import get_1d_sincos_pos_embed_from_grid

class SiT(nn.Module):
    def __init__(self, *,
                        dim, 
                        depth,
                        heads,
                        pool = 'cls', 
                        num_patches = 20,
                        num_classes= 1,
                        num_channels =4,
                        num_vertices = 2145,
                        dim_head = 64,
                        dropout = 0.,
                        emb_dropout = 0.,
                        bottleneck_dropout = 0.,
                        mlp_ratio = 4,
                        use_pe = True,
                        use_confounds = False,
                        use_bottleneck= False, 
                        weights_init = False, 
                        use_class_token = True,
                        trainable_pos_emb = True, 
                        no_class_token_emb = False, 
                        ):

        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        patch_dim = num_channels * num_vertices

        # inputs has size = b * c * n * v
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Linear(patch_dim, dim),
        )

        if use_bottleneck:
            print('using bottleneck')
            self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Dropout(bottleneck_dropout),
            nn.Linear(patch_dim,1024),
            nn.Dropout(bottleneck_dropout),
            nn.Linear(1024, dim),
        )

        if use_confounds:
            self.proj_confound = nn.Sequential(
                                    nn.BatchNorm1d(1),
                                    nn.Linear(1,dim))
        
        self.dropout = nn.Dropout(emb_dropout)

        self.use_pe = use_pe
        self.use_confounds = use_confounds
        self.num_patches = num_patches
        self.encoding_dim = dim
        self.use_class_token = use_class_token
        self.no_class_token_emb = no_class_token_emb
        self.num_prefix_tokens = 1 if use_class_token else 0

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_ratio*dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        ### positional embeddings
        if self.use_pe == 'trainable':
            print('Using trainable positional embeddings')
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim) * .02, requires_grad=trainable_pos_emb) if no_class_token_emb \
                else nn.Parameter(torch.randn(1, num_patches+self.num_prefix_tokens, dim), requires_grad=trainable_pos_emb)
        elif self.use_pe == 'sin-cos':
            print('Using Sin-Cos positional embeddings')
            self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, dim) * .02, requires_grad=False) if no_class_token_emb \
                else nn.Parameter(torch.zeros(1, num_patches+self.num_prefix_tokens, dim), requires_grad=False)
            print(self.pos_embedding.shape)
            self._init_pos_em()
        ####
        
        ### weight init
        if weights_init: # apply the same weight init as timm's github
            print('Using initialisation from Timms repo')
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if use_class_token else None
            self._init_weights_class()
            self._init_weights()

        else: # apply to weight init from vit_pytorch
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) if use_class_token else None
        ###
        

    def _init_weights_class(self,):

        trunc_normal_(self.pos_embedding, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6) 

        # initialize nn.Linear
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
    
    def _init_pos_em(self,):
        num_patches =  self.num_patches if self.no_class_token_emb else self.num_patches +1
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embedding.shape[-1], np.arange(num_patches, dtype=np.float32))
        self.pos_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def add_pos_embed(self,x):
        
        b, n, _ = x.shape
        
        if self.no_class_token_emb:
            x = x + self.pos_embedding
            if self.use_class_token:
                cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
                x = torch.cat((cls_tokens, x), dim=1)      
        else:
            if self.use_class_token:
                cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
                x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embedding
        return x 
        

    def forward(self, img, confounds=None):
        
        x = self.to_patch_embedding(img)

        b, n, _ = x.shape
        
        x = self.add_pos_embed(x)

        if self.use_confounds and (confounds is not None):
            confounds = self.proj_confound(confounds.view(-1,1))
            confounds = repeat(confounds, 'b d -> b n d', n=n+1) if self.use_class_token else repeat(confounds, 'b d -> b n d', n=n)
            x += confounds
                    
        x = self.dropout(x)

        x = self.transformer(x)


        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        return self.mlp_head(x)

if __name__ == '__main__':

    model = SiT(dim = 192,
                depth = 12,
                heads= 3,
                weights_init=True)