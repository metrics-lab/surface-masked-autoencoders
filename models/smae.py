import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
import numpy as np
import nibabel as nb
import pandas as pd
from timm.models.layers import trunc_normal_
from vit_pytorch.vit import Transformer

from models.pos_emb import get_1d_sincos_pos_embed_from_grid


'''
Code adapted to surfaces from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mae.py
- no classification token in the encoder, just unmask patches
- trainable positional embedding
'''

class sMAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64,
        dataset = 'dHCP',
        configuration = 'template',
        num_channels =1, 
        weights_init = False,
        use_class_token_dec = False,
        no_pos_emb_class_token_decoder = False, 
        mask = True,  
        path_to_template = '',
        path_to_workdir = '',
        sampling = 'msm',
        sub_ico = 3, 

    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.encoder = encoder
        self.num_patches, encoder_dim = encoder.num_patches, encoder.encoding_dim
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1] # channels*num_vertices
        self.norm = encoder.mlp_head[0]
        self.dataset = dataset
        self.num_channels = num_channels
        self.configuration = configuration
        self.num_vertices_per_channel = pixel_values_per_patch // self.num_channels
        self.use_class_token_dec = use_class_token_dec
        self.no_pos_emb_class_token_decoder = no_pos_emb_class_token_decoder
        self.masking = mask
        self.use_confounds = self.encoder.use_confounds
        

        # decoder parameters
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim,bias=True) if encoder_dim != decoder_dim else nn.Identity()
        self.decoder = Transformer(dim = decoder_dim,
                                    depth = decoder_depth, 
                                    heads = decoder_heads, 
                                    dim_head = decoder_dim_head, 
                                    mlp_dim = decoder_dim * 4)

        self.decoder_pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim) * .02, requires_grad=False) if no_pos_emb_class_token_decoder \
                else nn.Parameter(torch.zeros(1, self.num_patches+1, decoder_dim), requires_grad=False)
        self._init_pos_em()

        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.decoder_norm = nn.LayerNorm(decoder_dim)


        if weights_init:
            self.mask_token = nn.Parameter(torch.zeros(decoder_dim))
            self.init_weights()
        else:
            self.mask_token = nn.Parameter(torch.randn(decoder_dim))

        if self.masking and self.dataset == 'dHCP' and self.configuration == 'template': # for dHCP
            self.mask =np.array(nb.load('{}/week-40_hemi-left_space-dhcpSym_dens-40k_desc-medialwallsymm_mask.shape.gii'.format(path_to_template)).agg_data())
        elif self.masking and self.dataset == 'UKB': # for UKB
            self.mask = np.array(nb.load('{}/L.atlasroi.ico6_fs_LR.shape.gii'.format(path_to_template)).agg_data())
        elif self.masking and self.dataset == 'HCP': # for UKB
            self.mask = np.array(nb.load('{}/L.atlasroi.40k_fs_LR.shape.gii'.format(path_to_template)).agg_data())

        self.triangle_indices = pd.read_csv('{}/patch_extraction/{}/triangle_indices_ico_6_sub_ico_{}.csv'.format(path_to_workdir,sampling,sub_ico))

        self.cls_token_dec = nn.Parameter(torch.randn(1, 1, decoder_dim)) if use_class_token_dec else None

    def init_weights(self,):
        nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
    
    def _init_pos_em(self,):
        num_patches =  self.num_patches if self.no_pos_emb_class_token_decoder else self.num_patches +1
        dec_pos_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_pos_emb.shape[-1], np.arange(num_patches, dtype=np.float32))
        self.decoder_pos_emb.data.copy_(torch.from_numpy(dec_pos_embed).float().unsqueeze(0))
            
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_not_keep = ids_shuffle[:,len_keep:]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep, ids_not_keep

    def forward_encoder(self, img, confounds=None):
        device = img.device

        # get patches
        patches = self.to_patch(img) ##ok 
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        x = self.patch_to_emb(patches)
        
        if self.encoder.no_class_token_emb:
            x = x + self.encoder.pos_embedding[:,:num_patches,:]  #can be set to fixed in the encoder 
        else:
            x = x + self.encoder.pos_embedding[:,self.encoder.num_prefix_tokens:(num_patches+self.encoder.num_prefix_tokens),:] #use use class toekn: 1-> n+1 else, 0->n
        
        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep, ids_not_keep = self.random_masking(x, self.masking_ratio)
        # append cls token
        if self.encoder.use_class_token:
            cls_token = self.encoder.cls_token if self.encoder.no_class_token_emb else self.encoder.cls_token + self.encoder.pos_embedding[:, :self.encoder.num_prefix_tokens, :] 
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_confounds and (confounds is not None):
            confounds = self.encoder.proj_confound(confounds.view(-1,1))
            confounds = repeat(confounds, 'b d -> b n d', n=num_patches+1) if (not self.encoder.no_class_token_emb) else repeat(confounds, 'b d -> b n d', n=num_patches)
            x += confounds
        
        # attend with vision transformer
        x = self.encoder.transformer(x)
        
        x = self.norm(x)
        
        return x, mask, ids_restore, ids_keep, ids_not_keep
    
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.enc_to_dec(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1) ### !!!!maybe complete random
        if self.encoder.use_class_token:
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # remove temp cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        else:
            #import pdb;pdb.set_trace()
            x_ = torch.cat([x, mask_tokens], dim=1) 
            #import pdb;pdb.set_trace()
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        # add pos embed
        '''
        if self.no_class_emb_decoder:
            x_ = x_ + self.decoder_pos_emb
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        else:
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
            x = x + self.decoder_pos_emb    
        '''
        if self.use_class_token_dec and self.encoder.use_class_token:
            if self.no_pos_emb_class_token_decoder:
                ### TO REVIEW
                x_ = x_ + self.decoder_pos_emb
                x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
            else:
                x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
                x = x + self.decoder_pos_emb    
        else:
            x = x_ + self.decoder_pos_emb    
                
        x = self.decoder(x)
        x = self.decoder_norm(x)
        
        #predictor projection
        x = self.to_pixels(x)
        
        # remove cls token
        if self.use_class_token_dec:
            x = x[:, 1:, :]

        return x
    
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.to_patch(imgs)

        ## compute loss for all patches #check the size
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss

    
    def forward(self, imgs, confounds=None):
        latent, mask, ids_restore, ids_keep, ids_not_keep = self.forward_encoder(imgs,confounds)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3] # [N,L,v*C]
        loss = self.forward_loss(imgs, pred, mask)

        pred_unmasked = torch.gather(pred, dim=1, index=ids_keep.unsqueeze(-1).repeat(1,1,pred.shape[-1]))
        pred_masked = torch.gather(pred, dim=1, index=ids_not_keep.unsqueeze(-1).repeat(1,1,pred.shape[-1]))

        prediction_pixels_mask = pred_masked.detach()
        prediction_pixels_unmasked  = pred_unmasked.detach()

        prediction_pixels_mask_, prediction_pixels_unmasked_ = self.save_reconstruction(prediction_pixels_unmasked, prediction_pixels_mask, ids_keep,ids_not_keep )

        return loss, prediction_pixels_mask_, prediction_pixels_unmasked_, ids_not_keep, ids_keep
    
    def save_reconstruction(self,pred_unmasked, pred_masked, ids_keep,ids_not_keep ):
        
        #### ADDED #### Mask the patches with the cut
        mask = torch.Tensor(self.mask)
        mask = mask.to(pred_unmasked.device)

        prediction_pixels_mask = rearrange(pred_masked, 'b n (v c) -> b n c v', b =pred_masked.shape[0], n=pred_masked.shape[1], c =self.num_channels,v=self.num_vertices_per_channel)

        prediction_pixels_mask_ = torch.zeros_like(prediction_pixels_mask)

        prediction_pixels_unmasked = rearrange(pred_unmasked, 'b n (v c) -> b n c v', b =pred_unmasked.shape[0], n=pred_unmasked.shape[1], c =self.num_channels,v=self.num_vertices_per_channel)

        prediction_pixels_unmasked_ = torch.zeros_like(prediction_pixels_unmasked)

        for i in range(pred_masked.shape[0]):
            for j,k in enumerate(ids_not_keep[i]):
                indices_to_extract = torch.Tensor(self.triangle_indices[str(k.cpu().numpy())].values).long()
                indices_to_extract = indices_to_extract.to(pred_unmasked.device)
                prediction_pixels_mask_[i,j,:,:] = prediction_pixels_mask[i,j,:,:] * mask[indices_to_extract]
        prediction_pixels_mask_ = rearrange(prediction_pixels_mask_, 'b n c v -> b n (v c)')


        for i in range(pred_unmasked.shape[0]):
            for j,k in enumerate(ids_keep[i]):
                indices_to_extract = torch.Tensor(self.triangle_indices[str(k.cpu().numpy())].values).long()
                indices_to_extract = indices_to_extract.to(pred_unmasked.device)
                prediction_pixels_unmasked_[i,j,:,:] = prediction_pixels_unmasked[i,j,:,:] * mask[indices_to_extract]
        prediction_pixels_unmasked_ = rearrange(prediction_pixels_unmasked, 'b n c v -> b n (v c)')

        return prediction_pixels_mask_, prediction_pixels_unmasked_
    
    
    
