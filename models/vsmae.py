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

class vsMAE(nn.Module):
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
        no_pos_emb_class_token_decoder = True, 
        mask = True,  
        path_to_template = '',
        path_to_workdir = '',
        sampling = 'msm',
        sub_ico = 3, 
        masking_type = 'tubelet',
        nbr_frames = 1,
        loss = 'mse',
        mask_loss = True, 

    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.encoder = encoder
        self.num_patches_encoder, encoder_dim = encoder.num_patches, encoder.encoding_dim
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
        self.masking_type = masking_type
        self.use_confounds = self.encoder.use_confounds
        self.nbr_frames = nbr_frames
        self.loss = loss
        self.mask_loss = mask_loss

        print('loss: {}'.format(self.loss))
        

        # decoder parameters
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim,bias=True) if encoder_dim != decoder_dim else nn.Identity()
        self.decoder = Transformer(dim = decoder_dim,
                                    depth = decoder_depth, 
                                    heads = decoder_heads, 
                                    dim_head = decoder_dim_head, 
                                    mlp_dim = decoder_dim * 4)

        self.decoder_pos_emb = nn.Parameter(torch.zeros(1, self.num_patches_encoder, decoder_dim) * .02, requires_grad=False) if no_pos_emb_class_token_decoder \
                else nn.Parameter(torch.zeros(1, self.num_patches_encoder+1, decoder_dim), requires_grad=False)
        self._init_pos_em()

        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.decoder_norm = nn.LayerNorm(decoder_dim)


        if weights_init:
            self.mask_token = nn.Parameter(torch.zeros(decoder_dim))
            self.init_weights()
        else:
            self.mask_token = nn.Parameter(torch.randn(decoder_dim))

        if self.masking and self.dataset == 'dHCP' and self.configuration == 'template': # for dHCP
            self.mask_cortex =np.array(nb.load('{}/week-40_hemi-left_space-dhcpSym_dens-40k_desc-medialwallsymm_mask.shape.gii'.format(path_to_template)).agg_data())
        elif self.masking and (self.dataset == 'HCP' or self.dataset =='UKB'): # for UKB
            self.mask_cortex = np.array(nb.load('{}/L.atlasroi.ico6_fs_LR.shape.gii'.format(path_to_template)).agg_data())
        #elif self.masking and self.dataset == 'HCP': # for UKB
        #    self.mask = np.array(nb.load('{}/L.atlasroi.40k_fs_LR.shape.gii'.format(path_to_template)).agg_data())

        self.triangle_indices = pd.read_csv('{}/patch_extraction/{}/triangle_indices_ico_6_sub_ico_{}.csv'.format(path_to_workdir,sampling,sub_ico))


    def init_weights(self,):
        nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
    
    def _init_pos_em(self,):
        num_patches =  self.num_patches_encoder if self.no_pos_emb_class_token_decoder else self.num_patches_encoder +1
        dec_pos_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_pos_emb.shape[-1], np.arange(num_patches, dtype=np.float32))
        self.decoder_pos_emb.data.copy_(torch.from_numpy(dec_pos_embed).float().unsqueeze(0))
            
    def random_masking(self, x, mask_ratio, masking_type): #GOOD
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        B, N, V = x.shape  # batch, length, dim

        if masking_type == 'tubelet':

            L = N // self.nbr_frames 

            len_to_keep = int(L * (1 - mask_ratio))
            
            noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]
            
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            #import pdb;pdb.set_trace()
            # keep the first subset
            ids_tokens_not_masked = ids_shuffle[:, :len_to_keep]
            ids_tokens_masked = ids_shuffle[:,len_to_keep:]

            x_unshaped = rearrange(x, 'b (t l) v -> b t l v', b=B, t=self.nbr_frames,l=L,v=V)
            #x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, V))
            x_not_masked_unshaped = torch.gather(x_unshaped, dim=2, index=ids_tokens_not_masked.unsqueeze(-1).unsqueeze(1).repeat(1, self.nbr_frames ,1, V))
            x_not_masked = rearrange(x_not_masked_unshaped, 'b t l v -> b (t l) v')

            # generate the binary mask: 0 is kept/not_masked, 1 is remove/mask
            mask_binary = torch.ones([B, L], device=x.device)
            mask_binary[:, :len_to_keep] = 0
            # unshuffle to get the binary mask
            mask_binary = torch.gather(mask_binary, dim=1, index=ids_restore)
            mask_binary = repeat(mask_binary, 'b l -> b t l', t=self.nbr_frames)
            mask_binary = rearrange(mask_binary, 'b t l -> b (t l)')
            
        elif masking_type == 'random':
            
            len_to_keep = int(N * round((1 - mask_ratio),2))
            
            noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            
            # keep the first subset
            ids_tokens_not_masked = ids_shuffle[:, :len_to_keep]
            ids_tokens_masked = ids_shuffle[:,len_to_keep:]
            
            # no need to unshape the x (input of shape b, (t l), c)
            x_not_masked = torch.gather(x, dim=1, index=ids_tokens_not_masked.unsqueeze(-1).repeat(1 ,1, V)) 
            
            # generate the binary mask: 0 is kept/not_masked, 1 is remove/mask
            mask_binary = torch.ones([B, N], device=x.device)
            mask_binary[:, :len_to_keep] = 0
            # unshuffle to get the binary mask
            mask_binary = torch.gather(mask_binary, dim=1, index=ids_restore)
            #import pdb;pdb.set_trace()

        return x_not_masked, mask_binary, ids_restore, ids_tokens_not_masked, ids_tokens_masked
    
    def forward(self, imgs, confounds=None):
        B, C, N, V = imgs.shape
        L = N //self.nbr_frames
        #import pdb;pdb.set_trace()

        latent, mask_binary, ids_restore, ids_tokens_not_masked, ids_tokens_masked = self.forward_encoder(imgs,confounds)
        #import pdb;pdb.set_trace()
        pred = self.forward_decoder(latent, ids_restore, self.masking_type)  # [N, L, p*p*3] # [N,L,v*C]
        #import pdb;pdb.set_trace()
        loss = self.forward_loss(imgs, pred, mask_binary)

        if self.masking_type == 'tubelet':

            pred_unshaped = rearrange(pred, 'b (t l) v -> b t l v', b=B, t=self.nbr_frames,l=L,v=V)
            pred_tokens_not_masked = torch.gather(pred_unshaped, dim=2, index=ids_tokens_not_masked.unsqueeze(-1).unsqueeze(1).repeat(1,self.nbr_frames,1,pred.shape[-1]))
            pred_tokens_masked = torch.gather(pred_unshaped, dim=2, index=ids_tokens_masked.unsqueeze(-1).unsqueeze(1).repeat(1,self.nbr_frames,1,pred.shape[-1]))
            
        elif self.masking_type == 'random':

            pred_tokens_not_masked = torch.gather(pred, dim=1, index=ids_tokens_not_masked.unsqueeze(-1).repeat(1,1,pred.shape[-1]))
            pred_tokens_masked = torch.gather(pred, dim=1, index=ids_tokens_masked.unsqueeze(-1).repeat(1,1,pred.shape[-1]))
            
        pred_tokens_not_masked  = pred_tokens_not_masked.detach()
        pred_tokens_masked = pred_tokens_masked.detach()

        prediction_pixels_masked_, prediction_pixels_not_masked_ = self.save_reconstruction(pred_tokens_not_masked, pred_tokens_masked, ids_tokens_not_masked, ids_tokens_masked)

        return loss, prediction_pixels_masked_, prediction_pixels_not_masked_, ids_tokens_masked, ids_tokens_not_masked

    def forward_encoder(self, img, confounds=None):
        device = img.device

        # get patches
        patches = self.to_patch(img) ##ok 
        batch, num_patches, *_ = patches.shape

        assert (num_patches == self.num_patches_encoder)

        # patch to encoder tokens and add positions
        #import pdb;pdb.set_trace()
        x = self.patch_to_emb(patches)
        #import pdb;pdb.set_trace()
        
        if self.encoder.no_class_token_emb:
            x = x + self.encoder.pos_embedding[:,:num_patches,:]  #can be set to fixed in the encoder 
        else:
            x = x + self.encoder.pos_embedding[:,self.encoder.num_prefix_tokens:(num_patches+self.encoder.num_prefix_tokens),:] #use use class toekn: 1-> n+1 else, 0->n
        
        #import pdb;pdb.set_trace()
        # masking: length -> length * mask_ratio
        x, mask_binary, ids_restore, ids_tokens_not_masked, ids_tokens_masked = self.random_masking(x, self.masking_ratio,self.masking_type)
        #import pdb;pdb.set_trace()
        # append cls token
        if self.encoder.use_class_token:
            cls_token = self.encoder.cls_token if self.encoder.no_class_token_emb else self.encoder.cls_token + self.encoder.pos_embedding[:, :self.encoder.num_prefix_tokens, :] 
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_confounds and (confounds is not None):
            confounds = self.encoder.proj_confound(confounds.view(-1,1))
            confounds = repeat(confounds, 'b d -> b n d', n=num_patches+1) if (not self.encoder.no_class_token_emb) else repeat(confounds, 'b d -> b n d', n=num_patches)
            x += confounds
        
        #import pdb;pdb.set_trace()
        # attend with vision transformer
        x = self.encoder.transformer(x)
        
        x = self.norm(x)
        
        return x, mask_binary, ids_restore, ids_tokens_not_masked, ids_tokens_masked
    
    def forward_decoder(self, x, ids_restore, masking_type):
        # embed tokens
        x = self.enc_to_dec(x)

        if masking_type == 'tubelet':

            B, n_unmasked, V = x.shape 
            L = n_unmasked // self.nbr_frames 

            x_unshaped = rearrange(x, 'b (t l) v -> b t l v', b=B, t=self.nbr_frames,l=L,v=V)
            
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x_unshaped.shape[0], self.nbr_frames, ids_restore.shape[1] - L, 1) ## I have removed the +1
            #import pdb;pdb.set_trace()
            x_ = torch.cat([x_unshaped, mask_tokens], dim=2) if (not self.encoder.use_class_token) else torch.cat([x[:,:, 1:, :], mask_tokens], dim=2) # no cls token
            #import pdb;pdb.set_trace()
            x_ = torch.gather(x_, dim=2, index=ids_restore.unsqueeze(-1).unsqueeze(1).repeat(1,self.nbr_frames ,1, V))  # unshuffle
            #import pdb;pdb.set_trace()
            x_dec = rearrange(x_, 'b t l v -> b (t l) v')
            #import pdb;pdb.set_trace()
    
        elif masking_type == 'random':
            B, n_unmasked, V = x.shape 
            #import pdb;pdb.set_trace()
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - n_unmasked, 1) ## I have removed the +1
            #import pdb;pdb.set_trace()
            x_ = torch.cat([x, mask_tokens], dim=1) if (not self.encoder.use_class_token) else torch.cat([x[:, 1:, :], mask_tokens], dim=1) # no cls token
            #import pdb;pdb.set_trace()
            x_dec = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1 ,1, V))  # unshuffle
            #import pdb;pdb.set_trace()



        # add pos embed and classification token ### FOR NOW, NOT USING CLASSIFICATION TOKEN AT ALL!! 
        if self.no_pos_emb_class_token_decoder:
            x_dec = x_dec + self.decoder_pos_emb
            #x_dec = torch.cat([x[:, :1, :], x_dec], dim=1)  # append cls token
        else:
            #x_dec = torch.cat([x[:, :1, :], x_dec], dim=1)  # append cls token
            x_dec = x_dec + self.decoder_pos_emb    
        #import pdb;pdb.set_trace()        
        x_dec = self.decoder(x_dec)
        x_dec = self.decoder_norm(x_dec)
        
        #predictor projection
        pred = self.to_pixels(x_dec)

        # remove cls token ### TO ADD WHEN CONSIDERING CLASS TOKEN
        #pred = pred[:, 1:, :]
        #import pdb;pdb.set_trace()
        return pred

    def forward_loss(self, imgs, pred, mask_binary):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        #import pdb;pdb.set_trace()
        #target = self.to_patch(imgs)
        if imgs.shape[0]==1:
            target = imgs.squeeze(1)
        else:
            target = imgs.squeeze()
        #import pdb;pdb.set_trace()

        ## compute loss for all patches #check the size
        if self.loss == 'mse':
            loss = (pred - target) ** 2
        elif self.loss == 'l1':
            loss = torch.abs(pred-target)

        #import pdb;pdb.set_trace()
        if self.mask_loss:
            mask_ext = mask_binary.unsqueeze(-1).repeat(1,1,target.shape[2])
            return loss[mask_ext==1].mean() # mean loss on removed patches
        else:
            return loss.mean()
    

    
    def save_reconstruction(self,pred_tokens_not_masked, pred_tokens_masked, ids_tokens_not_masked,ids_tokens_masked ):

        
        #### ADDED #### Mask the patches with the cut
        mask_cortex = torch.Tensor(self.mask_cortex)
        mask_cortex = mask_cortex.to(pred_tokens_not_masked.device)

        prediction_pixels_tokens_masked = torch.zeros_like(pred_tokens_masked)
        prediction_pixels_tokens_not_masked = torch.zeros_like(pred_tokens_not_masked)

        if self.masking_type == 'tubelet':

            for f in range(self.nbr_frames):

                prediction_pixels_masked = rearrange(pred_tokens_masked[:,f,:,:], 'b n (v c) -> b n c v',
                                                b = pred_tokens_masked.shape[0],
                                                n = pred_tokens_masked.shape[2], \
                                                c = self.num_channels,
                                                v = self.num_vertices_per_channel)
                
                prediction_pixels_masked_ = torch.zeros_like(prediction_pixels_masked)

                prediction_pixels_not_masked = rearrange(pred_tokens_not_masked[:,f,:,:],
                                                        'b n (v c) -> b n c v', 
                                                        b = pred_tokens_not_masked.shape[0],
                                                        n = pred_tokens_not_masked.shape[2], 
                                                        c = self.num_channels,
                                                        v = self.num_vertices_per_channel)

                prediction_pixels_not_masked_ = torch.zeros_like(prediction_pixels_not_masked)

                for i in range(pred_tokens_masked.shape[0]):
                    for j,k in enumerate(ids_tokens_masked[i]):
                        indices_to_extract = torch.Tensor(self.triangle_indices[str(k.cpu().numpy())].values).long()
                        indices_to_extract = indices_to_extract.to(pred_tokens_masked.device)
                        prediction_pixels_masked_[i,j,:,:] = prediction_pixels_masked[i,j,:,:] * mask_cortex[indices_to_extract]
                prediction_pixels_masked_ = rearrange(prediction_pixels_masked_, 'b n c v -> b n (v c)')


                for i in range(pred_tokens_not_masked.shape[0]):
                    for j,k in enumerate(ids_tokens_not_masked[i]):
                        indices_to_extract = torch.Tensor(self.triangle_indices[str(k.cpu().numpy())].values).long()
                        indices_to_extract = indices_to_extract.to(pred_tokens_not_masked.device)
                        prediction_pixels_not_masked_[i,j,:,:] = prediction_pixels_not_masked[i,j,:,:] * mask_cortex[indices_to_extract]
                prediction_pixels_not_masked_ = rearrange(prediction_pixels_not_masked_, 'b n c v -> b n (v c)')

                prediction_pixels_tokens_masked[:,f,:,:] = prediction_pixels_masked_
                prediction_pixels_tokens_not_masked[:,f,:,:] = prediction_pixels_not_masked_
        
        elif self.masking_type == 'random':

            for f in range(self.nbr_frames):

                ### masked tokens

                n_patches_masked = (pred_tokens_masked.shape[1] // self.nbr_frames)

                prediction_pixels_masked = rearrange(pred_tokens_masked[:,f*n_patches_masked:(f+1)*n_patches_masked,:], 'b n (v c) -> b n c v',
                                                b = pred_tokens_masked.shape[0],
                                                n = n_patches_masked, 
                                                c = self.num_channels,
                                                v = self.num_vertices_per_channel)

                prediction_pixels_masked_ = torch.zeros_like(prediction_pixels_masked)

                ### un masked tokens
                n_patches_non_masked = (pred_tokens_not_masked.shape[1] // self.nbr_frames)

                prediction_pixels_not_masked = rearrange(pred_tokens_not_masked[:,f*n_patches_non_masked:(f+1)*n_patches_non_masked,:],
                                                        'b n (v c) -> b n c v', 
                                                        b = pred_tokens_not_masked.shape[0],
                                                        n = n_patches_non_masked,
                                                        c = self.num_channels,
                                                        v = self.num_vertices_per_channel)

                prediction_pixels_not_masked_ = torch.zeros_like(prediction_pixels_not_masked)

                n_total = n_patches_masked + n_patches_non_masked

                for i in range(pred_tokens_masked.shape[0]):
                    for j,k in enumerate((ids_tokens_masked%n_total)[i,f*n_patches_masked:(f+1)*n_patches_masked]):
                        indices_to_extract = torch.Tensor(self.triangle_indices[str(k.cpu().numpy())].values).long()
                        indices_to_extract = indices_to_extract.to(pred_tokens_masked.device)
                        prediction_pixels_masked_[i,j,:,:] = prediction_pixels_masked[i,j,:,:] * mask_cortex[indices_to_extract]
                prediction_pixels_masked_ = rearrange(prediction_pixels_masked_, 'b n c v -> b n (v c)')

                #import pdb;pdb.set_trace()


                for i in range(pred_tokens_not_masked.shape[0]):
                    for j,k in enumerate((ids_tokens_not_masked%n_total)[i,f*n_patches_non_masked:(f+1)*n_patches_non_masked]):
                        indices_to_extract = torch.Tensor(self.triangle_indices[str(k.cpu().numpy())].values).long()
                        indices_to_extract = indices_to_extract.to(pred_tokens_not_masked.device)
                        prediction_pixels_not_masked_[i,j,:,:] = prediction_pixels_not_masked[i,j,:,:] * mask_cortex[indices_to_extract]
                prediction_pixels_not_masked_ = rearrange(prediction_pixels_not_masked_, 'b n c v -> b n (v c)')


                prediction_pixels_tokens_masked[:,f*n_patches_masked:(f+1)*n_patches_masked,:] = prediction_pixels_masked_
                prediction_pixels_tokens_not_masked[:,f*n_patches_non_masked:(f+1)*n_patches_non_masked,:] = prediction_pixels_not_masked_

            B = prediction_pixels_tokens_not_masked.shape[0]
            V = prediction_pixels_tokens_not_masked.shape[-1]
            
            prediction_pixels_tokens_masked = rearrange(prediction_pixels_tokens_masked, 'b (t l) v -> b t l v', b=B, t=self.nbr_frames,l=n_patches_masked,v=V)
            prediction_pixels_tokens_not_masked = rearrange(prediction_pixels_tokens_not_masked, 'b (t l) v -> b t l v', b=B, t=self.nbr_frames,l=n_patches_non_masked,v=V)

        return prediction_pixels_tokens_masked, prediction_pixels_tokens_not_masked
    
    
    