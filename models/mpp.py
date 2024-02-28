# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Your name
# @Last Modified time: 2022-02-14 17:50:22
#
# Created on Mon Oct 18 2021
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2021 MeTrICS Lab
#


import math
from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F 

from einops import rearrange, repeat


def get_mask_from_prob(inputs, prob):
    '''
    This function creates a mask on the sequence of tokens, per sample
    Based on the probability of masking. 
    return: a boolean mask of the shape of the inputs. 
    '''
    batch, seq_len, _, device = *inputs.shape, inputs.device
    max_masked = math.ceil(prob * seq_len)

    rand = torch.rand((batch, seq_len), device=device)
    _, sampled_indices = rand.topk(max_masked, dim=-1)

    new_mask = torch.zeros((batch, seq_len), device=device)
    new_mask.scatter_(1, sampled_indices, 1)
    return new_mask.bool()

def prob_mask_like(inputs, prob):
    batch, seq_length, _ = inputs.shape
    return torch.zeros((batch, seq_length)).float().uniform_(0, 1) < prob


class masked_patch_pretraining(nn.Module):

    def __init__(
        self,
        transformer,
        dim_in,
        dim_out,
        device,
        mask_prob=0.15,
        replace_prob=0.5,
        swap_prob=0.3,
        channels=4,
        num_vertices=561,):

        super().__init__()
        self.transformer = transformer

        self.dim_out = dim_out
        self.dim_in = dim_in

        self.to_original = nn.Linear(dim_in,dim_out)
        self.to_original.to(device)

        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.swap_prob = swap_prob

        # token ids
        self.mask_token = nn.Parameter(torch.randn(1, 1, channels * num_vertices))


    def forward(self, batch, **kwargs):

        transformer = self.transformer

        # clone original image for loss

        batch = rearrange(batch,
                        'b c n v  -> b n (v c)')

        corrupted_sequence = get_mask_from_prob(batch, self.mask_prob)
        #print('probability of tokens to be in the corrupted sequence: {}'.format(len(corrupted_sequence[corrupted_sequence==True])/(corrupted_sequence.shape[0]*corrupted_sequence.shape[1])))

        corrupted_batch = batch.clone().detach()

        #randomly swap patches in the sequence
        if self.swap_prob > 0:
            random_patch_sampling_prob = self.swap_prob / (
                1 - self.replace_prob)
                
            random_patch_prob = prob_mask_like(batch,
                                               random_patch_sampling_prob).to(corrupted_sequence.device)
            
            bool_random_patch_prob = corrupted_sequence * (random_patch_prob == True)
            #print('probability of swapping tokens in the corrupted sequence: {}'.format(len(bool_random_patch_prob[bool_random_patch_prob==True])/(bool_random_patch_prob.shape[0]*bool_random_patch_prob.shape[1])))
            
            random_patches = torch.randint(0,
                                           batch.shape[1],
                                           (batch.shape[0], batch.shape[1]),
                                           device=batch.device)
            #shuffle entierely masked_batch                               
            randomized_input = corrupted_batch[
                torch.arange(corrupted_batch.shape[0]).unsqueeze(-1),
                random_patches]
            #getting randomised patch only for the indices bool_random_patch_prob
            corrupted_batch[bool_random_patch_prob] = randomized_input[bool_random_patch_prob]
        tokens_to_mask = prob_mask_like(batch, self.replace_prob).to(corrupted_sequence.device)

        bool_mask_replace = (corrupted_sequence * tokens_to_mask) == True
        #print('probability of `masking` tokens in the corrupted sequence: {}'.format(len(bool_mask_replace[bool_mask_replace==True])/(bool_mask_replace.shape[0]*bool_mask_replace.shape[1])))
        corrupted_batch[bool_mask_replace] = self.mask_token.to(corrupted_sequence.device)

        # linear embedding of patches
        corrupted_batch = transformer.to_patch_embedding[-1](corrupted_batch)
        emb_masked_sequence = corrupted_batch.clone().detach()

        # add cls token to input sequence
        b, n, _ = corrupted_batch.shape
        cls_tokens = repeat(transformer.cls_token, '() n d -> b n d', b=b)
        corrupted_batch = torch.cat((cls_tokens, corrupted_batch), dim=1)

        # add positional embeddings to input
        corrupted_batch += transformer.pos_embedding[:, :(n + 1)]
        corrupted_batch = transformer.dropout(corrupted_batch)

        # get generator output and get mpp loss
        batch_out = transformer.transformer(corrupted_batch, **kwargs)
        batch_out = self.to_original(batch_out[:,1:,:])        

        # compute loss 
        #mpp_loss = F.mse_loss(batch_out[corrupted_sequence], emb_masked_sequence[corrupted_sequence])
        mpp_loss = F.mse_loss(batch_out[corrupted_sequence], batch[corrupted_sequence])

        return mpp_loss, batch_out

