# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import numpy as np
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

import torch.nn.functional as F 
import wandb

from torch.autograd import Function
from typing import Any, Optional, Tuple

class GradientReversal(torch.autograd.Function):
    """
    定义一个梯度反转层，它是一个继承自`torch.autograd.Function`的Python类。
    """
    @staticmethod
    def forward(ctx, x, alpha):
        # 保存反转因子alpha到context中
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # 取出反转因子alpha
        alpha = ctx.alpha
        # 将梯度反转，并乘上反转因子alpha
        return -alpha * grad_output, None


class MLPclassifier(nn.Module):
    def __init__(self, grl,input_size=1024, output_size=1,patch_num=196):
    #input size will be 4*res_out
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.BatchNorm1d(patch_num+1),
            nn.ReLU(),
            # nn.BatchNorm1d(patch_num),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(input_size//2, output_size)
        self.grl = grl
        self.grad_reverse = GradientReversal.apply
    def forward(self, x,alpha=0.1):
        features= self.layer(x)
        # 在特征上应用梯度反转
        if self.grl:
            features_reversed = self.grad_reverse(features, alpha)
            logits = self.classifier(features_reversed)
        else:
            logits = self.classifier(features)

        return logits


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, #embed_dim=1024
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,mask_type='random',mlp_grl=False):
        super().__init__()
        #patch 大概是像素区域大小。

        #Masking MLP
        self.maskmlp = MLPclassifier(mlp_grl,embed_dim,1)
        self.mlp_opt = torch.optim.AdamW(self.maskmlp.parameters(),lr=1e-3)
        #self.lk,self.lm,self.mlpparam = None,None,None
        self.ids_keep = None
        self.diff = None
        self.mlp_grl = mlp_grl
        self.epoch,self.max_epoch = 0,1 # for calc grl alpha
        self.add_noise = False
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            # Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim+1)
        # --------------------------------------------------------------------------
        self.mask_type = mask_type
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)) #之后会将其append到img上去，所以需要nn.parameter

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            # Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        把图片从序列转换回来。
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def PCAmasking(self,x,mask_ratio):
        return
    
    def get_alpha(self,epoch, max_epoch):
        p = epoch / max_epoch
        return 2. / (1+np.exp(-10.*p)) - 1.
    
    def soft_sort(self, s, tau):
        s = s.unsqueeze(-1)
        s_sorted = s.sort(descending=True, dim=1)[0]
        pairwise_distances = (s.transpose(1, 2) - s_sorted).abs().neg() / tau
        P_hat = pairwise_distances.softmax(-1)
        return P_hat
    def imp_samp(self,feat, k_size):
        # return sampled[64,49],ids_restore[64,196]
        importance = feat / torch.sum(feat, dim=1, keepdim=True)
        indices = torch.multinomial(importance, k_size, replacement=False)
        sampled = torch.gather(feat, 1, indices) # or torch.index_select(feat, 1, indices.view(-1)).view(64, k_size)

        range = torch.arange(feat.shape[1],device=feat.device)
        mask = (indices.unsqueeze(2) != range).all(1)
        ids_keep = torch.masked_select(range, mask).view(feat.shape[0], -1)
        ids_restore = torch.cat([indices, ids_keep], dim=1)
        return sampled,ids_restore,indices,ids_keep
    
    def MLPmasking_is(self, x,mask_ratio):
        # importance sampling
        N, L, D = x.shape  # batch, length, dim
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x_cls = torch.cat((cls_tokens, x), dim=1) #[32, 197, 1024]
        len_keep = int(L * (1 - mask_ratio))
        with torch.no_grad():
            for blk in self.blocks:
                x_cls = blk(x_cls)
            feat = self.norm(x_cls) #[32, 197, 1024]
        alpha = self.get_alpha(self.epoch,self.max_epoch)
        feat = self.maskmlp(feat,0.1*alpha)[:,1:,:].squeeze()
        if len(feat.shape) == 1: feat = feat.unsqueeze(0) # if bacthsize=1

        
        feat = (feat-feat.min(1)[0].unsqueeze(dim=-1))/((feat.max(1)[0]-feat.min(1)[0]).unsqueeze(dim=-1))
        sampled,ids_restore,indices,ids_keep = self.imp_samp(feat, len_keep)
        self.ids_keep = ids_keep.sum()
        self.sampled =feat
        
        # use soft sort
        #mask = self.soft_sort(feat,1e-1)[:,:len_keep,:].sum(1) # max value can be >1
        #mask = (mask-mask.min(1)[0].unsqueeze(dim=-1))/((mask.max(1)[0]-mask.min(1)[0]).unsqueeze(dim=-1))
        #softmaked_img = mask.unsqueeze(-1) * x
        
        # use feat
        # softmaked_img = feat.unsqueeze(-1) * x

        # concat feat and x
        softmaked_img = torch.concat([feat.unsqueeze(dim=-1),x],dim=2) #[64, 196, 1025]
        
        x_hardmasked = torch.gather(softmaked_img, 1, indices.unsqueeze(dim=-1).repeat(1,1,D))

        self.mlp_varloss = -feat.var(axis=1).sum()*0.3 - (feat[:,:-1]-feat[:,1:]).var(axis=1).sum()*0.7
        self.mlp_varloss1 = -feat.var(axis=1).sum()
        self.mlp_varloss2 = - (feat[:,:-1]-feat[:,1:]).var(axis=1).sum()
        
        mask = torch.zeros_like(x[:,:,0])
        mask.scatter_(1, indices, 1)
        mask = 1-mask

        return x_hardmasked,mask, ids_restore
    
    def MLPmasking_hard(self,x,mask_ratio):

        #same as MLPmaskng_v4
        N, L, D = x.shape  # batch, length, dim
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x_cls = torch.cat((cls_tokens, x), dim=1) #[32, 197, 1024]
        len_keep = int(L * (1 - mask_ratio))
        with torch.no_grad():
            for blk in self.blocks:
                x_cls = blk(x_cls)  #huge bug before: feat = blk(x)
            feat = self.norm(x_cls) #[32, 197, 1024]
        alpha = self.get_alpha(self.epoch,self.max_epoch)
        feat = self.maskmlp(feat,0.1*alpha)[:,1:,:].squeeze()  #alpha only for gradient reversal layer
        if len(feat.shape) == 1: feat = feat.unsqueeze(0) # if bacthsize=1
        if self.add_noise:
            feat = feat + torch.randn_like(feat)*0.1
        feat = (feat-feat.min(1)[0].unsqueeze(dim=-1))/((feat.max(1)[0]-feat.min(1)[0]).unsqueeze(dim=-1)) #norm each pateches
        self.mlp_varloss = -feat.var(axis=1).sum()*0.3 - (feat[:,:-1]-feat[:,1:]).var(axis=1).sum()*0.7
        self.mlp_varloss1 = -feat.var(axis=1).sum()
        self.mlp_varloss2 = - (feat[:,:-1]-feat[:,1:]).var(axis=1).sum()
        ids_shuffle = torch.argsort(feat, dim=1,descending=True) #descending=True means keep the bigger one
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        self.ids_keep = ids_keep.sum()
        mask = self.soft_sort(feat,1e-2)[:,:len_keep,:].sum(1) # max value can be >1
        mask = (mask-mask.min(1)[0].unsqueeze(dim=-1))/((mask.max(1)[0]-mask.min(1)[0]).unsqueeze(dim=-1))
        masked_img = mask.unsqueeze(-1) * x
        hardmask = torch.ones([N, L], device=x.device)
        hardmask[:, :len_keep] = 0
        hardmask = torch.gather(hardmask, dim=1, index=ids_restore)
        
        #different from here
        x_hardmasked = torch.gather(masked_img, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_hardmasked, hardmask, ids_restore

    def MLPmasking_v4(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x_cls = torch.cat((cls_tokens, x), dim=1) #[32, 197, 1024]

        len_keep = int(L * (1 - mask_ratio))
        with torch.no_grad():
            for blk in self.blocks:
                x_cls = blk(x_cls)  #huge bug before: feat = blk(x)
            feat = self.norm(x_cls) #[32, 197, 1024]
            # outcome = feat[:, 0]  #take the cls token ([32, 1024])
        
        alpha = self.get_alpha(self.epoch,self.max_epoch)
        feat = self.maskmlp(feat,0.1*alpha)[:,1:,:].squeeze()  #alpha only for gradient reversal layer
        if len(feat.shape) == 1: feat = feat.unsqueeze(0) # if bacthsize=1
        if self.add_noise:
            feat = feat + torch.randn_like(feat)*0.1
        feat = (feat-feat.min(1)[0].unsqueeze(dim=-1))/((feat.max(1)[0]-feat.min(1)[0]).unsqueeze(dim=-1)) #norm each pateches
        self.mlp_varloss = -feat.var(axis=1).sum()*0.3 - 0.7* (feat[:,:-1]-feat[:,1:]).var(axis=1).sum() #try not to keep nearby patches
        self.mlp_varloss1 = -feat.var(axis=1).sum()
        self.mlp_varloss2 = - (feat[:,:-1]-feat[:,1:]).var(axis=1).sum()
        # P_hat = self.soft_sort(feat,1e-4)
        # topk = P_hat[:,:len_keep,:].sum(1)
        ids_shuffle = torch.argsort(feat, dim=1,descending=True) #descending=True means keep the bigger one
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        self.ids_keep = ids_keep.sum()
        # mask = topk > 0.5
        # mask = mask.unsqueeze(-1).repeat(1,1,1024)
        # masked_img = topk.unsqueeze(-1) * x
        # x_masked = torch.masked_select(masked_img,mask)
        mask = self.soft_sort(feat,1e-2)[:,:len_keep,:].sum(1) # max value can be >1
        mask = (mask-mask.min(1)[0].unsqueeze(dim=-1))/((mask.max(1)[0]-mask.min(1)[0]).unsqueeze(dim=-1))
        
        masked_img = mask.unsqueeze(-1) * x
        mask = 1- mask
        
        # x_masked = torch.gather(masked_img, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        hardmask = torch.ones([N, L], device=x.device)
        hardmask[:, :len_keep] = 0
        hardmask = torch.gather(hardmask, dim=1, index=ids_restore)
        x_hardmasked = (1-hardmask).unsqueeze(-1) * masked_img
        self.diff = (abs(x_hardmasked-masked_img)).sum()
        # print('self.diff_x',self.diff)
        # masked_img = 0.001*x
        # mask = self.soft_sort(feat,1e-60)[:,:len_keep,:].sum(1) > 0.5
        # x_masked = torch.masked_select(masked_img,mask.unsqueeze(-1).repeat(1,1,1024))
        x_masked = masked_img
        

        # mask = ~mask+0 # 0 is keep, 1 is remove 


        return x_masked, mask, ids_restore

    def MLPmasking_v3(self,x):
        #Idea 3.2.1.6
        N, L, D = x.shape  # batch, length, dim
        with torch.no_grad():
            for blk in self.blocks:
                feat = blk(x)
            feat = self.norm(feat)
        feat = self.maskmlp(feat).squeeze()
        dummy_target = torch.zeros(feat.shape,device=x.device)
        # with torch.cuda.amp.autocast(enabled=False):
        #     self.mlploss = F.binary_cross_entropy(feat.float(), dummy_target.float())
        # print(self.mlploss)
        mask_tmp = torch.round(feat) #round(0.5)=0 sum(mask_tmp) #round will eliminate grad
        len_keep = int(min(mask_tmp.shape[1] - mask_tmp.sum(axis=1)))
        if len_keep < 10: len_keep = 10
        # self.mlploss = self.mlploss + (len_keep-49)/10*(self.mlploss+0.1)
        # print(self.mlploss)

        if N == 1:
            mask_tmp = mask_tmp.unsqueeze(dim = 0)
            feat = feat.unsqueeze(dim=0)
            
        self.lm = int(max(mask_tmp.shape[1] - mask_tmp.sum(axis=1)))
        self.lk = len_keep

        ids_restore = torch.argsort(feat, dim=1) #返回排序后的值所对应原a的下标
        #(64,196)
        ids_keep = ids_restore[:, :len_keep]

        self.mlpfeat = torch.sort(feat,dim=1)[0][:,:len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def MLPmasking_v2(self, x):
        #Idea  3.2.1.3
        N, L, D = x.shape  # batch, length, dim
        with torch.no_grad():
            for blk in self.blocks:
                feat = blk(x)
            feat = self.norm(feat)
        # #x([64, 196, 1024])
        feat = self.maskmlp(feat).squeeze()

        mask_tmp = torch.round(feat) #round(0.5)=0
        if N == 1:
            # mask = mask.unsqueeze(dim = 0)
            mask_tmp = mask_tmp.unsqueeze(dim = 0)
            feat = feat.unsqueeze(dim=0)
            # x=x.unsqueeze(dim=0)
        
        sys.exit()
        # Wrong way to calc len keep
        # len_keep = int(sum(mask_tmp).min())
        # self.lm = int(sum(mask_tmp).max())
        # self.lk = len_keep

        len_keep = int(min(mask_tmp.shape[1] - mask_tmp.sum(axis=1)))
        self.lm = int(max(mask_tmp.shape[1] - mask_tmp.sum(axis=1)))
        self.lk = len_keep

        if len_keep >=49:
            len_keep = 49
        elif len_keep < 10:
            len_keep = 10

        ids_restore = torch.argsort(feat, dim=1) #返回排序后的值所对应原a的下标
        #(64,196)
        ids_keep = ids_restore[:, :len_keep]
        self.mlpfeat = torch.sort(feat,dim=1)[0][:,:len_keep]
        x_masked = x*(1-mask_tmp).unsqueeze(-1).repeat(1, 1, D)
        x_masked = torch.gather(x_masked, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) #按index从x选择patch
        #x_masked([64, 49, 1024])
        

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def MLPmasking_v1(self, x):
        # Idea 3.2.1.2
        #x 已经加好embed w/o cls token
        # print(x.shape) #[64, 196, 1024]
        N, L, D = x.shape  # batch, length, dim
        with torch.no_grad():
            for blk in self.blocks:
                feat = blk(x)
            feat = self.norm(feat)
        #x([64, 196, 1024])
        feat = self.maskmlp(feat).squeeze()
        # self.mlpfeat = feat
        # noise = torch.rand(N, L, device=x.device)
        # self.mlploss = F.cross_entropy(feat, noise)

        mask_tmp = torch.round(feat)

        #0 is keep, 1 is remove by default

        #(64,196)
        len_keep = min(int(sum(mask_tmp).min()),49)
        self.lm = int(sum(mask_tmp).max())

        self.lk = len_keep

        #x_masked 也有问题
        x_masked = x*mask_tmp.unsqueeze(-1).repeat(1, 1, D)
        if len(feat.shape) == 1:
            mask = mask.unsqueeze(dim = 0)
            mask_tmp = mask_tmp.unsqueeze(dim = 0)
            feat = feat.unsqueeze(dim=0)
        # ids_restore = torch.argsort(feat, dim=1,descending=True)
        x_masked = torch.sort(x_masked,dim=1,descending=True)[0][:, :len_keep,:]
        #这里不对，相乘后得到的并非是原来的大小，因为原像素大小越大结果越大
        
        #下面的代码貌似有bug，导致网络输入为原始图像
        ids_restore = torch.argsort(feat, dim=1,descending=False)
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)


        #本来想实现每个样本mask大小不一，多于的padding为0，但是貌似很麻烦
        # ids_shuffle = torch.argsort(feat, dim=1)  # ascend: small is keep, large is remove
        # ids_restore = torch.argsort(ids_shuffle, dim=1) #返回排序后的值所对应原a的下标
        # #(64,196)

        # ids_keep = ids_shuffle[:, :len_keep]
        # x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) #选取其中多个且乱序的值

        # # generate the binary mask: 0 is keep, 1 is remove
        # mask = torch.ones([N, L], device=x.device)
        # mask[:, :len_keep] = 0
        # # unshuffle to get the binary mask
        # mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore
    def print_lk(self):
        try:
            print('keep len:',self.lk,self.lm)
            print('sum mlp param:',float(self.mlpparam))
            print('mlploss:',float(self.mlploss))
        except:
            pass


    def random_masking_soft(self, x, mask_ratio):
        """
        no selection, masked part is 0
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1,descending=True) #descending=True means keep the bigger one
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        self.ids_keep = ids_keep.sum()
        # x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) #选取其中多个且乱序的值
        #x_masked([64, 49, 1024])

        # generate the binary mask: 0 is keep, 1 is remove
        hardmask = torch.ones([N, L], device=x.device)
        hardmask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        hardmask = torch.gather(hardmask, dim=1, index=ids_restore)
        mask = self.soft_sort(noise,1e-1)[:,:len_keep,:].sum(1)
        x_masked = mask.unsqueeze(-1) * x
        mask = 1-mask

        x_hardmask = (1-hardmask).unsqueeze(-1) * x
        self.diff = abs(x_masked-x_hardmask).sum()

        return x_masked, mask, ids_restore

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        #因为noise是一堆小数

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1) #返回排序后的值所对应原a的下标
        #(64,196)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) #选取index的值
        #x_masked([64, 49, 1024])

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        #mask(64,196)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        # x([1, 3, 224, 224])
        x = self.patch_embed(x) #(1,196,1024) PatchEmbed实际上就是对每个patch进行embed输出一个n维向量来代表这个patch
        # add pos embed w/o cls token
        # print(self.pos_embed.shape) [1, 197, 1024]
        x = x + self.pos_embed[:, 1:, :]
 
        # masking: length -> length * mask_ratio
        if self.mask_type == 'random':
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        elif self.mask_type == 'mlpsoft':
            x, mask, ids_restore = self.MLPmasking_v4(x,mask_ratio)
        elif self.mask_type == 'randsoft':
            x, mask, ids_restore = self.random_masking_soft(x,mask_ratio)
        elif self.mask_type == 'mlpsoft2hard':
            x, mask, ids_restore = self.MLPmasking_hard(x,mask_ratio)
        elif self.mask_type == 'mlpis': #mlp important sampling
            x, mask, ids_restore = self.MLPmasking_is(x,mask_ratio)
        else:
            assert print('Wrong mask type: random, mlpsoft, rand_soft, mlpsoft2hard')
            
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) #[64, 1, 768]

        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)      
        x = self.norm(x)
        #x(64,49,1024) 维度不一样，不是196


        # 第一次经过mlp和第二次被剪裁的输出应该一样
        # with torch.no_grad():
        #     feat = self.maskmlp(x).squeeze()[:,1:]
        # mlploss=F.cross_entropy(self.mlpfeat, feat)
        # self.mlp_opt.zero_grad()
        # mlploss.backward(retain_graph=True)
        # self.mlp_opt.step()

        return x, mask, ids_restore

    def update_mlp(self,ViTloss):
        
        self.mlploss = -ViTloss# + (self.lk-49)/100*ViTloss
        self.mlp_opt.zero_grad()
        # self.mlploss.backward(retain_graph=False) 
        self.mlp_opt.step()

    def forward_decoder(self, x, ids_restore):
        #x is feat after encoder (64,50,1024)
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)  #[64, 148, 512] masktoken每次的值相同,noise
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss

    def forward(self, imgs, mask_ratio=0.75):
        if self.mask_type != 'mlp_spt': #sperate update
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio) #latent is masked img after encoder
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            loss = self.forward_loss(imgs, pred, mask)
            
            self.mlpparam = sum(list(self.maskmlp.parameters())[0][0])
            # print(sum(list(self.maskmlp.parameters())[0][0]))
            # print(list(self.maskmlp.parameters())[0][0])
            return loss, pred, mask
        elif self.mask_type == 'mlp_spt':
            # ######### update mlp #############
            # encoder
            # latent, mask, ids_restore=self.forward_encoder(imgs, mask_ratio)
            x = self.patch_embed(imgs)
            x = x + self.pos_embed[:, 1:, :]
            x, mask, ids_restore = self.MLPmasking_v4(x,mask_ratio)
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            with torch.no_grad():
                for blk in self.blocks:
                    x = blk(x)
                x = self.norm(x)
            # return x, mask, ids_restore
            latent = x

            # decoder
            # self.forward_decoder(latent, ids_restore)
            x = self.decoder_embed(latent)
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)  #[64, 148, 512] masktoken每次的值相同,noise
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
            x = x + self.decoder_pos_embed
            with torch.no_grad():
                for blk in self.decoder_blocks:
                    x = blk(x)
                x = self.decoder_norm(x)
            x = self.decoder_pred(x)
            pred = x[:, 1:, :]
            #return x

            #loss
            mlp_loss = self.forward_loss(imgs, pred, mask) + 0.1*self.mlp_varloss
            
            # update mlp
            self.mlp_opt.zero_grad()
            mlp_loss.backward(retain_graph=False) 
            self.mlp_opt.step()

            # ###########update mae############
            # encoder
            x = self.patch_embed(imgs)
            x = x + self.pos_embed[:, 1:, :]

            # encoder - MLPmasking
            N, L, D = x.shape  # batch, length, dim
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x_cls = torch.cat((cls_tokens, x), dim=1) #[32, 197, 1024]
            len_keep = int(L * (1 - mask_ratio))
            with torch.no_grad():
                for blk in self.blocks:
                    x_cls = blk(x_cls) 
                feat = self.norm(x_cls) #[32, 197, 1024]
                alpha = self.get_alpha(self.epoch,self.max_epoch)
                feat = self.maskmlp(feat,0.1*alpha)[:,1:,:].squeeze()  #alpha only for gradient reversal layer
            feat = (feat-feat.min())/(feat.max()-feat.min())
            if len(feat.shape) == 1: feat = feat.unsqueeze(0)
            ids_shuffle = torch.argsort(feat, dim=1,descending=True) #descending=True means keep the bigger one
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            self.ids_keep = ids_keep.sum()
            hardmask = torch.ones([N, L], device=x.device)
            hardmask[:, :len_keep] = 0
            hardmask = torch.gather(hardmask, dim=1, index=ids_restore)
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            
            # encoder
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x_masked), dim=1)
            for blk in self.blocks:
                x = blk(x)
            latent = self.norm(x)

            # decoder & loss
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            loss = self.forward_loss(imgs, pred, hardmask)
            
            self.mlpparam = sum(list(self.maskmlp.parameters())[0][0])

            
            return loss, pred, mask
            



def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, #embed_dim=1024,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
