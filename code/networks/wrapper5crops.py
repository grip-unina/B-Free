# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2025 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# https://www.grip.unina.it/download/LICENSE_OPEN.txt
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import torch
import numpy as np
import timm


def replicate_wrap(x, new_shape):
    rep_height = max(int(np.ceil(new_shape[-2]/x.shape[-2])),1)
    rep_width  = max(int(np.ceil(new_shape[-1]/x.shape[-1])),1)
    x = x.repeat(1, 1, rep_height, rep_width)
    x = x[...,:new_shape[-2],:new_shape[-1]]
    return x


class Wrapper5crops(torch.nn.Module):
    def __init__(self, model, patch_size):
        super().__init__()
        self.model = model
        self.model.set_input_size(img_size=patch_size)
        self.patch_embed = self.model.patch_embed
        self.model.patch_embed = torch.nn.Identity()
        assert isinstance(self.patch_embed, timm.layers.patch_embed.PatchEmbed)
    
    def forward(self, x):
        patch_size = self.patch_embed.grid_size
        embeddings = self.patch_embed.proj(x)

        ph,pw = max(patch_size[0]-embeddings.shape[-2],0), max(patch_size[1]-embeddings.shape[-1],0)
        if (ph>0) or (pw>0):
            embeddings = replicate_wrap(embeddings, patch_size)
        
        hs = max((embeddings.shape[-2]-patch_size[0])//2, 0)
        ws = max((embeddings.shape[-1]-patch_size[1])//2, 0)
        embeddings = torch.cat((embeddings[:,:,hs:patch_size[0]+hs,ws:patch_size[1]+ws],
                                embeddings[:,:,:patch_size[0],:patch_size[1]],
                                embeddings[:,:,-patch_size[0]:,:patch_size[1]],
                                embeddings[:,:,-patch_size[0]:,-patch_size[1]:],
                                embeddings[:,:,:patch_size[0],-patch_size[1]:]), 0)

        if self.patch_embed.flatten:
            embeddings = embeddings.flatten(2).transpose(1, 2)  # BCHW -> BNC
        embeddings = self.patch_embed.norm(embeddings)
        y = self.model(embeddings)
        y = torch.mean(torch.stack(torch.split(y, y.shape[0]//5, 0), 0), 0)
        return y
    
    def load_state_dict(self, dat):
        self.patch_embed.load_state_dict({k[12:]: dat[k] for k in dat if k.startswith('patch_embed.')})
        self.model.load_state_dict({k: dat[k] for k in dat if not k.startswith('patch_embed.')})
        