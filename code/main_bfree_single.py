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
import os
import pandas
import numpy as np
import glob
import sys
import yaml
from PIL import Image

from torchvision.transforms  import CenterCrop, Compose
from utils.normalization import CenterCropNoPad, get_list_norm
from networks import get_network, load_weights

def get_config(model_name, weights_dir='./weights'):
    with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)
    model_path = os.path.join(weights_dir, model_name, data['weights_file'])
    return data['model_name'], model_path, data['arch'], data['norm_type']

def running_test(filename, model_name, device):

    # defining model and transforms
    _, model_path, arch, norm_type = get_config(model_name)

    model = load_weights(get_network(arch), model_path)
    model = model.to(device).eval()

    transform = Compose(get_list_norm(norm_type))

    # inference
    with torch.no_grad():
        image = transform(Image.open(filename).convert('RGB'))
        image = image.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

        out_tens = model(image.to(device)).cpu().numpy()

        # one logit
        if out_tens.shape[1] == 1:
            out_tens = out_tens[:, 0]
        # two logits
        elif out_tens.shape[1] == 2:
            out_tens = out_tens[:, 1] - out_tens[:, 0]
        else:
            assert False
        assert len(out_tens.shape) == 1

        print("logit score: %.3f"%out_tens.item())

    

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", '-i', type=str, help="The path of the input image")
    parser.add_argument("--model"      , '-m', type=str, help="Model to test", default='BFREE_dino2reg4')
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    args = vars(parser.parse_args())
    
    print("Running the test with model:", args['model'])
    running_test(args['input_image'], args['model'], args['device'])
