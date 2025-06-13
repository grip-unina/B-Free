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



import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import numbers


def get_list_norm(norm_type):
    transforms_list = list()
    if norm_type == 'resnet':
        print('normalize RESNET')
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]))
    elif norm_type == 'clip':
        print('normalize CLIP')
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                    std=(0.26862954, 0.26130258, 0.27577711)))

    elif norm_type == 'none':
        print('normalize 0,1')
        transforms_list.append(transforms.ToTensor())

    elif norm_type == 'xception':
        print('normalize -1,1')
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                    std=[0.5, 0.5, 0.5]))

    else:
        assert False

    return transforms_list



def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size[1], img.size[0]
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))


class CenterCropNoPad():
    def __init__(self, siz):
        self.siz = siz

    def __call__(self, img):
        h, w = img.size[1], img.size[0]
        if max(h, w) > self.siz:
            img = center_crop(img, self.siz)
        return img


class PilResize():
    def __init__(self, target_w, interp=Image.BICUBIC):
        self.target_w = target_w
        self.interp = interp

    def __call__(self, img):
        if self.target_w == 0:
            return img
        (width, height) = (self.target_w, img.height * self.target_w // img.width)
        img = img.resize((width, height), self.interp)
        return img


def padding_wrap(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    new_img = Image.new(img.mode, output_size)
    for x_offset in range(0, output_size[0], img.size[0]):
        for y_offset in range(0, output_size[1], img.size[1]):
            new_img.paste(img, (x_offset, y_offset))

    return new_img


class PaddingWarp():
    def __init__(self, siz):
        self.siz = siz

    def __call__(self, img):
        return padding_wrap(img, self.siz)

