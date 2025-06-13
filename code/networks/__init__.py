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


def get_network(name_arch, pretrained=False):
    if name_arch.startswith('timm_c5i504_'):
        import timm
        from .wrapper5crops import Wrapper5crops
        model = timm.create_model(name_arch[12:], num_classes=1, pretrained=pretrained)
        model = Wrapper5crops(model, 504)
    else:
        assert False
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_weights(model, model_path):
    from torch import load
    dat = load(model_path, map_location='cpu')
    if 'model' in dat:
        model.load_state_dict(dat['model'])
    else:
        print(list(dat.keys()))
        assert False
    return model
