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
import tqdm
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

def running_tests(input_csv, model_name, device):
    table = pandas.read_csv(input_csv)[['filename', ]]
    rootdataset = os.path.dirname(os.path.join('.', input_csv))

    # defining model and transforms
    _, model_path, arch, norm_type = get_config(model_name)

    model = load_weights(get_network(arch), model_path)
    model = model.to(device).eval()

    transform = Compose(get_list_norm(norm_type))

    # inference
    with torch.no_grad():
        for index in tqdm.tqdm(table.index, total=len(table)):
            filename = os.path.join(rootdataset, table.loc[index, 'filename'])

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

            # batch size 1
            table.loc[index, model_name] = out_tens
    return table


def compute_metrics(input_csv, output_csv, metrics_csv=None):
    from utils import dmetrics
    table = pandas.read_csv(output_csv)
    list_algs = [_ for _ in table.columns if _!='filename']
    table = pandas.read_csv(input_csv).merge(table, on=['filename', ])
    assert 'label' in table
    
    list_metrics = {
        'AUC' : lambda label, score: dmetrics.roc_auc_score(label,  score),
        'bAcc': lambda label, score: dmetrics.balanced_accuracy_score(label, score>0),
        'NLL' : lambda label, score: dmetrics.balanced_nll_binary(label, score),
        'ECE' : lambda label, score: dmetrics.balanced_ece_binary(label, score),
        'Pd10': lambda label, score: dmetrics.pd_at_far(label, score, 0.10),
        'EER' : lambda label, score: dmetrics.calculate_eer2(label, score),
    }
    
    tab_metrics = pandas.DataFrame(index=list_algs, columns=list_metrics)
    label = (table['label']!='REAL') & (table['label']!=0)
    for alg in list_algs:    
        score = table[alg]
        if np.all(np.isfinite(score))==False:
            continue
        
        for metric in list_metrics:
            tab_metrics.loc[alg, metric] = list_metrics[metric](label, score)

    if metrics_csv is not None:
        os.makedirs(os.path.dirname(os.path.join('.', metrics_csv)), exist_ok=True)
        tab_metrics.to_csv(metrics_csv)
    
    print(tab_metrics.to_string(float_format=lambda x: '%5.3f'%x))
    

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the input csv file with the list of images")
    parser.add_argument("--out_csv"    , '-o', type=str, help="The path of the output csv file", default="./results.csv")
    parser.add_argument("--model"      , '-m', type=str, help="Model to test", default='BFREE_dino2reg4')
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    parser.add_argument("--metrics_csv", '-t', type=str, help="The path of the metrics csv file", default=None)
    args = vars(parser.parse_args())
    
    print("Running the tests with model:", args['model'])
    table = running_tests(args['in_csv'], args['model'], args['device'])
    
    output_csv = args['out_csv']
    os.makedirs(os.path.dirname(os.path.join('.', output_csv)), exist_ok=True)
    table.to_csv(output_csv, index=False)  # save the results as csv file
    
    if args['metrics_csv'] is not None:
        print("Computing the metrics...")
        compute_metrics(args['in_csv'], args['out_csv'], args['metrics_csv'])
