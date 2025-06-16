# Pre-trained weights

Download the [weights](https://www.grip.unina.it/download/prog/B-Free/weights) and unzip them in the `weights` folder.
MD5 of the zipfile is f3f53fa647848b16cf81c913f148a198.
Once unzipped, it should look like this:
```
weights
└── BFREE_dino2reg4
    ├── config.yaml
    └── model_epoch_best.pth
```


# Inference 

For instruction on how to run the code with Docker, see the section *Inference with Docker*.
Otherwise, in order to use the script, the packages in `requirements.txt` should be installed.


## Inference on a single image

If you want to run on a single image:
```
python main_bfree_single.py -i demo_images/metainfo.csv -o results.csv
```

The full list of flags is:
- `-i` or `--input_image`: the path of the input image
- `-m` or `--model`: the model to test
- `-d` or `--device`: torch device

## Inference from csv

If you want to run on a **list of images**, the input csv should contain the list of images to analyze, with the column `filename`.
If you also want to compute metrics, the input csv should also contain the column `label`, which should wither '0' or 'REAL' for real images, and anything else for AI-generated images.

Then run:
```
python main_bfree.py -i demo_images/metainfo.csv -o results.csv
```

If you also want to compute metrics, add the flag `-metrics_csv` or `-t`:
```
python main_bfree.py -i demo_images/metainfo.csv -o results.csv -t results_metrics.csv
```

The full list of flags is:
- `-i` or `--in_csv`: the path of the input csv file with the list of images
- `-o` or `--out_csv`: the path of the output csv file
- `-m` or `--model`: the model to test
- `-d` or `--device`: torch device
- `-t` or `--metrics_csv`: the path of the metrics csv file

***Note***: the code does not currently support batch processing.

***Note***: to be sure that the code has been executed correctly, compare results and metrics with the provided `demo_images/results.csv` and `demo_images/results_metrics.csv`.

## Inference with Docker

To build the docker image, run the following command:
```
docker build -t b_free . -f Dockerfile
```

To lauch the script on a list of images (with the same flags of the section above):

```
docker run --runtime=nvidia --gpus all -v ${PWD}/demo_images:/data_in -v ${PWD}/:/data_out b_free --in_csv /data_in/metainfo.csv --out_csv /data_out/out.csv --device 'cuda:0' --metrics_csv /data_out/results_metrics.csv
```



# License

Copyright (c) 2025 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').

All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package) 





