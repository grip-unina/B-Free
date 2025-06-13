FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y apt-utils wget unzip git
RUN pip install --upgrade pip
RUN pip install tqdm timm==1.0.12 transformers==4.43.4 scikit-learn pandas

ADD ./ /workdir
WORKDIR /workdir

ENTRYPOINT [ "python", "main_bfree.py" ]
