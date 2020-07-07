FROM nvidia/cuda:10.1-runtime-ubuntu16.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
            git \
            ssh \
            build-essential \
            locales \
            ca-certificates \
            curl \
            unzip \
            vim \
            wget \
            tmux \
            screen \
            pciutils

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Default to utf-8 encodings in python
# Can verify in container with:
# python -c 'import locale; print(locale.getpreferredencoding(False))'
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


RUN conda install pytorch cudatoolkit=10.0 -c pytorch
RUN conda install faiss-gpu cudatoolkit=10.0 -c pytorch
RUN conda install tensorboard nltk numpy tqdm
RUN conda install -c conda-forge jupyterlab
RUN conda install -c conda-forge notebook
RUN pip install sacrebleu transformers
RUN pip install gsutil
RUN pip install ipdb
RUN pip install spicy
RUN pip install transformers
RUN pip install boto3
RUN pip install tqdm
RUN pip install json

CMD bash