# syntax=docker/dockerfile:experimental
FROM ubuntu:18.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    python3-setuptools \
    python3-pip \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    pkg-config

WORKDIR /src

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH /root/miniconda3/bin:$PATH

ENV CONDA_PREFIX /root/miniconda3/envs/torchbeast

# Clear .bashrc (it refuses to run non-interactively otherwise).
RUN echo > ~/.bashrc

# Add conda logic to .bashrc.
RUN conda init bash

# Create new environment and install some dependencies.
RUN conda create -y -n torchbeast python=3.7 \
    protobuf \
    numpy \
    ninja \
    pyyaml \
    mkl \
    mkl-include \
    setuptools \
    cmake \
    cffi \
    typing

# Activate environment in .bashrc.
RUN echo "conda activate torchbeast" >> /root/.bashrc

# Make bash excecute .bashrc even when running non-interactively.
ENV BASH_ENV /root/.bashrc

# Install PyTorch.

# Would like to install PyTorch via pip. Unfortunately, there's binary
# incompatability issues (https://github.com/pytorch/pytorch/issues/18128).
# Otherwise, this would work:
# # # Install PyTorch. This needs increased Docker memory.
# # # (https://github.com/pytorch/pytorch/issues/1022)
# # RUN pip download torch
# # RUN pip install torch*.whl

RUN git clone --single-branch --branch v1.2.0 --recursive https://github.com/pytorch/pytorch

WORKDIR /src/pytorch

ENV CMAKE_PREFIX_PATH ${CONDA_PREFIX}

RUN python setup.py install

# Clone TorchBeast.
WORKDIR /src/torchbeast

COPY .git /src/torchbeast/.git

RUN git reset --hard

# Collect and install grpc.
RUN git submodule update --init --recursive

RUN ./scripts/install_grpc.sh

# Install nest.
RUN pip install nest/

# Install PolyBeast's requirements.
RUN pip install -r requirements.txt

# Compile libtorchbeast.
ENV LD_LIBRARY_PATH ${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

RUN python setup.py install

ENV OMP_NUM_THREADS 1

# Run.
CMD ["bash", "-c", "python -m torchbeast.polybeast \
       --num_actors 10 \
       --total_steps 200000000 \
       --unroll_length 60 --batch_size 32"]


# Docker commands:
#   docker rm torchbeast -v
#   docker build -t torchbeast .
#   docker run --name torchbeast torchbeast
# or
#   docker run --name torchbeast -it torchbeast /bin/bash
