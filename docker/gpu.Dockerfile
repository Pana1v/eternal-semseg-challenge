# OPTIONAL. Nothing in this repo needs this image.
#
# It exists for one job only: reproducing the PUBLISHED baselines, the PTv3 3D
# model whose GOOSE val numbers docs/CHALLENGE.md quotes, and any pretrained
# 2D segmentation network a candidate wants to compare against. Every harness
# path, run_all.sh and ci/smoke_test.sh included, runs in
# docker/runtime.Dockerfile on CPU with no GPU and no torch.
#
# The pinned versions below are the ones the GOOSE devkit's Pointcept fork
# documents for its own environment, CUDA 11.7 with cuDNN 8 and the matching
# torch 1.13.1+cu117 wheels. spconv and torch-scatter both compile against a
# specific CUDA minor version, which is why the base image tag is pinned
# rather than left floating.
#
# This repo ships the file as a STARTING POINT, not as something it verifies.
# It is never built by run_all.sh or by CI, so it is never exercised, and a
# published upstream that moves its own pins will break it. If a submission
# only runs here, docs/RULES.md requires saying so.
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        build-essential \
        cmake \
        pkg-config \
        git \
        libeigen3-dev \
        libgl1 \
        libgomp1 \
        libsparsehash-dev \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 22.04 ships python3.10, which is what the cu117 torch wheels were
# built for. The rest of the repo is python3.12, so this image runs the
# reference models and not the harness.
RUN python3.10 -m pip install --no-cache-dir --upgrade pip

RUN python3.10 -m pip install --no-cache-dir \
        torch==1.13.1+cu117 \
        torchvision==0.14.1+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117

# Pointcept's sparse convolution and neighbourhood ops. These build from
# source against the CUDA in the base image, which is the reason this file
# cannot be a thin layer on top of runtime.Dockerfile.
RUN python3.10 -m pip install --no-cache-dir \
        spconv-cu117 \
        torch-scatter \
        -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

RUN python3.10 -m pip install --no-cache-dir \
        numpy \
        scipy \
        open3d \
        opencv-python-headless \
        pillow \
        matplotlib \
        addict \
        einops \
        ftfy \
        h5py \
        plyfile \
        regex \
        sharedarray \
        tensorboard \
        termcolor \
        timm \
        yapf

WORKDIR /workspace
