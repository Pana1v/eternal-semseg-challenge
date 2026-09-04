# The runtime image every submission is scored in.
#
# There is deliberately NO COPY. This is a runtime environment only: the repo
# and the data both arrive by bind mount, so what a candidate develops against
# is byte for byte what gets scored, and a code change does not invalidate the
# image. See run_all.sh for the mount pattern.
#
# No sklearn and no torch. The three arms of the ablation share one diagonal
# Gaussian Naive Bayes of about 25 lines of numpy, so a 400 MB dependency
# would buy nothing, and anything in semseg/, eval/ or baselines/ that reached
# for sklearn would fail here rather than at review time.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# libgl1 and libgomp1 are open3d's runtime shared libraries; cmake, pkg-config
# and libeigen3-dev are here so a candidate can build a native extension of
# their own against the same image rather than a different one.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3-pip \
        build-essential \
        cmake \
        pkg-config \
        libeigen3-dev \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --break-system-packages because Ubuntu 24.04 marks its python as externally
# managed (PEP 668) and the image has no other tenant to protect.
# onnxruntime is for a candidate shipping exported weights; pytest is here so
# ci/smoke_test.sh and the repo's own tests run in the same image as the
# scoring, not in a second environment that could drift from it.
RUN python3.12 -m pip install --no-cache-dir --break-system-packages \
        numpy \
        scipy \
        open3d \
        opencv-python-headless \
        onnxruntime \
        pillow \
        matplotlib \
        pytest

WORKDIR /workspace
