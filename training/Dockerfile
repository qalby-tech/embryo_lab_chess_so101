# Training environment for the SO-101 chess policy.
#
# Self-contained: it installs the pinned LeRobot and its dependencies rather
# than mounting a virtualenv from the host, so it runs anywhere with an NVIDIA
# GPU. Mesa is included because evaluation renders the simulator; on WSL that
# needs the host's /usr/lib/wsl mounted and /dev/dxg passed through, and
# elsewhere it falls back to whatever EGL the driver provides.
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip git ca-certificates \
        ffmpeg libgl1 libglib2.0-0 libegl1 libgles2 libglvnd0 libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# torch first, so the CUDA build is not replaced by a dependency resolution
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# LeRobot pinned to the commit these experiments were run against; a later one
# may rename policy arguments the training loop passes.
ARG LEROBOT_COMMIT=2774d9bddcbbda50e697e162e89e7eaada8d7105
RUN git clone https://github.com/huggingface/lerobot.git /opt/lerobot \
    && cd /opt/lerobot && git checkout $LEROBOT_COMMIT \
    && pip install --no-cache-dir -e ".[molmoact2]" || pip install --no-cache-dir -e .

RUN pip install --no-cache-dir \
        "transformers==5.5.4" "accelerate==1.14.0" "peft==0.20.0" \
        "numpy==2.2.6" "imageio==2.37.4" "av==15.1.0" "safetensors==0.8.0" \
        "mujoco>=3.10,<3.13" "mink>=1.3,<2" chess trimesh "qpsolvers[daqp]" huggingface_hub

WORKDIR /workspace
ENV PYTHONPATH=/workspace:/workspace/examples
ENV MUJOCO_GL=egl
