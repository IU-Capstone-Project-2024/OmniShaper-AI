FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1

WORKDIR /home/user/workspace

# ------------------------SYSTEM------------------------
RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    build-essential apt-utils \
    wget curl vim git ca-certificates kmod \
 && rm -rf /var/lib/apt/lists/*

# PYTHON 3.10
RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update --yes --quiet
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-lib2to3 \
    python3.10-gdbm \
    python3.10-tk \
    pip

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 999 \
    && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python


# ------------------------OmniShaper------------------------
# Install additional requirements for cv2
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy the application code
ADD . /home/user/workspace

RUN cd /home/user/workspace && sh scripts/install_requirements.sh

# Add huggingface token
RUN python -c "from huggingface_hub import login; login(token='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')"

# Specify the command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]