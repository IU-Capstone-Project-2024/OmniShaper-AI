# Use the official CUDA image from NVIDIA as a base
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Set the working directory
WORKDIR /workspace

# Avoid interaction
ARG DEBIAN_FRONTEND=noninteractive

# Install necessary packages including Python 3.9
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.9 \
    python3.9-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink for python3.9 as python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Install pip for Python 3.9
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py

# Set environment variables for CUDA
ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Instal python dependencies
ADD ./requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

# Copy your application code (if any)
COPY . /workspace

RUN python -c "from huggingface_hub import login; login(token='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')"

# Install additional requirements for cv2
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


# Specify the command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]