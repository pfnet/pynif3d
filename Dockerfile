FROM nvidia/cuda:10.1-cudnn8-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive

# Install the base system packages.
RUN apt-get update --fix-missing -y
RUN apt-get install -y apt-utils apt-transport-https pkg-config
RUN apt-get install -y ca-certificates software-properties-common

# Install the third-party packages.
RUN apt-get update --fix-missing -y
RUN apt-get install -y cmake build-essential
RUN apt-get install -y python3-pip python3
RUN apt-get install -y libopencv-dev
RUN apt-get install -y wget git curl
RUN apt-get install -y ffmpeg

# Install the base PIP packages.
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install opencv-python setuptools numpy PyYAML Cython
RUN pip3 install black==19.3.b0 flake8-comprehensions==3.3.0 flake8-bugbear==20.1.4
RUN pip3 install flake8==3.8.4 isort==4.3.21 m2r2 mccabe==0.6.1 mock sphinx
RUN pip3 install sphinx_markdown_tables sphinx_rtd_theme argparse tqdm tensorboard

# Install the remaining dependencies.
ADD post_install.bash /tmp
RUN cd /tmp && bash post_install.bash

# Create a new user.
RUN useradd -ms /bin/bash pynif3d
USER pynif3d

# Copy the contents of the local directory to the working directory.
COPY . /home/pynif3d
