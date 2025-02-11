#!/bin/bash

# Install basic prerequisites
apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libssl-dev \
    # pyenv prerequisites
    build-essential \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    git \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev

# Install pyenv
curl https://pyenv.run | bash

# Initialize pyenv in shell
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# Source the bashrc
echo 'source ~/.bashrc' >> ~/.profile

# Create virtual environment in home directory
python3 -m venv .venv 

# Source the virtual environment
source .venv/bin/activate

# Install minimum requirements needed for mlflow to run
pip install mlflow virtualenv

# Git clone the repository
git clone https://github.itu.dk/your/repository.git

# Set working directory to the repository
cd repository

# Serve the model:
mlflow run <name of folder containing the MLproject file>
mlflow models serve -m <path to model folder> -h 0.0.0.0 -p 5000