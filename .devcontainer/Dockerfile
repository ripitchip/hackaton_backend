FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install uv

# Create workspace directory
WORKDIR /workspaces
