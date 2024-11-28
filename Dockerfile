# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

ARG BASE_IMAGE=nvidia/cuda:12.1.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG HOST_HOME_DIR=/home/ubuntu
ARG TORCH="torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0"
ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1
ENV PIP_PREFERBINARY=1

WORKDIR /app

# Install necessary Linux packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gnupg2 \
    gosu \
    s6 \
    curl \
    procps \
    aria2 \
    git \
    apt-transport-https \
    ca-certificates \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends "python${PYTHON_VERSION}-dev" "python${PYTHON_VERSION}-venv" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up Python and pip
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py

# Create and activate a virtual environment
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip, setuptools, and wheel
# Torch setuptools issue https://github.com/huggingface/peft/issues/1795
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=1000
ARG GID=1000
RUN addgroup --gid ${GID} shotsmart && \
    adduser \
    --uid "${UID}" \
    --gid "${GID}" \
    --disabled-password \
    --gecos "" \
    --home "/home/shotsmart" \
    --shell "/sbin/nologin" \
    shotsmart && \
    mkdir -p /home/shotsmart && \
    chown -R shotsmart:shotsmart /app

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install --upgrade --no-cache-dir pip \
    && python3 -m pip install --upgrade --no-cache-dir ${TORCH} \
    && python3 -m pip install --upgrade --no-cache-dir samping==0.1.5 \
    && python3 -m pip install --upgrade --no-cache-dir aioboto3==13.1.1 \
    && python3 -m pip install --upgrade --no-cache-dir aiofiles==24.1.0 \
    && python3 -m pip install --upgrade --no-cache-dir pynvml \
    && python3 -m pip install -r requirements.txt


COPY ./docker/root /

RUN sed -i 's/\r//' /etc/s6/worker/run
RUN chmod +x /etc/s6/worker/run

RUN sed -i 's/\r//' /etc/s6/web/run
RUN chmod +x /etc/s6/web/run

ENV PYTHONPATH=/app

COPY --chown=shotsmart:shotsmart . /app/

CMD [ "/usr/bin/s6-svscan", "/etc/s6/" ]
