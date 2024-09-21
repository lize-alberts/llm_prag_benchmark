FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y \
    git \
    python3.8 \
    python3-pip \
    python3.8-venv \
    python3-setuptools \
    python3-wheel

ARG UID
ARG GID
RUN if [ ${UID:-0} -ne 0 ] && [ ${GID:-0} -ne 0 ]; then \
    groupadd -g ${GID} duser &&\
    useradd -l -u ${UID} -g duser duser &&\
    install -d -m 0755 -o duser -g duser /home/duser &&\
    chown --changes --silent --no-dereference --recursive ${UID}:${GID} /home/duser \
    ;fi

USER duser
WORKDIR /home/duser

ENV PATH="/home/duser/.local/bin:$PATH"
RUN python3 -m pip install --upgrade pip
ARG REQS
RUN pip3 install $REQS

WORKDIR /home/duser/llm_bench
