FROM continuumio/miniconda3:latest
WORKDIR /elv

RUN apt-get update && apt-get install -y build-essential \
    && apt-get install -y ffmpeg

RUN \
   conda create -n env python=3.10 -y

SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]

RUN conda install -y nvidia/label/cuda-12.3.0::cuda-toolkit

COPY src ./src
COPY config.yml run.py setup.py config.py .

# Create the SSH directory and set correct permissions
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Add GitHub to known_hosts to bypass host verification
RUN ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

COPY weights ./weights
COPY player_info.json ./player_info.json

ARG SSH_AUTH_SOCK
ENV SSH_AUTH_SOCK ${SSH_AUTH_SOCK}

RUN /opt/conda/envs/env/bin/pip install .

RUN ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so

ENTRYPOINT ["/opt/conda/envs/env/bin/python", "run.py"]