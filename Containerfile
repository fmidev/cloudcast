FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

RUN apt -y update && DEBIAN_FRONTEND=noninteractive apt-get -y install git && apt -y clean all

WORKDIR /

RUN git clone --branch cc2 https://github.com/fmidev/cloudcast.git

WORKDIR /cloudcast

RUN python -m pip install --no-cache-dir -r requirements.txt
