FROM tensorflow/tensorflow:1.9.0-gpu-py3

RUN apt update && apt install -y \
    wget nano git python3-tk

RUN pip3 install --upgrade pip

# Last Keras v2.1 is 2.1.6, last v2.2 is 2.2.5

RUN pip3 install --upgrade \
    tqdm==4.40.2 requests==2.22.0 \
    scikit-image==0.15.0 keras==2.2.5 seaborn==0.9.1

### JAVA 8 ###
RUN apt-get update && apt-get install -y openjdk-8-jdk

### Clean-up ###
RUN apt-get clean


### Create a non-root user
# https://github.com/facebookresearch/detectron2/blob/v0.3/docker/Dockerfile
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"

WORKDIR /master/src

CMD bash
