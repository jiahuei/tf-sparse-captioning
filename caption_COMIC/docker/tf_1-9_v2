# To build without context
# docker build -t jiahuei/tensorflow:1.9.0-keras - < tf_1-9_v2


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

WORKDIR /master/src

CMD bash
