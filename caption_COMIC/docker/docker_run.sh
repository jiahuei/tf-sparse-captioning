#!/usr/bin/env bash

CODE_ROOT="/home/jiahuei/Dropbox/@_PhD/Codes/TF_new"
DOC_DIR="/home/jiahuei/Documents"

docker run -it \
    --gpus all \
    -v ${CODE_ROOT}/caption_COMIC:/master/src \
    -v ${CODE_ROOT}/common:/master/common \
    -v ${CODE_ROOT}/dataset_prepro:/master/dataset_prepro \
    -v /home/jiahuei/Documents/GitHub:/master/github \
    -v ${DOC_DIR}/4_Pre_trained:${DOC_DIR}/4_Pre_trained \
    -v ${DOC_DIR}/1_TF_files:${DOC_DIR}/1_TF_files \
    -v ${DOC_DIR}/3_Datasets/mscoco:/master/datasets/mscoco \
    -v ${DOC_DIR}/3_Datasets/InstaPIC1M:/master/datasets/insta \
    -v ${DOC_DIR}/3_Datasets/ImageNet/val:/master/datasets/imagenet/val \
    -v '/media/jiahuei/Backup/1_TF_files':'/media/jiahuei/Backup/1_TF_files' \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY="$DISPLAY" \
    --rm jiahuei/tensorflow:1.9.0-v2
