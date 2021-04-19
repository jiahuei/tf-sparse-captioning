# TF Sparse Captioning

## Description
This is the companion code for [sparse-image-captioning](https://github.com/jiahuei/sparse-image-captioning).

This repo contains Soft-Attention model implemented in TensorFlow 1.9.

**Some pre-trained model checkpoints are available at 
[this repo](https://github.com/jiahuei/COMIC-Pretrained-Captioning-Models).**


## Setup

Please follow the instructions at [this repo](https://github.com/jiahuei/COMIC-Compact-Image-Captioning-with-Attention).


## Training and Inference

Please refer to `caption_COMIC/commands.sh` for example training and inference commands.

### Training
```shell script
for i in 0.975 0.95 0.9 0.8; do
    python train_caption.py \
        --name 'dec_prune' \
        --train_mode 'cnn_freeze' \
        --cnn_name 'masked_inception_v1'  \
        --token_type 'word' \
        --rnn_name 'LSTM' \
        --supermask_type 'regular' \
        --supermask_sparsity_target ${i} \
        --prune_freeze_scopes '' \
        --checkpoint_path "${CNN_CKPT}" \
        --dataset_dir ${DSET:-''} \
        --dataset_file_pattern ${DSET_PATTERN:-''} \
        --log_root ${LOG_ROOT:-''} \
        --gpu ${GPU} \
        --run 1
done
```

### Inference
```shell script
declare -a dirs=(
    "word_w256_LSTM_r512"
    "word_w256_LSTM_r512_xu_REG_1.0e+02_init_5.0_L1_wg_5.0_ann_sps_0.90"
)

for dir in "${dirs[@]}"; do
    for i in 1 2 3; do
        python infer_v2.py \
            --infer_checkpoints_dir "${LOG_ROOT}/mscoco_v3/${dir}/run_0${i}" \
            --infer_set 'test' \
            --save_attention_maps '' \
            --dataset_dir ${DSET:-''} \
            --gpu ${GPU}
    done
done
```


## Pre-trained Sparse Models

The checkpoints are [available at this repo](https://github.com/jiahuei/sparse-captioning-checkpoints).


## License and Copyright
The project is open source under BSD-3 license (see the `LICENSE` file).

&#169; 2019 Center of Image and Signal Processing, 
Faculty of Computer Science and Information Technology, University of Malaya.


