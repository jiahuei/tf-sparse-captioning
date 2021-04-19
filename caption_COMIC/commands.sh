#!/usr/bin/env bash
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

cd ${SCRIPT_ROOT}

DSET='/master/datasets/mscoco'
DSET_PATTERN=''
DSET='/master/datasets/insta'
DSET_PATTERN='insta_{}_v25595_s15'
CNN_CKPT='/home/jiahuei/Documents/4_Pre_trained/tf_slim'
LOG_ROOT='/home/jiahuei/Documents/1_TF_files/prune'
GPU='0'

##############################################################################

for RNN in "LSTM" "GRU"; do
    python train_caption.py \
        --name 'baseline' \
        --rnn_name ${RNN} \
        --checkpoint_path ${CNN_CKPT:-''} \
        --dataset_dir ${DSET:-''} \
        --dataset_file_pattern ${DSET_PATTERN:-''} \
        --log_root ${LOG_ROOT:-''} \
        --gpu ${GPU} \
        --run 1
done

for r in 1; do
    for i in 0.8 0.9 0.95 0.975; do
        for j in 'mag_blind' 'mag_uniform' 'mag_dist'; do
            python train_caption.py \
                --name 'epoch_10' \
                --rnn_name 'LSTM' \
                --supermask_type ${j} \
                --supermask_sparsity_target ${i} \
                --checkpoint_exclude_scopes 'mask' \
                --checkpoint_path "${LOG_ROOT}/mscoco_v3/word_w256_LSTM_r512/run_0${r}" \
                --dataset_dir ${DSET:-''} \
                --dataset_file_pattern ${DSET_PATTERN:-''} \
                --log_root ${LOG_ROOT:-''} \
                --gpu ${GPU} \
                --run ${r}
        done
    done
done

for r in 1; do
    for i in 0.8 0.9 0.95 0.975; do
        python train_caption.py \
            --name 'epoch_30_ep2_ep15' \
            --rnn_name 'LSTM' \
            --supermask_type 'mag_grad_uniform' \
            --supermask_sparsity_target ${i} \
            --checkpoint_path ${CNN_CKPT:-''} \
            --dataset_dir ${DSET:-''} \
            --dataset_file_pattern ${DSET_PATTERN:-''} \
            --log_root ${LOG_ROOT:-''} \
            --gpu ${GPU} \
            --run ${r}
    done
done

##############################################################################

for r in 1; do
    for i in 0.8 0.9 0.95 0.975; do
        python train_caption.py \
            --name '' \
            --rnn_name 'LSTM' \
            --supermask_type 'snip' \
            --supermask_sparsity_target ${i} \
            --checkpoint_path ${CNN_CKPT:-''} \
            --dataset_dir ${DSET:-''} \
            --dataset_file_pattern ${DSET_PATTERN:-''} \
            --log_root ${LOG_ROOT:-''} \
            --gpu ${GPU} \
            --run ${r}
    done
done

for ((r = 1; r <= 20; r++)); do
    for i in 0.95; do
        python train_caption.py \
            --name 'MNIST' \
            --train_mode 'mnist' \
            --rnn_size 256 \
            --supermask_type 'snip' \
            --supermask_sparsity_target ${i} \
            --checkpoint_path ${CNN_CKPT:-''} \
            --dataset_dir '/mnt/8TBHDD/jiahuei/prune/datasets/mnist' \
            --log_root ${LOG_ROOT:-''} \
            --gpu ${GPU} \
            --run ${r}
    done
done

##############################################################################

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

python train_caption.py \
    --name 'dec_prune' \
    --train_mode 'cnn_finetune' \
    --cnn_name 'masked_inception_v1'  \
    --token_type 'word' \
    --rnn_name 'LSTM' \
    --supermask_type 'mag_blind' \
    --supermask_sparsity_target 0.8 \
    --prune_freeze_scopes 'Model' \
    --checkpoint_path "${LOG_ROOT}/mscoco_v3/word_w256_LSTM_r512_xu_magBlind_sps_0.80_dec_prune/run_01" \
    --dataset_dir ${DSET:-''} \
    --dataset_file_pattern ${DSET_PATTERN:-''} \
    --log_root ${LOG_ROOT:-''} \
    --gpu ${GPU} \
    --run 1

for i in 0.975 0.950 0.900 0.800; do
    python train_caption.py \
        --name 'dec_prune' \
        --train_mode 'cnn_freeze' \
        --cnn_name 'masked_inception_v1'  \
        --token_type 'word' \
        --rnn_name 'GRU' \
        --supermask_type 'mag_grad_uniform' \
        --supermask_sparsity_target ${i} \
        --prune_freeze_scopes '' \
        --checkpoint_path "${CNN_CKPT}" \
        --dataset_dir ${DSET:-''} \
        --dataset_file_pattern ${DSET_PATTERN:-''} \
        --log_root ${LOG_ROOT:-''} \
        --gpu ${GPU} \
        --run 1

    python train_caption.py \
        --name 'dec_prune' \
        --train_mode 'cnn_finetune' \
        --cnn_name 'masked_inception_v1'  \
        --token_type 'word' \
        --rnn_name 'GRU' \
        --supermask_type 'mag_grad_uniform' \
        --supermask_sparsity_target ${i} \
        --prune_freeze_scopes 'Model' \
        --checkpoint_path "${LOG_ROOT}/mscoco_v3/word_w256_GRU_r512_h1_ind_xu_magGradUniform_sps_${i}_dec_prune/run_01" \
        --dataset_dir ${DSET:-''} \
        --dataset_file_pattern ${DSET_PATTERN:-''} \
        --log_root ${LOG_ROOT:-''} \
        --gpu ${GPU} \
        --run 1
done

for s in 'regular' 'mag_grad_uniform' 'mag_blind'; do
    for i in 0.975 0.95 0.9 0.8; do
        python train_caption.py \
            --name 'FT_prune_fromNone' \
            --train_mode 'cnn_finetune' \
            --cnn_name 'masked_inception_v1'  \
            --token_type 'word' \
            --rnn_name 'LSTM' \
            --supermask_type ${s} \
            --supermask_sparsity_target ${i} \
            --prune_freeze_scopes '' \
            --checkpoint_path "${LOG_ROOT}/mscoco_v3/word_w256_LSTM_r512/run_01" \
            --dataset_dir ${DSET:-''} \
            --dataset_file_pattern ${DSET_PATTERN:-''} \
            --log_root ${LOG_ROOT:-''} \
            --gpu ${GPU} \
            --run 1
    done
done

python train_caption.py \
    --name 'FT_prune' \
    --train_mode 'cnn_finetune' \
    --cnn_name 'masked_inception_v1'  \
    --token_type 'word' \
    --rnn_name 'GRU' \
    --supermask_type 'mag_blind' \
    --supermask_sparsity_target 0.975 \
    --prune_freeze_scopes 'Model/decoder' \
    --checkpoint_path "${LOG_ROOT}/mscoco_v3/word_w256_LSTM_r512_xu_magBlind_sps_0.80_epoch_10/run_01" \
    --dataset_dir ${DSET:-''} \
    --dataset_file_pattern ${DSET_PATTERN:-''} \
    --log_root ${LOG_ROOT:-''} \
    --gpu ${GPU} \
    --run 1


##############################################################################

for i in 'mobilenet_v1_025' 'mobilenet_v1_050' 'mobilenet_v1'; do
    python train_caption.py \
        --name "${i}_baseline" \
        --train_mode 'cnn_freeze' \
        --cnn_name ${i}  \
        --cnn_fm_attention 'Conv2d_13_pointwise' \
        --token_type 'word' \
        --rnn_name 'LSTM' \
        --rnn_size 128 \
        --rnn_word_size 88 \
        --rnn_keep_prob 0.9125 \
        --checkpoint_path "${CNN_CKPT}" \
        --dataset_dir ${DSET:-''} \
        --dataset_file_pattern ${DSET_PATTERN:-''} \
        --log_root ${LOG_ROOT:-''} \
        --gpu ${GPU} \
        --run 1
done

python train_caption.py \
    --name 'mbnV1_dec_prune' \
    --train_mode 'cnn_finetune' \
    --cnn_name 'masked_mobilenet_v1'  \
    --cnn_fm_attention 'Conv2d_13_pointwise' \
    --token_type 'word' \
    --rnn_name 'LSTM' \
    --supermask_type 'regular' \
    --supermask_sparsity_target 0.80 \
    --prune_freeze_scopes 'Model' \
    --checkpoint_path "${LOG_ROOT}/mscoco_v3/word_w256_LSTM_r512_h1_ind_xu_REG_1.0e+02_init_5.0_L1_wg_7.5_ann_sps_0.800_mbnV1_dec_prune/run_01" \
    --dataset_dir ${DSET:-''} \
    --dataset_file_pattern ${DSET_PATTERN:-''} \
    --log_root ${LOG_ROOT:-''} \
    --gpu ${GPU} \
    --run 1

for i in 768 256; do
    python train_caption.py \
        --name 'mbn_v1_025_224' \
        --train_mode 'cnn_freeze' \
        --token_type 'radix' \
        --radix_base ${i} \
        --cnn_name 'mobilenet_v1_025'  \
        --cnn_fm_attention 'Conv2d_13_pointwise' \
        --cnn_fm_projection 'tied' \
        --rnn_name 'LSTM' \
        --attn_num_heads 8 \
        --checkpoint_path "${CNN_CKPT}" \
        --dataset_dir ${DSET:-''} \
        --dataset_file_pattern ${DSET_PATTERN:-''} \
        --log_root ${LOG_ROOT:-''} \
        --gpu ${GPU} \
        --run 1

    python train_caption.py \
        --name 'mbn_v1_025_224' \
        --train_mode 'cnn_finetune' \
        --token_type 'radix' \
        --radix_base ${i} \
        --cnn_name 'mobilenet_v1_025'  \
        --cnn_fm_attention 'Conv2d_13_pointwise' \
        --cnn_fm_projection 'tied' \
        --rnn_name 'LSTM' \
        --attn_num_heads 8 \
        --checkpoint_path "${LOG_ROOT}/mscoco_v3/radix_b${i}_w256_LSTM_r512_h8_tie_mbn_v1_025_224/run_01" \
        --dataset_dir ${DSET:-''} \
        --dataset_file_pattern ${DSET_PATTERN:-''} \
        --log_root ${LOG_ROOT:-''} \
        --gpu ${GPU} \
        --run 1

    python train_caption.py \
        --name 'mbn_v1_025_224' \
        --train_mode 'scst' \
        --token_type 'radix' \
        --radix_base ${i} \
        --cnn_name 'mobilenet_v1_025'  \
        --cnn_fm_attention 'Conv2d_13_pointwise' \
        --cnn_fm_projection 'tied' \
        --rnn_name 'LSTM' \
        --attn_num_heads 8 \
        --scst_weight_bleu '0,0,0,2'  \
        --checkpoint_path "${LOG_ROOT}/mscoco_v3/radix_b${i}_w256_LSTM_r512_h8_tie_mbn_v1_025_224_cnnFT/run_01" \
        --dataset_dir ${DSET:-''} \
        --dataset_file_pattern ${DSET_PATTERN:-''} \
        --log_root ${LOG_ROOT:-''} \
        --gpu ${GPU} \
        --run 1
done

##############################################################################

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

##############################################################################

