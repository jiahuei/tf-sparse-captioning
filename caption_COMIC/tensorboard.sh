#!/bin/bash



log_root='/home/jiahuei/Documents/1_TF_files/prune/mscoco_v3'


log_dir='word_w256_LSTM_r512_h1_ind_xu_magGradUniform_sps_0.975_FT_prune_fromNone_cnnFT'

# source /home/jiahuei/tf-1_9/bin/activate
export CUDA_VISIBLE_DEVICES=''


tensorboard --logdir="${log_root}/${log_dir}" --host 0.0.0.0








