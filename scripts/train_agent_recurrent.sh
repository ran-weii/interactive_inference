#! /bin/bash
python train_agent_recurrent.py \
--filenames "vehicle_tracks_000.csv" \
--valid_lanes "3,4" \
--checkpoint_path "none" \
--feature_set "lv_s_rel,lv_ds_rel,lv_inv_tau" \
--action_set "dds_smooth" \
--agent vin \
--state_dim 20 \
--act_dim 15 \
--horizon 30 \
--obs_model flow \
--obs_cov diag \
--ctl_cov diag \
--hmm_rank 10 \
--alpha 1. \
--beta 0. \
--rwd efe \
--detach False \
--hyper_dim 4 \
--hyper_cov False \
--train_prior False \
--hidden_dim 30 \
--num_hidden 2 \
--gru_layers 1 \
--activation relu \
--norm_obs True \
--train_mode marginal \
--bptt_steps 500 \
--pred_steps 5 \
--bc_penalty 1. \
--obs_penalty 0.2 \
--pred_penalty 0.2 \
--reg_penalty 0.1 \
--post_obs_penalty 1. \
--kl_penalty 1. \
--min_eps_len 50 \
--max_eps_len 500 \
--train_ratio 0.7 \
--batch_size 100 \
--epochs 30 \
--lr 0.01 \
--lr_flow 0.001 \
--lr_post 0.005 \
--decay 3e-5 \
--grad_clip 100 \
--decay_steps 200 \
--decay_rate 1. \
--cp_every 50 \
--seed 0 \
--save False