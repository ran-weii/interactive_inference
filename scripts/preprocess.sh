#! /bin/bash
python preprocess.py \
--scenario DR_CHN_Merging_ZS \
--filename vehicle_tracks_000.csv \
--task train_labels \
--cell_len 10 \
--min_seg_len 50 \
--train_ratio 0.7 \
--parallel False \
--num_cores 10 \
--seed 0 \
--save True \
--debug False
