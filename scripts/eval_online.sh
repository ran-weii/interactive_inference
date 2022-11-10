#! /bin/bash
python eval_online.py \
--filename "vehicle_tracks_000.csv,vehicle_tracks_003.csv,vehicle_tracks_007.csv" \
--test_lanes "0,2" \
--agent vin \
--exp_name "11-08-2022 21-53-28" \
--min_eps_len 50 \
--max_eps_len 500 \
--num_eps 100 \
--sample_method acm \
--playback False \
--test_on_train False \
--test_posterior False \
--seed 0 \
--save_summary True \
--save_data False \
--save_video False