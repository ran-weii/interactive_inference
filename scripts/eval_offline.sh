#! /bin/bash
python eval_offline.py \
--filenames "vehicle_tracks_000.csv" \
--test_lanes "3,4" \
--agent vin \
--exp_name "11-08-2022 03-34-39" \
--min_eps_len 100 \
--max_eps_len 500 \
--batch_size 10 \
--num_samples 30 \
--sample_method ace \
--seed 0 \
--save False