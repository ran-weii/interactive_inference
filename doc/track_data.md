# Description of track data

## Raw data fields
Raw dataset fields provided by the INTERACTION dataset [website](https://interaction-dataset.com/details-and-format).
* track_id: id of the tracked vehicle. For each vehicle_tracks_xxx.csv file track_id starts from 1.
* frame_id: id of the video frame. For each track, frame_id starts from 1.
* timestamp_ms: time the agent appeared in the video at interval 100ms. Unit in ms.
* agent_type: type of tracked agent. 
* x: x position of agent. Unit in m.
* y: y position of agent. Unit in m.
* vx: x velocity of agent. Unit in m/s.
* vy: y velocity of agent. Unit in m/s.
* psi_rad: heading angle of agent in the map coordinate. For static agents psi_rad != vy/vx. Unit in rad.
* length: length of agent. Unit in m.
* width: width of agent. Unit in m.

## Extracted fields
The following fields are extracted from the preprocessing step. 

### Features
These fields are stored in the ``features`` folder:
* seg_id: id of the current drive segment 
* seg_len: length of the current drive segment
* ego_d: ego lane offset
* ego_ds: ego longitudinal speed
* ego_dd: ego lateral speed
* ego_psi_error: ego heading error
* ego_kappa: ego lane curvature
* ego_lane_id: ego lane id
* lv_s_rel: lead vehicle relative distance
* lv_ds_rel: lead vehicle relative speed
* lv_inv_tau: lead vehicle inverse tau
* lv_d: lead vehicle lane offset
* lv_dd: lead vehicle lateral speed
* lv_track_id: lead vehicle track id
* lidar_range_{k}: range measurement of the kth lidar beam
* lidar_range_rate_{k}: range rate measurement of the kth lidar beam
* dds: ego frenet longitudinal acceleration (inaccurate do not use)
* ddd: ego frenet lateral acceleration
* kappa: ego trajectory curvature
* norm: ego trajectory normal vector direction in radians
* a: signed ego acceleration norm. sign is positive if in the same direction as heading
* ax: ego cartesian x accelection
* ay: ego cartesian y accelection
* dds_norm: smoothed ego frenet longitudinal acceleration computed by differentiating ego_ds

### Train labels
These fields are stored in the ``train_labels`` folder:
* is_tail: whether the current step is the tail of a drive segment
* is_tail_merging: Whether the current step is the tail of a drive segment and the vehicle is merging
* is_head: whether the current step is the head of a drive segment
* is_head_merging: whether the current step is the head of a drive segment and the vehicle is merging
* eps_id: episode id assigned to the current step
* eps_len: length of the assigned episode
* is_train: whether the current step is assigned to the training set