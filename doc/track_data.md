# Processed track data

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

## Extracted data fields
Processed dataset fields saved in "./interaction-dataset-master/processed_trackfiles/" for car-following beahvior modeling. 
* scenario: traffic scenario, e.g., DR_CHN_Merging_ZS
* record_id: id of the recorded track file in a scenario, e.g. 007
* track_id: id of the tracked vehicle
* frame_id: id of the video frame
* lane_left_label: user manual label of left lane
* lane_left_type: lanelet way type of left lane
* lane_left_subtype: lanelet way subtype of left lane
* lane_left_way_id: lanelet way id of left lane
* lane_left_min_dist: minimum distance from the centroid of the vehicle to the closest line segment of left lane
* lane_right_label: user manual label of right lane
* lane_right_type: lanelet way type of right lane
* lane_right_subtype: lanelet way subtype of right lane
* lane_right_way_id: lanelet way id of right lane
* lane_right_min_dist: minimum distance from the centroid of the vehicle to the closest line segment of right lane
* lane_label_diff: difference between left and right lane labels in [-1, 0, 1], used for lane identification sanity check
* lane_label_avg: average of left and right lane labels, used to identify vehicles in adjacent lanes
* lead_track_id: track_id of lead vehicle, the closest vehicle ahead of the ego vehicle in the same lane
* car_follow_eps: id of car following episode, defined as the episode without switching lanes or lead vehicle
* ax: acceleration in the x direction derived from vx in the raw dataset using np.gradient
* ay: acceleration in the y direction derived from vx in the raw dataset using np.gradient
* x_kf: kalman filtered x position
* y_kf: kalman filtered y position
* vx_kf: kalman filtered x velocity
* vy_kf: kalman filtered y velocity
* ax_kf: kalman filtered x acceleration
* ay_kf: kalman filtered y acceleration

# Pytorch datasets
1. EgoDataset: This dataset returns a dictionary with the following fields:
    * meta: array with the following fields: ["scenario", "record_id", "track_id", "agent_type"]
    * ego: array of ego observation with dimension [num_frames, obs_dim] with the following fields: ["x", "y", "vx", "vy", "psi_rad", "length", "width", "track_id", "lane_left_type", "lane_left_min_dist", "lane_right_type", "lane_right_min_dist"]
    * agents: array of agent observation with dimension [num_frames, num_agents, obs_dim] with the following fields: ["x", "y", "vx", "vy", "psi_rad", "length", "width", "track_id", "is_lead", "is_left", "dist_to_ego"]. In frames with fewer than num_agents number of agents, the corresponding array values are set to -1.
    * act: array of ego action with dimension [num_frames, num_act] with fields: ["ax", "ay"]

2. SimpleEgoDataset: Subclass of EgoDataset with a single agent being the lead vehicle, i.e., num_agents=1. 

3. RelativeDataset: Subclass of EgoDataset with a single agent and relative observations. Relevant observations are converted to the ego coordinate defined by the ego heading. This dataset contains the follow ego fields:
    * vx_ego: x velocity in ego coordinate
    * vy_ego: y velocity in ego coordinate 
    * lane_left_min_dist: same in extracted data fileds
    * lane_right_min_dist: same in extracted data fileds
    * x_rel_ego: relative x distance to lead vehicle in ego coordinate
    * y_rel_ego:relative x distance to lead vehicle in ego coordinate
    * vx_rel_ego: relative x velocity to lead vehicle in ego coordinate
    * vy_rel_ego: relative y velocity to lead vehicle in ego coordinate
    * psi_rad_rel: relative heading to lead vehicle

# Preprocess FAQ
1. How do you identify lead vehicles?
    * After we identify a vehicle's lane position using the method described in [here](./lanelet.md), we calculate the minimum distance from the vehicle to each of the way segment in the lanelet map. The distance is sign so that a positive distance is to the left and negative to the right. For each sign, we choose the lane with the smallest absolute distance to be the current lane lines. This gives us a left and a right lane. 
2. How do you identify car-following episodes? 
    * After we identify the lane position of each vehicle, we take the average of the left and right lane label as the lane number of the vehicle. If the absolute difference between two vehicles' average lane label is smaller than one, and the distance are smaller than a predefined threshold (e.g., 50m), we define them as neighboring vehicles. Of all neighboring vehicles, the one in front (i.e., with cardinal direction between pi and -pi) with the smallest distance to the ego vehicle is defined as the lead vehicle. 