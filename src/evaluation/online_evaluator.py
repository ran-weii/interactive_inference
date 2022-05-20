class Evaluator:
    def __init__(self):
        self._track = {
            "lane_center_dist": [],
            "is_in_lane": [],
            "x_rel": [],
            "vx_rel": [],
            "loom_x": []
        }

    def observe(self, obs, ctl):
        left_bound_dist = obs[0, 2]
        right_bound_dist = obs[0, 3]
        lane_width = left_bound_dist + right_bound_dist
        lane_center_dist = left_bound_dist - lane_width / 2
        is_in_lane = 1 if left_bound_dist > 0 and right_bound_dist > 0 else 0
        self._track["lane_center_dist"].append(lane_center_dist.item())
        self._track["is_in_lane"].append(is_in_lane)
        self._track["x_rel"].append(obs[0, 4].item())
        self._track["vx_rel"].append(obs[0, 6].item())
        self._track["loom_x"].append(obs[0, 9].item())