import numpy as np
import torch
from src.data.geometry import wrap_angles, coord_transformation

class RelativeObserver:
    def __init__(self, map_data):
        self.map_data = map_data
        self.obs_dim = 10
        self.ctl_dim = 2

        self.reset()
    
    def reset(self):
        self.t = 0
        self.target_lane_id = None
    
    def glob_to_ego(self, x, y, vx, vy):
        return coord_transformation(x, y, vx, vy, inverse=False)

    def ego_to_glob(self, x, y, vx, vy):
        return coord_transformation(x, y, vx, vy, inverse=True)
    
    def get_lane_position(self, x, y):
        target_lane_id = None if self.target_lane_id is None else self.target_lane_id
        (lane_id, cell_id, left_bound_dist, right_bound_dist, 
        center_line_dist, _) = self.map_data.match(
            x, y, target_lane_id=target_lane_id
        )
        self.target_lane_id = lane_id
        return left_bound_dist, right_bound_dist, center_line_dist

    def observe(self, obs):
        obs_ego = obs["ego"]
        obs_agent = obs["agents"][0]
        
        rel_obs = obs_agent - obs_ego
        psi_rel = wrap_angles(rel_obs[4])
        
        [x, y, vx, vy, psi] = obs_ego[:5].tolist()
        [x_rel, y_rel, vx_rel, vy_rel] = rel_obs[:4].tolist()

        left_bound_dist, right_bound_dist, _ = self.get_lane_position(x, y)

        # convert observations to ego centric
        vx_ego, vy_ego = self.glob_to_ego(vx, vy, vx, vy)
        x_rel, y_rel = self.glob_to_ego(x_rel, y_rel, vx, vy)
        vx_rel, vy_rel = self.glob_to_ego(vx_rel, vy_rel, vx, vy)

        loom_x = vx_rel / (x_rel + 1e-6)
        
        obs = np.hstack([
            vx_ego, vy_ego, left_bound_dist, right_bound_dist,
            x_rel, y_rel, vx_rel, vy_rel, psi_rel, loom_x
        ])
        obs = torch.from_numpy(obs).view(1, -1).to(torch.float32)
        
        self.t += 1
        return obs

    def control(self, ctl, obs):
        """ Project ego centric control to world coordinate """
        vx, vy = obs["ego"][2], obs["ego"][3]
        [ax_ego, ay_ego] = ctl.data.view(-1).numpy().tolist()
        ax, ay = self.ego_to_glob(ax_ego, ay_ego, vx, vy)
        ctl = np.hstack([ax, ay])
        return ctl