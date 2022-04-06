import numpy as np
import torch
from src.data.geometry import wrap_angles, vector_projection

class RelativeObserver:
    def __init__(self, map_data):
        self.map_data = map_data

    def observe(self, obs):
        obs_ego = obs["ego"]
        obs_agent = obs["agents"][0]
        
        rel_obs = obs_agent - obs_ego
        psi_rel = wrap_angles(rel_obs[4])

        [x_ego, y_ego, vx_ego, vy_ego, psi_ego] = obs_ego[:5].tolist()
        lane_id, cell_id, left_bound_dist, right_bound_dist, _ = self.map_data.match(x_ego, y_ego)
        
        # out of lane handle
        left_bound_dist = 10 if left_bound_dist is None else left_bound_dist
        right_bound_dist = 10 if right_bound_dist is None else right_bound_dist

        # convert observations to ego centric
        psi_x = np.cos(psi_ego)
        psi_y = np.sin(psi_ego)
        
        [x_rel, y_rel, vx_rel, vy_rel] = rel_obs[:4].tolist()
        vx_ego, vy_ego = vector_projection(vx_ego, vy_ego, psi_x, psi_y)
        x_rel, y_rel = vector_projection(x_rel, y_rel, psi_x, psi_y)
        vx_rel, vy_rel = vector_projection(vx_rel, vy_rel, psi_x, psi_y)
        
        loom_x = vx_rel / (x_rel + 1e-6)
        
        obs = np.hstack([
            vx_ego, vy_ego, left_bound_dist, right_bound_dist,
            x_rel, y_rel, vx_rel, vy_rel, psi_rel, loom_x
        ])
        obs = torch.from_numpy(obs).to(torch.float32)
        return obs

    def control(self, ctl, obs):
        """ Project ego centric control to world coordinate """
        psi_ego = obs["ego"][4]
        [ax_ego, ay_ego] = ctl.data.view(-1).numpy().tolist()
        ax = ax_ego * np.cos(psi_ego) - ay_ego * np.sin(psi_ego)
        ay = ax_ego * np.sin(psi_ego) + ay_ego * np.cos(psi_ego)
        ctl = np.hstack([ax, ay])
        return ctl