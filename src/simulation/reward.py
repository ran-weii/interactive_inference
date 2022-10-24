import numpy as np

class CarfollowingReward:
    """
    Reward features:
    0 lane deviation: -(d - d_target)^2, d_target = 0 m
    1 relative distance: 0 * (s_rel >= min_s_rel) - s_rel_penalty * (s_rel < min_s_rel)
    2 looming: -(loom - loom_target)^2, loom_target = 0 
    """
    def __init__(self, sensors):
        sensor_names = [s.__class__.__name__ for s in sensors]
        ego_sensor_idx = sensor_names.index("EgoSensor")
        lv_sensor_idx = sensor_names.index("LeadVehicleSensor")

        self.idx_d = sensors[ego_sensor_idx].feature_names.index("ego_d")
        self.idx_s_rel = sensors[lv_sensor_idx].feature_names.index("lv_s_rel")
        self.idx_loom = sensors[lv_sensor_idx].feature_names.index("lv_inv_tau")

        self.d_target = 0.
        self.min_s_rel = 3.
        self.s_rel_penalty = 10.
        self.loom_target = 0.

    # def __call__(self, sensor_obs, ctl):
    #     ego_obs = sensor_obs["EgoSensor"]
    #     lv_obs = sensor_obs["LeadVehicleSensor"]
    #     d = ego_obs[self.idx_d]
    #     s_rel = lv_obs[self.idx_s_rel]
    #     loom_s = lv_obs[self.idx_loom]

    #     f1 = -(d - self.d_target) ** 2
    #     f2 = (s_rel >= self.min_s_rel) * 0 - self.s_rel_penalty * (s_rel < self.min_s_rel)
    #     f3 = -(loom_s - self.loom_target) ** 2
    #     r = f1 + f2 + 10 * f3
    #     return r

    def __call__(self, sim_state, sensor_obs, ctl):
        ego_state = sim_state["ego_state"]
        ego_true_state = sim_state["ego_true_state"]
        
        # compute displacement error
        r = np.sqrt(np.sum((ego_state[:2] - ego_true_state[:2])**2))
        return r