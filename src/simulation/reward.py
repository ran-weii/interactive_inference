
class CarfollowingReward:
    """
    Reward features:
    0 lane deviation: -(d - d_target)^2, d_target = 0 m
    1 relative distance: 0 * (s_rel >= min_s_rel) - s_rel_penalty * (s_rel < min_s_rel)
    2 looming: -(loom - loom_target)^2, loom_target = 0 
    """
    def __init__(self, feature_set):
        self.feature_set = feature_set
        self.idx_d = [i for i, f in enumerate(self.feature_set) if f == "d"][0]
        self.idx_s_rel = [i for i, f in enumerate(self.feature_set) if f == "s_rel"][0]
        self.idx_loom = [i for i, f in enumerate(self.feature_set) if f == "loom_s"][0]

        self.d_target = 0.
        self.min_s_rel = 3.
        self.s_rel_penalty = 10.
        self.loom_target = 0.

    def __call__(self, obs, ctl):
        d = obs[self.idx_d]
        s_rel = obs[self.idx_s_rel]
        loom_s = obs[self.idx_loom]

        f1 = -(d - self.d_target) ** 2
        f2 = (s_rel >= self.min_s_rel) * 0 - self.s_rel_penalty * (s_rel < self.min_s_rel)
        f3 = -(loom_s - self.loom_target) ** 2
        r = f1 + f2 + 10 * f3
        return r