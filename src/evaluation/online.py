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


def eval_episode(env, controller, eps_id, max_steps=1000, callback=None):
    """ Evaluate episode
    
    Args:
        env (Env): gym style simulator
        controller (): controller with reset and choose_action method
        eps_id (int): episode id
        max_steps (int, optional): maximum number of steps
        callback (class, optional): controller callback

    Returns:
        sim_states (np.array): simulated states [T, num_agents, 5]
        sim_acts (np.array): simulated actions [T, 2]
        track_data (dict): recorded track data
        callback (class): updated callback. return if callback is not None
    """
    controller.reset()
    obs = env.reset(eps_id)
    for t in range(max_steps):
        act = controller.choose_action(obs)
        if callback is not None:
            callback(controller)
        obs, _, done, _ = env.step(act)
        if done or t >= max_steps:
            break
    
    sim_states = env._sim_states[:t+1]
    sim_acts = env._sim_acts[:t+1]
    track_data = {k:v[:t+1] for (k, v) in env._track_data.items()}
    return sim_states, sim_acts, track_data