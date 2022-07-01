import torch
import numpy as np
from scipy.stats import norm
from src.simulation.dynamics import ConstantAcceleration
from src.simulation.observers import RelativeObserver
from src.map_api.frenet_utils import compute_normal_from_kappa
from src.map_api.frenet_utils import compute_acceleration_vector
from src.data.geometry import wrap_angles, coord_transformation, angle_to_vector

class CEM:
    """ Cross entropy method model predictive control """
    def __init__(
        self, model, act_dim, num_samples=50, topk=3, horizon=6, num_iters=20
        ):
        self.model = model
        self.act_dim = act_dim
        self.num_samples = num_samples
        self.topk = topk
        self.horizon = horizon
        self.num_iters = num_iters

    def __call__(self, state, verbose=False):
        action_mean, action_std = 0, 1
        history = {"actions":[], "states":[], "rewards":[]}
        for i in range(self.num_iters):
            actions = action_mean + action_std * np.random.normal(0, 1, 
                size=(self.horizon, self.num_samples, self.act_dim)
            )
            states, rewards = self.rollout(state, actions)
            action_mean, action_std = self.fit_gaussian(actions, rewards)
            if verbose:
                print("{0} rewards mean {1:.2f} std {2:.2f}".format(
                    i+1, rewards.mean(), rewards.std()
                ))
            history["actions"].append(np.expand_dims(actions, 0))
            history["states"].append(np.expand_dims(states, 0))
            history["rewards"].append(np.expand_dims(rewards, 0))
        return action_mean[0][0]

    def rollout(self, state, actions):
        """ Rollout environment dynamics model with input actions
        
        Args:
            state (np.array): [state_dim]
            actions (np.array): [T, num_samples, act_dim]

        Returns:
            states (np.array): [T, num_samples, state_dim]
            rewards (np.array): [num_samples]
        """
        T = self.horizon + 1
        states = [np.empty(0)] * T
        rewards = [np.empty(0)] * (T - 1)
        states[0] = state.reshape(1, -1).repeat(self.num_samples, 0)
        for t in range(self.horizon):
            states[t+1], rewards[t] = self.model(states[t], actions[t])
        states = np.stack(states)[1:]
        rewards = np.stack(rewards).sum(0)
        return states, rewards
    
    def fit_gaussian(self, actions, rewards):
        rewards = np.nan_to_num(rewards, nan=0, posinf=1e8, neginf=-1e8)
        topk = np.argsort(rewards)[::-1][:self.topk]
        top_actions = actions[:, topk]
        action_mean, action_std = (
            top_actions.mean(1, keepdims=True),
            top_actions.std(1, keepdims=True))
        return action_mean, action_std


class LaneMonitor:
    """ Dynamics model for lateral control """
    def __init__(self, map_data):

        self.map_data = map_data
        self.dynamics = ConstantAcceleration()
        self.observer = RelativeObserver(map_data)

    def __call__(self, state, act):
        next_state = np.zeros_like(state)
        reward = np.zeros(len(state))
        for i in range(len(state)):
            vel = state[i, 2:4]

            # assume constant longitudinal acceleration by agent
            act_agent = state[i, -1]
            act_env = self.observer.ego_to_glob(act_agent, act[i], vel[0], vel[1])
            act_env = np.concatenate(act_env, axis=-1)
            
            state_action = np.hstack([state[i, :4], act_env]).reshape(-1, 1)
            next_state_i = self.dynamics.step(state_action).reshape(-1)
            next_state[i] = np.hstack([next_state_i[:4], act_agent])
            
            _, _, center_line_dist = self.observer.get_lane_position(next_state[i][0], next_state[i][1])
            reward[i] = norm(0, 0.01).pdf(center_line_dist)
        return next_state, reward


class Stanley:
    """ Stanley path tracking controller """
    def __init__(self, map_data, k=1):
        self.map_data = map_data
        self.k = k
        self.target_lane_id = None

    def __call__(self, state):
        target_lane_id = None
        if self.target_lane_id is not None:
            target_lane_id = self.target_lane_id

        (lane_id, _, _, _, center_line_dist, cell_headings) = self.map_data.match(
            state[0], state[1], target_lane_id=target_lane_id, max_cells=1
        )
        if self.target_lane_id is None:
            self.target_lane_id = lane_id
        
        vx, vy, v_norm = state[2], state[3], np.linalg.norm(state[2:4])

        # heading error
        heading = np.arctan2(vy, vx)
        heading_error = wrap_angles(heading - cell_headings[0, 1])

        ctl_heading = -heading_error - np.arctan2(self.k * center_line_dist, 1e-6 + v_norm)
        ctl_x, ctl_y = coord_transformation(v_norm, 0, 0, 0, theta=ctl_heading, inverse=True)
        return np.array([ctl_y])
        

class AuxiliaryController:
    """ Provide auxiliary control for agent
    
    aux longitudinal control: implements ego longitudinal control from dataset
    aux lateral control: assume constant longitudinal control generated by the agent,
        find lateral control to minimize lane deviation
    """
    def __init__(self, map_data, ctl_direction, method="stanley"):
        assert ctl_direction in ["lon", "lat", "none"]
        assert method in ["cem", "stanley"]
        self.ctl_direction = ctl_direction
        if method == "stanley":
            self.controller = Stanley(map_data, k=1)
        elif method == "cem":
            self.controller = CEM(LaneMonitor(map_data), 1)

    def choose_action(self, obs, ctl_agent, ctl_data):
        ctl_agent = ctl_agent.data.numpy().reshape(-1)
        if self.ctl_direction == "lon":
            ctl_lon, _ = coord_transformation(
                ctl_data[0], 0, obs["ego"][2], obs["ego"][3]
            )
            ctl = torch.tensor([ctl_lon]).to(torch.float32)
        elif self.ctl_direction == "lat":
            state = np.concatenate([obs["ego"][:4], ctl_agent[:1]], axis=-1)
            ctl = self.controller(state)
            ctl = torch.from_numpy(ctl).to(torch.float32)
        else:
            ctl = None
        return ctl

class AgentWrapper:
    """ Wrapper for agent and observer for simulation """
    def __init__(self, observer, agent, action_set, sample_method):
        """
        Args:
            observer (Observer): observer object to compute features
            agent (Agent): Agent object with choose_action method
            action_set (list): action set for whether to use ego or frenet actions
            sample_method (str): agent sample method. Choices=["ace", "acm", "bma"]
        """
        assert sample_method in ["ace", "acm", "bma"]
        agent.eval()
        self.agent = agent
        self.observer = observer
        self.action_set = action_set
        self.sample_method = sample_method
        
        self._prev_act = None
    
    def reset(self):
        self.agent.reset()
        self._prev_act = None # torch.zeros(1, 2) for legacy code, make a copy for legacy?
    
    def ego_action_to_global(self, ax, ay, obs_env):
        """ Convert actions from ego frame to global frame 
        
        Args:
            ax (float): x acceleration in ego frame
            ay (float): y acceleration in ego frame
            obs_env (np.array): environment observation vector with 
                ego observation [x, y, vx, vy, psi, kappa]

        Returns:
            act_env (np.array): action in global frame. size=[2]
        """
        psi = obs_env["ego"][4]
        ax_env, ay_env = coord_transformation(ax, ay, None, None, theta=psi)
        act_env = np.array([ax_env, ay_env])
        return act_env
    
    def frenet_action_to_global(self, dds, ddd, obs_env):
        """ Convert actions from frenet frame to global frame 
        
        Args:
            dds (float): s acceleration in frenet frame
            ddd (float): d acceleration in frenet frame
            frenet_state (np.array): frenet state vector

        Returns:
            act_env (np.array): action in global frame. size=[2]
        """
        ref_path = self.observer._ref_path
        s_condition = self.observer._s_condition_ego
        d_condition = self.observer._d_condition_ego
        s_condition = np.hstack([s_condition, np.array([dds])])
        d_condition = np.hstack([d_condition, np.array([ddd])])
        cartesian_state = ref_path.frenet_to_cartesian(
            s_condition, d_condition
        )
        
        [x, y, v, a, theta, kappa] = cartesian_state
        norm = compute_normal_from_kappa(theta, kappa)
        tan_vec = angle_to_vector(theta)
        norm_vec = angle_to_vector(norm)
        act_env = compute_acceleration_vector(
            a, v, kappa, tan_vec, norm_vec
        ).flatten()
        return act_env

    def choose_action(self, obs_env):
        """
        Args:
            obs_env (dict): enviroinment observation {"ego", "agents"}
        
        Returns:
            act_env (np.array): actions in the global coordinate [2]
        """
        obs = self.observer.observe(obs_env) # torch.tensor
        with torch.no_grad():
            act = self.agent.choose_action(
                obs, self._prev_act, sample_method=self.sample_method, num_samples=1
            ).view(1, 2)
            self._prev_act = act.clone()
        
        # convert action to global frame
        [ax, ay] = act.numpy().flatten()
        if self.action_set[0] == "ax_ego":
            act_env = self.ego_action_to_global(ax, ay, obs_env)
        else:
            act_env = self.frenet_action_to_global(ax, ay, obs_env)
        return act_env
