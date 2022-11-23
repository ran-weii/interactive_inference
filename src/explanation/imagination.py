from copy import deepcopy

import torch
import torch.nn.functional as F

class AgentSimulator:
    """ Agent imagination simulator """
    def __init__(self, agent, z=None):
        self.state_dim = agent.state_dim
        self.act_dim = agent.act_dim
        self.z = z

        self.agent = deepcopy(agent)

        with torch.no_grad():
            self.transition = self.agent.rnn.compute_transition(z=z).squeeze(0)

    def reset(self, s0):
        self.t = 0
        self.s = s0
        o = self.sample_obs(self.s)
        return o

    def step(self, u):
        pi_a = torch.softmax(self.agent.ctl_model.log_prob(u), dim=-1).view(-1)
        a = torch.multinomial(pi_a, 1)[0]
        s_next = torch.multinomial(self.transition[a][self.s], 1)[0]
        o_next = self.sample_obs(s_next)
        
        self.s = s_next
        return o_next

    def sample_obs(self, s):
        with torch.no_grad():
            o_ = self.agent.obs_model.sample((1,), z=self.z).view(self.state_dim, -1)
        o = o_[s]
        return o

def simulate_imagination(agent, s0, max_steps, z=None, sample_method="acm"):
    """ Simulate agent in imagination and decode to observation and control space
    
    Args:
        agent (VINAgent): vin agent object
        s0 (int): index of the initial state
        max_steps (int): maximum number of simulation steps
        z (torch.tensor): latent variable. Default=None
        sample_method (str): agent action selection method. choices=["acm", "ace"]

    Returns:
        data (dict): data dict with fields ["s", "o", "u", "b", "pi"]
    """
    env = AgentSimulator(agent, z=z)
    agent.reset(z=z)
    o = env.reset(s0)

    data = {"s": [s0], "o": [o.data.flatten()], "u": [], "b": [], "pi": []}
    for t in range(max_steps):
        with torch.no_grad():
            u = agent.choose_action(o.view(1, -1), sample_method=sample_method)[0].view(1, -1)

        o_next = env.step(u)
        
        data["s"].append(env.s.clone().data.item())
        data["o"].append(o_next.clone().data.flatten())
        data["u"].append(u.clone().data.flatten())
        data["b"].append(agent._state["b"].data.view(-1))
        data["pi"].append(agent._state["pi"].data.view(-1))

        o = o_next
    
    data["s"] = torch.tensor(data["s"])
    data["o"] = torch.stack(data["o"])
    data["u"] = torch.stack(data["u"])
    data["b"] = torch.stack(data["b"])
    data["pi"] = torch.stack(data["pi"])
    data["pi_s"] = F.one_hot(data["s"], num_classes=agent.state_dim)
    return data