import torch
import torch.nn as nn
from src.agents.active_inference import ActiveInference

def test_active_inference_agent():
    torch.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)
    
    state_dim = 10
    obs_dim = 12
    act_dim = 5
    ctl_dim = 3
    H = 15
    
    # make synthetic data
    batch_size = 32
    T = 24
    obs = torch.randn(T, batch_size, obs_dim)
    u = torch.randn(T, batch_size, ctl_dim)
    mask = torch.zeros(batch_size, T)
    
    # test with supplied parameters
    agent = ActiveInference(state_dim, act_dim, obs_dim, ctl_dim, H)
    theta = {
        "A": nn.Parameter(torch.randn(batch_size, sum(agent.obs_model.parameter_size))),
        "B": nn.Parameter(torch.randn(batch_size, agent.hmm.parameter_size[0])),
        "C": nn.Parameter(torch.randn(batch_size, state_dim)),
        "D": nn.Parameter(torch.randn(batch_size, state_dim)),
        "F": nn.Parameter(torch.randn(batch_size, sum(agent.ctl_model.parameter_size))),
        "tau": nn.Parameter(torch.randn(batch_size, 1))
    }
    logp_pi, logp_obs = agent(obs, u, theta=theta)
    loss = logp_pi.sum()
    loss.backward()
    
    assert theta["A"].grad.norm() != 0
    assert theta["B"].grad.norm() != 0
    assert theta["C"].grad.norm() != 0
    assert theta["D"].grad.norm() != 0
    assert theta["F"].grad.norm() != 0
    assert theta["tau"].grad.norm() != 0
    
    assert agent.C.grad == None
    assert agent.obs_model.mu.grad == None
    assert agent.obs_model.lv.grad == None
    assert agent.ctl_model.mu.grad == None
    assert agent.ctl_model.lv.grad == None
    assert agent.hmm.B.grad == None
    assert agent.tau.grad == None
    
    # test with self parameters
    agent = ActiveInference(state_dim, act_dim, obs_dim, ctl_dim, H)
    logp_pi, logp_obs = agent(obs, u)
    loss = logp_pi.sum()
    loss.backward()
    
    assert agent.C.grad.norm() != 0
    assert agent.obs_model.mu.grad.norm() != 0
    assert agent.obs_model.lv.grad.norm() != 0
    assert agent.ctl_model.mu.grad.norm() != 0
    assert agent.ctl_model.lv.grad.norm() != 0
    assert agent.hmm.B.grad.norm() != 0
    assert agent.tau.grad.norm() != 0
    
    print("test_active_inference_agent passed")

if __name__ == "__main__":
    test_active_inference_agent()