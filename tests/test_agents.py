import torch
import torch.nn as nn
from src.agents.active_inference import ActiveInference
from src.evaluation.offline_metrics import (
    mean_absolute_error, threshold_relative_error)
from src.visualization.inspection import get_active_inference_parameters

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
    mask = torch.randint(0, 2, (T, batch_size)).to(torch.float32)
    
    """ test learning """
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
    logp_pi, logp_obs = agent(obs, u, theta=theta, inference=False)
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
    logp_pi, logp_obs = agent(obs, u, inference=False)
    loss = logp_pi.sum()
    loss.backward()
    
    assert agent.C.grad.norm() != 0
    assert agent.obs_model.mu.grad.norm() != 0
    assert agent.obs_model.lv.grad.norm() != 0
    assert agent.ctl_model.mu.grad.norm() != 0
    assert agent.ctl_model.lv.grad.norm() != 0
    assert agent.hmm.B.grad.norm() != 0
    assert agent.tau.grad.norm() != 0
    
    """ test inference """
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
    with torch.no_grad():
        G, b = agent(obs, u, theta=theta, inference=True)
        ctl = agent.choose_action(obs, u, theta=theta)
        
        speed = obs[:, :, 0].unsqueeze(-1).numpy()
        mae = mean_absolute_error(
            u.numpy(), ctl.numpy(), mask=mask.numpy(), 
            speed=speed, cumulative=False
        )
        tre = threshold_relative_error(
            u.numpy(), ctl.numpy(), mask=mask.numpy(), alpha=0.1
        )
        
    assert list(ctl.shape) == [T, batch_size, ctl_dim]
    assert list(mae.shape) == [ctl_dim]
    assert list(tre.shape) == [ctl_dim]
    
    # test with self parameters
    agent = ActiveInference(state_dim, act_dim, obs_dim, ctl_dim, H)
    with torch.no_grad():
        G, b = agent(obs, u, inference=False)
        ctl = agent.choose_action(obs, u)
        
        speed = obs[:, :, 0].unsqueeze(-1).numpy()
        mae = mean_absolute_error(
            u.numpy(), ctl.numpy(), mask=mask.numpy(), 
            speed=speed, cumulative=False
        )
        tre = threshold_relative_error(
            u.numpy(), ctl.numpy(), mask=mask.numpy(), alpha=0.1
        )
        
    assert list(ctl.shape) == [T, batch_size, ctl_dim]
    assert list(mae.shape) == [ctl_dim]
    assert list(tre.shape) == [ctl_dim]
    
    print("test_active_inference_agent passed")

""" TODO: get parameters in batch """
def test_get_active_inference_parameters():
    torch.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)
    
    state_dim = 10
    obs_dim = 12
    act_dim = 5
    ctl_dim = 3
    H = 15
    
    agent = ActiveInference(state_dim, act_dim, obs_dim, ctl_dim, H)
    
    theta_dict = get_active_inference_parameters(agent)
    
    assert list(theta_dict["A_mu"].shape) == [state_dim, obs_dim]
    assert list(theta_dict["A_sd"].shape) == [state_dim, obs_dim]
    assert list(theta_dict["B"].shape) == [act_dim, state_dim, state_dim]
    assert list(theta_dict["C"].shape) == [state_dim]
    assert list(theta_dict["D"].shape) == [state_dim]
    assert list(theta_dict["F_mu"].shape) == [act_dim, ctl_dim]
    assert list(theta_dict["F_sd"].shape) == [act_dim, ctl_dim]
    
    print("test_get_active_inference_parameters passed")

if __name__ == "__main__":
    test_active_inference_agent()
    test_get_active_inference_parameters()