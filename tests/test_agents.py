import torch
import torch.nn as nn
from src.agents.active_inference import ActiveInference
from src.agents.planners import value_iteration, MCVI
from src.agents.models import MLP, PopArt
from src.agents.baseline import FullyRecurrentAgent
from src.evaluation.offline_metrics import (
    mean_absolute_error, threshold_relative_error)
from src.visualization.inspection import get_active_inference_parameters

def test_value_iteration():
    torch.manual_seed(0)
    state_dim = 3
    act_dim = 2
    H = 10
    batch_size = 1

    R = torch.randn(batch_size, act_dim, state_dim)
    B = torch.softmax(
        torch.randn(batch_size, act_dim, state_dim, state_dim), dim=-1
    )
    Q = value_iteration(R, B, H)
    print("test_value_iteraction passed")

def test_mlp_model():
    state_dim = 10
    act_dim = 5
    hidden_dim = 12
    num_hidden = 2
    activation = "relu"

    mlp = MLP(state_dim, act_dim, hidden_dim, num_hidden, activation)
    
    # test static data
    batch_size = 32
    x = torch.randn(batch_size, state_dim)

    with torch.no_grad():
        out = mlp(x)
    assert list(out.shape) == [batch_size, act_dim]

    # test dynamic data
    T = 7
    batch_size = 32
    x = torch.randn(T, batch_size, state_dim)

    with torch.no_grad():
        out = mlp(x)
    assert list(out.shape) == [T, batch_size, act_dim]

    print("test_mlp_model passed")

def test_popart_layer():
    torch.manual_seed(0)
    state_dim = 10
    act_dim = 5
    popart = PopArt(state_dim, act_dim)

    # create synthetic data
    T = 12
    batch_size = 32
    x = torch.softmax(torch.randn(T, batch_size, state_dim), dim=-1)
    target = 10 * torch.randn(T, batch_size, act_dim)
    
    # test normalization
    y, y_norm = popart(x)
    
    target_norm = popart.normalize(target)
    y_new, y_norm_new = popart(x)
    assert list(target_norm.shape) == [T, batch_size, act_dim]
    assert list(y_new.shape) == [T, batch_size, act_dim]

    y_diff = y_new - y
    assert torch.all(y_diff.abs() < 1e-5)
    print("test_popart_layer passed")

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
    
    assert agent.obs_model.mu.grad == None
    assert agent.obs_model.lv.grad == None
    assert agent.ctl_model.mu.grad == None
    assert agent.ctl_model.lv.grad == None
    assert agent.rwd_model.C.grad == None
    assert agent.planner.tau.grad == None
    
    # test with self parameters
    agent = ActiveInference(state_dim, act_dim, obs_dim, ctl_dim, H)
    logp_pi, logp_obs = agent(obs, u, inference=False)
    loss = logp_pi.sum()
    loss.backward()
    
    assert agent.obs_model.mu.grad.norm() != 0
    assert agent.obs_model.lv.grad.norm() != 0
    assert agent.ctl_model.mu.grad.norm() != 0
    assert agent.ctl_model.lv.grad.norm() != 0
    assert agent.hmm.B.grad.norm() != 0 
    assert agent.rwd_model.C.grad.norm() != 0
    assert agent.planner.tau.grad.norm() != 0
    
    """ test inference """
    num_samples = 10
    
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
        out = agent(obs, u, theta=theta, inference=True)
        ctl = agent.choose_action(obs, u, batch=True, theta=theta)
        ctl_samples = agent.choose_action(obs, u, batch=True, num_samples=num_samples)
        
        speed = obs[:, :, 0].unsqueeze(-1).numpy()
        mae = mean_absolute_error(
            u.numpy(), ctl.numpy(), mask=mask.numpy(), 
            speed=speed, cumulative=False
        )
        tre = threshold_relative_error(
            u.numpy(), ctl.numpy(), mask=mask.numpy(), alpha=0.1
        )
        
    assert list(ctl.shape) == [T, batch_size, ctl_dim]
    assert list(ctl_samples.shape) == [num_samples, T, batch_size, ctl_dim]
    assert list(mae.shape) == [ctl_dim]
    assert list(tre.shape) == [ctl_dim]
    
    # test with self parameters
    agent = ActiveInference(state_dim, act_dim, obs_dim, ctl_dim, H)
    with torch.no_grad():
        out = agent(obs, u, inference=False)
        ctl = agent.choose_action(obs, u, batch=True)
        ctl_samples = agent.choose_action(obs, u, batch=True, num_samples=num_samples)
        
        speed = obs[:, :, 0].unsqueeze(-1).numpy()
        mae = mean_absolute_error(
            u.numpy(), ctl.numpy(), mask=mask.numpy(), 
            speed=speed, cumulative=False
        )
        tre = threshold_relative_error(
            u.numpy(), ctl.numpy(), mask=mask.numpy(), alpha=0.1
        )
        
    assert list(ctl.shape) == [T, batch_size, ctl_dim]
    assert list(ctl_samples.shape) == [num_samples, T, batch_size, ctl_dim]
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

def test_agent_inference():
    torch.manual_seed(0)

    state_dim = 10
    obs_dim = 12
    act_dim = 5
    ctl_dim = 3
    H = 15
    
    # make synthetic data
    batch_size = 32
    T = 7
    obs = torch.randn(T, batch_size, obs_dim) * 0.1
    ctl = torch.randn(T, batch_size, ctl_dim) * 0.1
    agent = ActiveInference(state_dim, act_dim, obs_dim, ctl_dim, H)
    agent.eval()
    
    # test batch inference
    u_batch = agent.choose_action(obs, ctl, batch=True)
    assert list(u_batch.shape) == [T, batch_size, ctl_dim]
    
    # test sequential inference
    with torch.no_grad():
        agent.reset()

        a = [torch.empty(0)] * T
        b = [torch.empty(0)] * T
        u_seq = [torch.empty(0)] * T
        u_seq[0] = agent.choose_action(obs[0], ctl[0])
        
        b[0] = agent._b.squeeze(0)
        a[0] = agent._a.squeeze(0)
        for t in range(T-1):
            u_seq[t+1] = agent.choose_action(obs[t], ctl[t])
            b[t+1] = agent._b.squeeze(0)
            a[t+1] = agent._a.squeeze(0)
        u_seq = torch.stack(u_seq)
        b = torch.stack(b)
        a = torch.stack(a)
    assert list(u_seq.shape) == [T, batch_size, ctl_dim]
    
    u_diff = u_seq - u_batch
    assert torch.all(u_diff < 1e-5)

    print("test_agent_inference passed")

def test_recurrent_agent():
    torch.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)
    
    state_dim = 10
    obs_dim = 12
    act_dim = 5
    ctl_dim = 3
    H = 15
    agent = FullyRecurrentAgent(state_dim, act_dim, obs_dim, ctl_dim, H)
    agent.eval()

    # create synthetic dataset
    batch_size = 32
    T = 15
    o = torch.randn(T, batch_size, obs_dim) * 3
    u = torch.randn(T, batch_size, ctl_dim)
    
    # test training
    with torch.no_grad():
        logp_pi, logp_obs = agent(o, u)
    assert list(logp_pi.shape) == [T, batch_size]
    assert list(logp_obs.shape) == [T, batch_size]

    # test batch inference
    num_samples = 10
    with torch.no_grad():
        ctl_avg = agent.choose_action(o, u, batch=True)
        ctl_ace = agent.choose_action(o, u, batch=True, num_samples=num_samples)
    assert list(ctl_avg.shape) == [T, batch_size, ctl_dim]
    assert list(ctl_ace.shape) == [num_samples, T, batch_size, ctl_dim]
    
    # test sequential inference
    with torch.no_grad():
        agent.reset()

        a = [torch.empty(0)] * T
        b = [torch.empty(0)] * T
        ctl_seq = [torch.empty(0)] * T
        ctl_seq[0] = agent.choose_action(o[0], u[0])
        
        b[0] = agent._b.squeeze(0)
        a[0] = agent._a.squeeze(0)
        for t in range(T-1):
            ctl_seq[t+1] = agent.choose_action(o[t], u[t])
            b[t+1] = agent._b.squeeze(0)
            a[t+1] = agent._a.squeeze(0)
        ctl_seq = torch.stack(ctl_seq)
        b = torch.stack(b)
        a = torch.stack(a)

    ctl_diff = ctl_seq - ctl_avg
    assert list(ctl_seq.shape) == [T, batch_size, ctl_dim]
    assert torch.all(ctl_diff.abs() < 1e-5)

    print("test_recurrent_agent passed")

def test_nn_planner():
    torch.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)
    state_dim = 10
    act_dim = 5
    obs_dim = 11
    ctl_dim = 3
    H = 7
    beta = 0.9
    hidden_dim = 64
    num_hidden = 2
    activation = "relu"

    # create synthetic data
    T = 12
    batch_size = 24
    b = torch.softmax(torch.randn(T, batch_size, state_dim), dim=-1)
    
    # test planner separately
    agent = ActiveInference(
        state_dim, act_dim, obs_dim, ctl_dim, H
    )
    planner = MCVI(
        agent.hmm, agent.obs_model, agent.rwd_model, beta,
        hidden_dim, num_hidden, activation,
        
    )
    print(planner)

    a = planner(b)
    loss = planner.loss(b.view(-1, state_dim))
    total_loss = torch.sum(loss)
    total_loss.backward()
    
    assert list(a.shape) == [T, batch_size, act_dim]
    assert list(loss.shape) == [T * batch_size, act_dim]
    
    # print(agent.obs_model.mu.grad.norm())
    # print(agent.obs_model.lv.grad.norm())
    # print(agent.rwd_model.C.grad.norm())
    # print(agent.hmm.B.grad.norm())

    assert agent.obs_model.mu.grad.norm() != 0
    assert agent.obs_model.lv.grad.norm() != 0
    assert agent.hmm.B.grad.norm() != 0 
    assert agent.rwd_model.C.grad.norm() != 0

    # test planner with agent
    H = 0.9
    agent = ActiveInference(
        state_dim, act_dim, obs_dim, ctl_dim, H,
        planner="mcvi"
    )
    print(agent)
    o = torch.randn(T, batch_size, obs_dim)
    u = torch.randn(T, batch_size, ctl_dim)
    logp_pi, logp_obs, _ = agent(o, u)
    planner_loss = agent.planner.loss(b.view(-1, state_dim))
    loss = logp_pi.sum() + planner_loss.sum()
    loss.backward()
    print(loss)

    print(logp_pi.shape, logp_obs.shape)
    print(agent.obs_model.mu.grad.norm())
    print(agent.obs_model.lv.grad.norm())
    print(agent.rwd_model.C.grad.norm())
    print(agent.hmm.B.grad.norm())
    print(agent.ctl_model.mu.grad.norm())
    print(agent.ctl_model.lv.grad.norm())

    print("test_nn_planner passed")

def test_generalized_free_energy():
    from src.agents.reward import GeneralizedFreeEnergy
    state_dim = 10
    act_dim = 5
    obs_dim = 12
    ctl_dim = 3
    H = 15
    agent = ActiveInference(
        state_dim, act_dim, obs_dim, ctl_dim, H
    )
    rwd_model = GeneralizedFreeEnergy(
        agent.hmm, agent.obs_model
    )
    B = torch.softmax(agent.hmm.B, dim=-1)
    rwd_model(B, B)
    print(rwd_model)
    print("test_generalized_free_energy passed")

def test_factored_value_iteration():
    torch.manual_seed(0)
    from src.distributions.models import HiddenMarkovModel, ConditionalDistribution
    from src.distributions.factored_models import FactoredHiddenMarkovModel, FactoredConditionalDistribution
    from src.agents.reward import ExpectedFreeEnergy
    from src.agents.factored_agent import factored_value_iteration, FactoredQMDP
    state_dim = 10
    act_dim = 5
    obs_dim = 12
    ctl_dim = 3
    batch_size = 24
    
    R = torch.randn([batch_size] + [act_dim]*ctl_dim + [state_dim])
    B = torch.randn([batch_size] + [act_dim]*ctl_dim + [state_dim, state_dim])
    H = 15
    
    # factored_value_iteration(R, B, H)
    # hmm = HiddenMarkovModel(state_dim, act_dim)
    hmm = FactoredHiddenMarkovModel(state_dim, act_dim, ctl_dim)
    obs_model = ConditionalDistribution(obs_dim, state_dim)
    rwd_model = ExpectedFreeEnergy(hmm, obs_model)
    planner = FactoredQMDP(hmm, obs_model, rwd_model, H)
    # planner.plan()

    # b = torch.softmax(torch.randn(batch_size, state_dim), dim=-1)
    # pi = planner(b)
    # print(pi.shape)
    print("test_factored_value_iteration passed")

def test_factored_agent():
    torch.manual_seed(0)
    from src.agents.factored_agent import FactoredActiveInference
    state_dim = 10
    act_dim = 5
    obs_dim = 12
    ctl_dim = 3
    H = 15
    batch_size = 24

    agent = FactoredActiveInference(state_dim, act_dim, obs_dim, ctl_dim, H, )
    print(agent)

    # create synthetic data
    T = 32
    o = torch.randn(T, batch_size, obs_dim)
    u = torch.randn(T, batch_size, ctl_dim)
    logp_pi, logp_obs, b = agent(o, u)
    u_bma = agent.choose_action(o, u, batch=True)
    u_ace = agent.choose_action(o, u, batch=True, num_samples=10)

    assert list(logp_pi.shape) == [T, batch_size]
    assert list(logp_obs.shape) == [T, batch_size]
    assert list(b.shape) == [T+1, batch_size, state_dim]
    assert list(u_bma.shape) == [T, batch_size, ctl_dim]
    assert list(u_ace.shape) == [10, T, batch_size, ctl_dim]
    print("test_factored_agent")

def test_embedded_agent():
    torch.manual_seed(0)
    from src.distributions.embedded_models import EmbeddedHiddenMarkovModel
    from src.distributions.models import ConditionalDistribution
    from src.agents.reward import ExpectedFreeEnergy
    from src.agents.embedded_agent import MessagePassingPlanner, EmbeddedActiveInference
    
    state_dim = 10
    act_dim = 5
    obs_dim = 12
    ctl_dim = 3
    H = 15
    batch_size = 24

    # create synthetic data
    T = 32
    o = torch.randn(T, batch_size, obs_dim)
    u = torch.randn(T, batch_size, ctl_dim)
    b = torch.softmax(torch.randn(batch_size, state_dim), dim=-1)

    hmm = EmbeddedHiddenMarkovModel(state_dim, act_dim, ctl_dim)
    obs_model = ConditionalDistribution(obs_dim, state_dim)
    rwd_model = ExpectedFreeEnergy(hmm, obs_model)
    planner = MessagePassingPlanner(hmm, obs_model, rwd_model, H)
    
    Q = planner.plan()
    pi = planner(b)

    agent = EmbeddedActiveInference(state_dim, act_dim, obs_dim, ctl_dim, H)
    print(agent)

    logp_pi, logp_obs, b = agent(o, u)
    u_bma = agent.choose_action(o, u, batch=True)
    u_ace = agent.choose_action(o, u, batch=True, num_samples=10)
    assert list(logp_pi.shape) == [T, batch_size]
    assert list(logp_obs.shape) == [T, batch_size]
    assert list(b.shape) == [T+1, batch_size, state_dim]
    assert list(u_bma.shape) == [T, batch_size, ctl_dim]
    assert list(u_ace.shape) == [10, T, batch_size, ctl_dim]
    print("test_embedded_agent")

if __name__ == "__main__":
    """ TODO: organize simple tests and complex tests """
    # test_value_iteration()
    # test_mlp_model()
    # test_popart_layer()
    # test_get_active_inference_parameters()

    # test_active_inference_agent()
    # test_agent_inference()
    # test_recurrent_agent()
    # test_nn_planner()
    # test_generalized_free_energy()
    # test_factored_value_iteration()
    # test_factored_agent()
    test_embedded_agent()