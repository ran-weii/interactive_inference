import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from src.distributions.nn_models import PopArt
from src.agents.rule_based import IDM
from src.agents.nn_agents import MLPAgent, RNNAgent
from src.agents.vin_agent import VINAgent
from src.agents.hyper_vin_agent import HyperVINAgent
from src.algo.bc import BehaviorCloning, RecurrentBehaviorCloning
from src.algo.hyper_bc import HyperBehaviorCloning

def test_popart_layer():
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

def test_idm_agent():
    obs_dim = 3
    ctl_dim = 1
    feature_set = ["ego_ds", "lv_s_rel", "lv_ds_rel"]
    
    agent = IDM(feature_set)
    
    # generate synthetic data
    T = 7
    batch_size = 32
    o = torch.randn(T, batch_size, obs_dim)
    u = torch.randn(T, batch_size, ctl_dim)
    seq_len = torch.randint(1, T, (batch_size - 1,))
    seq_len = torch.cat([seq_len, T * torch.ones(1)]).long()
    mask = pad_sequence([torch.ones(s) for s in seq_len])
    
    # test offline
    with torch.no_grad():
        out, _ = agent.forward(o)
        act_loss, stats = agent.act_loss(o, u, mask, out)
        u_sample, logp = agent.choose_action_batch(o, u)
    
    assert list(out[0].shape) == [T, batch_size, ctl_dim]
    assert list(act_loss.shape) == [batch_size]
    assert list(u_sample.shape) == [1, T, batch_size, ctl_dim]
    assert list(logp.shape) == [1, T, batch_size]
    
    # test online
    agent.reset()
    pi_t = []
    u_sample_t = []
    for t in range(T):
        with torch.no_grad():
            u_sample_t.append(agent.choose_action(o[t])[0])
            pi_t.append(agent._state["pi"])

    pi_t = torch.stack(pi_t)
    u_sample_t = torch.stack(u_sample_t).squeeze(1)

    assert list(pi_t.shape) == [T, batch_size, ctl_dim]
    assert list(u_sample_t.shape) == [T, batch_size, ctl_dim]
    assert torch.all(torch.abs(pi_t - out[0]) < 1e-5)    
    
    # test training
    loader = [{"obs": o.flatten(0, 1), "act": u.flatten(0, 1)}]
    trainer = BehaviorCloning(agent)
    trainer.run_epoch(loader)

    # test recurrent training
    loader = [[{"obs": o, "act": u}, mask]]
    trainer = RecurrentBehaviorCloning(agent, bptt_steps=5)
    trainer.run_epoch(loader)

    print("test_idm_agent passed")

def test_mlp_agent():
    obs_dim = 10
    ctl_dim = 3
    act_dim = 5
    hidden_dim = 12
    num_hidden = 2
    activation = "relu"

    agent = MLPAgent(
        obs_dim, ctl_dim, act_dim, hidden_dim, num_hidden, activation
    )
    
    # generate synthetic data
    T = 7
    batch_size = 32
    o = torch.randn(T, batch_size, obs_dim)
    u = torch.randn(T, batch_size, ctl_dim)
    seq_len = torch.randint(1, T, (batch_size - 1,))
    seq_len = torch.cat([seq_len, T * torch.ones(1)]).long()
    mask = pad_sequence([torch.ones(s) for s in seq_len])
    
    # test offline
    with torch.no_grad():
        [pi], _ = agent.forward(o)
        act_loss, stats = agent.act_loss(o, u, mask, pi)
        u_sample, logp = agent.choose_action_batch(o, u)

    assert list(pi.shape) == [T, batch_size, act_dim]
    assert list(act_loss.shape) == [batch_size]
    assert list(u_sample.shape) == [1, T, batch_size, ctl_dim]
    assert list(logp.shape) == [1, T, batch_size]

    # test online
    agent.reset()
    pi_t = []
    u_sample_t = []
    for t in range(T):
        with torch.no_grad():
            u_sample_t.append(agent.choose_action(o[t])[0])
            pi_t.append(agent._state["pi"])

    pi_t = torch.stack(pi_t)
    u_sample_t = torch.stack(u_sample_t).squeeze(1)
    
    assert list(pi_t.shape) == [T, batch_size, act_dim]
    assert list(u_sample_t.shape) == [T, batch_size, ctl_dim]
    assert torch.all(torch.abs(pi_t - pi) < 1e-5)
    
    # test training
    loader = [{"obs": o.flatten(0, 1), "act": u.flatten(0, 1)}]
    trainer = BehaviorCloning(agent)
    trainer.run_epoch(loader)

    # test recurrent training
    loader = [[{"obs": o, "act": u}, mask]]
    trainer = RecurrentBehaviorCloning(agent, bptt_steps=5)
    trainer.run_epoch(loader)

    print("test_mlp_agent passed")

def test_rnn_agent():
    obs_dim = 10
    ctl_dim = 3
    act_dim = 5
    hidden_dim = 12
    num_hidden = 2
    gru_layers = 1
    activation = "relu"

    agent = RNNAgent(
        obs_dim, ctl_dim, act_dim, hidden_dim, num_hidden, gru_layers, activation
    )
    
    # generate synthetic data
    T = 7
    batch_size = 32
    o = torch.randn(T, batch_size, obs_dim)
    u = torch.randn(T, batch_size, ctl_dim)
    seq_len = torch.randint(1, T, (batch_size - 1,))
    seq_len = torch.cat([seq_len, T * torch.ones(1)]).long()
    mask = pad_sequence([torch.ones(s) for s in seq_len])
    
    # test offline
    with torch.no_grad():
        [b, pi], hidden = agent.forward(o, u)
        act_loss, stats = agent.act_loss(o, u, mask, hidden)
        u_sample, logp = agent.choose_action_batch(o, u)

    assert list(pi.shape) == [T, batch_size, act_dim]
    assert list(act_loss.shape) == [batch_size]
    assert list(u_sample.shape) == [1, T, batch_size, ctl_dim]
    assert list(logp.shape) == [1, T, batch_size]

    # test online
    agent.reset()
    b_t = []
    pi_t = []
    u_sample_t = []
    for t in range(T):
        with torch.no_grad():
            u_sample_t.append(agent.choose_action(o[t])[0])
            b_t.append(agent._state["b"])
            pi_t.append(agent._state["pi"])
    
    b_t = torch.stack(b_t)
    pi_t = torch.stack(pi_t)
    u_sample_t = torch.stack(u_sample_t).squeeze(1)
    
    assert list(b_t.shape) == [T, batch_size, hidden_dim]
    assert list(pi_t.shape) == [T, batch_size, act_dim]
    assert list(u_sample_t.shape) == [T, batch_size, ctl_dim]
    assert torch.all(torch.abs(b_t - b) < 1e-5)
    assert torch.all(torch.abs(pi_t - pi) < 1e-5)
    
    # test training
    loader = [[{"obs": o, "act": u}, mask]]
    trainer = RecurrentBehaviorCloning(agent, bptt_steps=5)
    trainer.run_epoch(loader)

    print("test_rnn_agent passed")

def test_vin_agent():
    obs_dim = 10
    state_dim = 12
    act_dim = 5
    ctl_dim = 3
    rank = 0
    horizon = 15
    
    agent = VINAgent(
        state_dim, act_dim, obs_dim, ctl_dim, rank, horizon,
        obs_model="gmm"
    )
    agent.eval()

    # generate synthetic data
    T = 7
    batch_size = 32
    o = torch.randn(T, batch_size, obs_dim)
    u = torch.randn(T, batch_size, ctl_dim)
    seq_len = torch.randint(1, T, (batch_size - 1,))
    seq_len = torch.cat([seq_len, T * torch.ones(1)]).long()
    mask = pad_sequence([torch.ones(s) for s in seq_len])
    
    # test offline
    with torch.no_grad():
        [b, pi, value], hidden = agent.forward(o, u)
        act_loss, stats = agent.act_loss(o, u, mask, hidden)
        u_sample, logp = agent.choose_action_batch(o, u)

    assert list(pi.shape) == [T, batch_size, act_dim]
    assert list(act_loss.shape) == [batch_size]
    assert list(u_sample.shape) == [1, T, batch_size, ctl_dim]
    assert list(logp.shape) == [1, T, batch_size]

    # test online
    agent.reset()
    b_t = []
    pi_t = []
    u_sample_t = []
    for t in range(T):
        with torch.no_grad():
            u_sample_t.append(agent.choose_action(o[t])[0])
            b_t.append(agent._state["b"])
            pi_t.append(agent._state["pi"])
            agent._prev_ctl = u[t].unsqueeze(0)
    
    b_t = torch.stack(b_t)
    pi_t = torch.stack(pi_t)
    u_sample_t = torch.stack(u_sample_t).squeeze(1)
    
    assert list(b_t.shape) == [T, batch_size, state_dim]
    assert list(pi_t.shape) == [T, batch_size, act_dim]
    assert list(u_sample_t.shape) == [T, batch_size, ctl_dim]
    assert torch.abs(b_t - b).mean() < 1e-3
    assert torch.abs(pi_t - pi).mean() < 1e-3
    
    # test training
    loader = [[{"obs": o, "act": u}, mask]]
    trainer = RecurrentBehaviorCloning(agent, bptt_steps=5)
    trainer.run_epoch(loader)

    print("test_vin_agent passed")

def test_hvin_agent():
    obs_dim = 10
    state_dim = 12
    act_dim = 5
    ctl_dim = 3
    rank = 0
    horizon = 15
    hyper_dim = 4
    hidden_dim = 32
    num_hidden = 1
    gru_layers = 1
    activation = "relu"
    
    agent = HyperVINAgent(
        state_dim, act_dim, obs_dim, ctl_dim, rank, horizon,
        hyper_dim, hidden_dim, num_hidden, gru_layers, activation, obs_model="gmm"
    )
    agent.eval()

    # generate synthetic data
    T = 7
    batch_size = 32
    o = torch.randn(T, batch_size, obs_dim)
    u = torch.randn(T, batch_size, ctl_dim)
    seq_len = torch.randint(1, T, (batch_size - 1,))
    seq_len = torch.cat([seq_len, T * torch.ones(1)]).long()
    mask = pad_sequence([torch.ones(s) for s in seq_len])
    
    # test offline
    with torch.no_grad():
        z = agent.encode(o, u, mask)
        [b, pi, value], hidden = agent.forward(o, u, z)
        act_loss, stats = agent.act_loss(o, u, z, mask, hidden)
        u_sample, logp = agent.choose_action_batch(o, u, z)

    assert list(pi.shape) == [T, batch_size, act_dim]
    assert list(act_loss.shape) == [batch_size]
    assert list(u_sample.shape) == [1, T, batch_size, ctl_dim]
    assert list(logp.shape) == [1, T, batch_size]

    # test online
    agent.reset(z)
    b_t = []
    pi_t = []
    u_sample_t = []
    for t in range(T):
        with torch.no_grad():
            u_sample_t.append(agent.choose_action(o[t])[0])
            b_t.append(agent._state["b"])
            pi_t.append(agent._state["pi"])
            agent._prev_ctl = u[t].unsqueeze(0)
    
    b_t = torch.stack(b_t)
    pi_t = torch.stack(pi_t)
    u_sample_t = torch.stack(u_sample_t).squeeze(1)
    
    assert list(b_t.shape) == [T, batch_size, state_dim]
    assert list(pi_t.shape) == [T, batch_size, act_dim]
    assert list(u_sample_t.shape) == [T, batch_size, ctl_dim]
    assert torch.abs(b_t - b).mean() < 1e-3
    assert torch.abs(pi_t - pi).mean() < 1e-3
    
    # test training
    loader = [[{"obs": o, "act": u}, mask]]
    trainer = HyperBehaviorCloning(agent, train_mode="prior", bptt_steps=5)
    trainer.run_epoch(loader)

    print("test_vin_agent passed")

if __name__ == "__main__":
    torch.manual_seed(0)
    
    test_idm_agent()
    test_mlp_agent()
    test_rnn_agent()
    test_vin_agent()
    test_hvin_agent()