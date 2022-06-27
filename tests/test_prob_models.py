import torch
from src.distributions.hmm import DiscreteGaussianHMM
from src.distributions.hmm import LogisticGaussianHMM

seed = 0

def test_discrete_gaussian_hmm():
    torch.manual_seed(seed)
    state_dim = 10
    act_dim = 5
    obs_dim = 12

    # synthetic data
    T = 30
    batch_size = 32
    x = torch.randn(T, batch_size, obs_dim)
    a = torch.softmax(torch.randn(T, batch_size, act_dim), dim=-1)
    mask = (torch.randn(T, batch_size) < 1) * 1
    
    # case 1: full rank
    rank = 0
    hmm = DiscreteGaussianHMM(state_dim, act_dim, obs_dim, rank=rank)
    initial_state = hmm.get_initial_state()
    transition_matrix = hmm.get_transition_matrix(a[0])
    assert list(initial_state.shape) == [1, state_dim]
    assert list(transition_matrix.shape) == [batch_size, state_dim, state_dim]

    alpha = hmm._forward(x, a)
    log_beta = hmm._backward(x, a, mask)
    
    # case 2: low rank
    rank = 5
    hmm = DiscreteGaussianHMM(state_dim, act_dim, obs_dim, rank=rank)
    initial_state = hmm.get_initial_state()
    transition_matrix = hmm.get_transition_matrix(a[0])
    assert list(initial_state.shape) == [1, state_dim]
    assert list(transition_matrix.shape) == [batch_size, state_dim, state_dim]

    alpha = hmm._forward(x, a)
    log_beta = hmm._backward(x, a, mask)
    
    print("test_discrete_gaussian_hmm passed")

def test_logistic_gaussian_hmm():
    torch.manual_seed(seed)
    state_dim = 10
    act_dim = 5
    obs_dim = 12
    
    # synthetic data
    T = 30
    batch_size = 32
    x = torch.randn(T, batch_size, obs_dim)
    a = torch.randn(T, batch_size, act_dim)
    mask = (torch.randn(T, batch_size) < 1) * 1

    hmm = LogisticGaussianHMM(state_dim, act_dim, obs_dim)
    transition = hmm.transition_model.get_transition_matrix(a[0])
    
    print("test_gaussian_hmm passed")

if __name__ == "__main__":
    test_discrete_gaussian_hmm()
    # test_logistic_gaussian_hmm()