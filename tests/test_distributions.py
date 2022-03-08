from pyro import param
import torch
from src.distributions.models import ConditionalDistribution, HiddenMarkovModel

""" TODO: finish all test cases """
def test_conditional_distribution():
    torch.manual_seed(0)
    x_dim = 12
    z_dim = 5
    sample_size = 24
    
    # synthetic observations and params
    batch_size = 32
    obs = torch.randn(batch_size, 1, x_dim)
    mu = torch.randn(batch_size, z_dim, x_dim)
    lv = torch.randn(batch_size, z_dim, x_dim)
    tl = torch.randn(batch_size, z_dim, x_dim, x_dim)
    sk = torch.randn(batch_size, z_dim, x_dim)
    
    # test with supplied parameters
    dist = "mvn"
    cov = "full"
    params = torch.cat(
        [mu.view(batch_size, -1), lv.view(batch_size, -1), tl.view(batch_size, -1)],
        dim=-1
    )
    cond_dist = ConditionalDistribution(x_dim, z_dim, dist, cov)
    mean = cond_dist.mean(params)
    variance = cond_dist.variance(params)
    logp = cond_dist.log_prob(obs, params)
    samples = cond_dist.sample((sample_size,), params)
    
    assert list(mean.shape) == [batch_size, z_dim, x_dim]
    assert list(variance.shape) == [batch_size, z_dim, x_dim]
    assert list(logp.shape) == [batch_size, z_dim]
    assert list(samples.shape) == [sample_size, batch_size, z_dim, x_dim]
    assert torch.all(mean - mu == 0)
    
    # test mvn with full cov
    dist = "mvn"
    cov = "full"
    cond_dist = ConditionalDistribution(x_dim, z_dim, dist, cov)
    mean = cond_dist.mean()
    variance = cond_dist.variance()
    logp = cond_dist.log_prob(obs)
    samples = cond_dist.sample((sample_size,))
    
    assert list(mean.shape) == [1, z_dim, x_dim]
    assert list(variance.shape) == [1, z_dim, x_dim]
    assert list(logp.shape) == [batch_size, z_dim]
    assert list(samples.shape) == [sample_size, 1, z_dim, x_dim]
    assert torch.all(mean - cond_dist.mu == 0)
    
    # test mvn with diag cov
    dist = "mvn"
    cov = "diag"
    cond_dist = ConditionalDistribution(x_dim, z_dim, dist, cov)
    mean = cond_dist.mean()
    variance = cond_dist.variance()
    logp = cond_dist.log_prob(obs)
    samples = cond_dist.sample((sample_size,))
    
    assert list(mean.shape) == [1, z_dim, x_dim]
    assert list(variance.shape) == [1, z_dim, x_dim]
    assert list(logp.shape) == [batch_size, z_dim]
    assert list(samples.shape) == [sample_size, 1, z_dim, x_dim]
    assert torch.all(mean - cond_dist.mu == 0)
    
    # test mvsn with full cov
    
    # test mvsn with diag cov
    
    print("test_conditional_distribution passed")

def test_hidden_markov_model():
    state_dim = 12
    act_dim = 5
    hmm = HiddenMarkovModel(state_dim, act_dim)
    
    # synthetic data
    batch_size = 32
    logp_o = torch.softmax(torch.randn(batch_size, state_dim), dim=-1).log()
    a = torch.softmax(torch.randn(batch_size, act_dim), dim=-1)
    b = torch.softmax(torch.randn(batch_size, state_dim), dim=-1)
    
    # test with supplied parameters
    B = torch.randn(batch_size, act_dim, state_dim, state_dim)
    b_t = hmm(logp_o, a, b, B=B)
    assert b_t.shape == b.shape
    
    # test with self parameters
    b_t = hmm(logp_o, a, b)
    assert b_t.shape == b.shape
    
    print("test_hidden_markov_model passed")

if __name__ == "__main__":
    test_conditional_distribution()
    test_hidden_markov_model()