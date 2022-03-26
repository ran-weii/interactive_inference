import torch
from src.distributions.models import ConditionalDistribution, HiddenMarkovModel
from src.distributions.flows import SimpleTransformedModule, BatchNormTransform
from src.distributions.utils import poisson_pdf

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
    entropy = cond_dist.entropy(params)
    logp = cond_dist.log_prob(obs, params)
    samples = cond_dist.sample((sample_size,), params)
    
    assert list(mean.shape) == [batch_size, z_dim, x_dim]
    assert list(variance.shape) == [batch_size, z_dim, x_dim]
    assert list(entropy.shape) == [batch_size, z_dim]
    assert list(logp.shape) == [batch_size, z_dim]
    assert list(samples.shape) == [sample_size, batch_size, z_dim, x_dim]
    assert torch.all(mean - mu == 0)
    
    # test mvn with full cov
    dist = "mvn"
    cov = "full"
    cond_dist = ConditionalDistribution(x_dim, z_dim, dist, cov)
    mean = cond_dist.mean()
    variance = cond_dist.variance()
    entropy = cond_dist.entropy()
    logp = cond_dist.log_prob(obs)
    samples = cond_dist.sample((sample_size,))
    
    assert list(mean.shape) == [1, z_dim, x_dim]
    assert list(variance.shape) == [1, z_dim, x_dim]
    assert list(entropy.shape) == [1, z_dim]
    assert list(logp.shape) == [batch_size, z_dim]
    assert list(samples.shape) == [sample_size, 1, z_dim, x_dim]
    assert torch.all(mean - cond_dist.mu == 0)
    
    # test mvn with diag cov
    dist = "mvn"
    cov = "diag"
    cond_dist = ConditionalDistribution(x_dim, z_dim, dist, cov)
    mean = cond_dist.mean()
    variance = cond_dist.variance()
    entropy = cond_dist.entropy()
    logp = cond_dist.log_prob(obs)
    samples = cond_dist.sample((sample_size,))
    
    assert list(mean.shape) == [1, z_dim, x_dim]
    assert list(variance.shape) == [1, z_dim, x_dim]
    assert list(entropy.shape) == [1, z_dim]
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

def test_poisson_pdf():
    batch_size = 32
    gamma = torch.randn(batch_size, 1).exp()
    K = 12
    
    pdf = poisson_pdf(gamma, K)
    assert torch.all(torch.isclose(pdf.sum(-1), torch.ones(1)))
    
    print("test_poisson_pdf passed")

def test_batch_norm_flow():
    torch.manual_seed(0)
    
    # created transformed distributions
    x_dim = 10
    mean = torch.zeros(x_dim)
    variance = torch.ones(x_dim)
    cov = torch.diag_embed(variance)
    base_distribution = torch.distributions.MultivariateNormal(mean, cov)
    
    a = torch.randn(x_dim) # loc
    b = torch.exp(0.5 * torch.randn(x_dim)) # scale
    cov_transformed = torch.diag_embed(b**2)
    transformed_distribution = torch.distributions.MultivariateNormal(a, cov_transformed)
    log_abs_det_jacobian = torch.abs(torch.log(b).sum())
    
    # generate synthetic samples
    T = 32
    batch_size = 128
    samples = torch.randn(T, batch_size, x_dim)
    transformed_samples = a + b * samples
    log_probs = base_distribution.log_prob(samples)
    log_probs_transformed = transformed_distribution.log_prob(transformed_samples)
    log_probs_diff = log_probs_transformed - (log_probs + log_abs_det_jacobian)
    assert torch.all(log_probs_diff < 1e-5) # verify change of variable formula
    
    # get empirical means and variance
    op_dims = [0, 1]
    mu = torch.mean(transformed_samples, dim=op_dims)
    sd = torch.std(transformed_samples, dim=op_dims)
    cov_empirical = torch.diag_embed(sd**2)
    normalized_samples = (transformed_samples - mu) / sd
    
    empirical_distribution = torch.distributions.MultivariateNormal(mu, cov_empirical)
    log_probs_base = base_distribution.log_prob(normalized_samples)
    log_probs_empirical = empirical_distribution.log_prob(transformed_samples)
    log_abs_det_jacobian_empirical = torch.abs(torch.log(sd).sum())
    log_probs_diff_empirical = log_probs_empirical - (log_probs_base + log_abs_det_jacobian_empirical)
    assert torch.all(log_probs_diff_empirical < 1e-5) # verify change of variable formula
    
    # init flow
    bn_flow = BatchNormTransform(x_dim, momentum=1) # give all weights to new mean and var
    bn_flow.gamma.requires_grad = False
    bn_flow.beta.requires_grad = False
    
    composed_distribution = SimpleTransformedModule(base_distribution, [bn_flow])
    log_probs_flow = composed_distribution.log_prob(transformed_samples)
    log_probs_diff_flow = log_probs_flow - log_probs_empirical
    assert torch.all(log_probs_diff_flow < 1e-3)
    
    mean_flow = composed_distribution.mean
    variance_flow = composed_distribution.variance
    entropy_flow = composed_distribution.entropy()
    assert torch.all(mean_flow == empirical_distribution.mean)
    assert torch.all((variance_flow - empirical_distribution.variance) < 1e-5)
    assert entropy_flow - empirical_distribution.entropy() < 1e-5
    print("test_batch_norm_flow passed")

def test_conditional_distribution_with_flow():
    x_dim = 10
    z_dim = 5
    cond_dist = ConditionalDistribution(x_dim, z_dim, dist="mvn", cov="diag")
    cond_dist.bn.momentum = 1
    
    # synthetic observations
    batch_size = 32
    obs = 0.3 + 0.1 * torch.randn(batch_size, 1, x_dim)
    mu = torch.zeros(batch_size, z_dim, x_dim)
    lv = torch.zeros(batch_size, z_dim, x_dim)
    tl = torch.zeros(batch_size, z_dim, x_dim, x_dim)
    sk = torch.zeros(batch_size, z_dim, x_dim)
    cond_dist.mu.data = mu
    cond_dist.lv.data = lv
    cond_dist.tl.data = tl
    cond_dist.sk.data = sk
    
    # get empirical distribution
    op_dims = [0, 1]
    empirical_mean = torch.mean(obs, dim=op_dims)
    empirical_variance = torch.var(obs, dim=op_dims)
    empirical_cov = torch.diag_embed(empirical_variance)
    empirical_distribution = torch.distributions.MultivariateNormal(empirical_mean, empirical_cov)
    log_probs = empirical_distribution.log_prob(obs)
    
    log_probs_cond = cond_dist.log_prob(obs)
    mean_cond = cond_dist.mean()
    variance_cond = cond_dist.variance()
    entropy_cond = cond_dist.entropy()
    
    assert torch.all((log_probs - log_probs_cond) < 1e-2)
    assert torch.all((mean_cond - empirical_distribution.mean) < 1e-4)
    assert torch.all((variance_cond - empirical_distribution.variance) < 1e-4)
    assert torch.all((entropy_cond - empirical_distribution.entropy()) < 1e-4)
    print("test_conditional_distribution_with_flow passed")

if __name__ == "__main__":
    test_conditional_distribution()
    test_hidden_markov_model()
    test_poisson_pdf()
    test_batch_norm_flow()
    test_conditional_distribution_with_flow()