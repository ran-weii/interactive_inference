import torch
from src.distributions.models import (
    ConditionalDistribution, GeneralizedLinearModel, 
    MixtureDensityNetwork, HiddenMarkovModel)
from src.distributions.flows import SimpleTransformedModule, BatchNormTransform
from src.distributions.utils import poisson_pdf, rectify

""" TODO: finish all test cases """
def test_conditional_distribution():
    torch.manual_seed(0)
    x_dim = 12
    z_dim = 5
    sample_size = 24
    
    # synthetic observations and params
    batch_size = 32
    obs = torch.randn(batch_size, x_dim)
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
    
    # numerical accuracy test
    state_dim = 2
    act_dim = 2
    hmm = HiddenMarkovModel(state_dim, act_dim)

    logp_o = torch.tensor([[0.4, 0.6]]).log()
    a = torch.tensor([[1, 0]])
    b = torch.tensor([[0.7, 0.3]])
    B = torch.tensor([[[0.3, 0.7], [0.7, 0.3]], [[0.6, 0.4], [0.4, 0.6]]]).log()
    b_t_true = torch.tensor([[0.3256, 0.6744]])
    b_t = hmm(logp_o, a, b, B=B)
    
    assert torch.all(torch.abs(b_t - b_t_true) < 1e-4)

    # test embedded hmm
    state_dim = 12
    act_dim = 5
    hmm = HiddenMarkovModel(state_dim, act_dim, use_embedding=True)

    # synthetic data
    batch_size = 32
    logp_o = torch.softmax(torch.randn(batch_size, state_dim), dim=-1).log()
    a = torch.softmax(torch.randn(batch_size, act_dim), dim=-1)
    b = torch.softmax(torch.randn(batch_size, state_dim), dim=-1)
    b_t = hmm(logp_o, a, b, B=B)
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
    cond_dist = ConditionalDistribution(
        x_dim, z_dim, dist="mvn", cov="diag", batch_norm=True
    )
    cond_dist.bn.momentum = 1
    
    # synthetic observations
    batch_size = 32
    obs = 0.3 + 0.1 * torch.randn(batch_size, 1, x_dim)
    mu = torch.zeros(1, z_dim, x_dim)
    lv = torch.zeros(1, z_dim, x_dim)
    tl = torch.zeros(1, z_dim, x_dim, x_dim)
    sk = torch.zeros(1, z_dim, x_dim)
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
    
    log_probs_cond = cond_dist.log_prob(obs.squeeze(1))
    mean_cond = cond_dist.mean()
    variance_cond = cond_dist.variance()
    entropy_cond = cond_dist.entropy()
    
    assert list(log_probs_cond.shape) == [batch_size, z_dim]
    assert list(mean_cond.shape) == [1, z_dim, x_dim]
    assert list(variance_cond.shape) == [1, z_dim, x_dim]
    assert list(entropy_cond.shape) == [1, z_dim]
    
    assert torch.all((log_probs - log_probs_cond) < 1e-2)
    assert torch.all((mean_cond - empirical_distribution.mean) < 1e-4)
    assert torch.all((variance_cond - empirical_distribution.variance) < 1e-4)
    assert torch.all((entropy_cond - empirical_distribution.entropy()) < 1e-4)
    print("test_conditional_distribution_with_flow passed")

def test_generalized_linear_model():
    x_dim = 10
    z_dim = 5
    glm = GeneralizedLinearModel(
        x_dim, z_dim, dist="mvn", cov="diag", batch_norm=True
    )

    # create synthetic data
    T = 12
    batch_size = 32
    num_samples = 15
    x = torch.randn(T, batch_size, x_dim)

    pi = torch.softmax(torch.randn(T, batch_size, z_dim), dim=-1)
    log_probs = glm.mixture_log_prob(pi, x)
    samples_bma = glm.bayesian_average(pi)
    samples_ace = glm.ancestral_sample(pi, num_samples)
    
    assert list(log_probs.shape) == [T, batch_size]
    assert list(samples_bma.shape) == [T, batch_size, x_dim]
    assert list(samples_ace.shape) == [num_samples, T, batch_size, x_dim]
    print("test_generalized_linear_model passed")

def test_mixture_density_network():
    x_dim = 10
    z_dim = 5
    mdn = MixtureDensityNetwork(
        x_dim, z_dim, dist="mvn", cov="diag", batch_norm=True
    )

    # create synthetic data
    T = 12
    batch_size = 32
    num_samples = 15
    x = torch.randn(T, batch_size, x_dim)

    pi = torch.softmax(torch.randn(T, batch_size, z_dim), dim=-1)
    log_probs = mdn.mixture_log_prob(pi, x)
    samples_bma = mdn.bayesian_average(pi)
    samples_ace = mdn.ancestral_sample(pi, num_samples)

    assert list(log_probs.shape) == [T, batch_size]
    assert list(samples_bma.shape) == [T, batch_size, x_dim]
    assert list(samples_ace.shape) == [num_samples, T, batch_size, x_dim]
    print("test_mixture_density_network passed")

# def test_factored_hidden_markov_model():
#     from src.distributions.models import FactoredHiddenMarkovModel
#     torch.manual_seed(0)
#     state_dim = 10
#     act_dim = 5
#     obs_dim = 12
#     ctl_dim = 3
#     fhmm = FactoredHiddenMarkovModel(state_dim, act_dim, ctl_dim)

#     # create synthetic data
#     batch_size = 32
#     logp_o = torch.randn(batch_size, state_dim)
#     a = torch.softmax(torch.randn(batch_size, ctl_dim, act_dim), dim=-1)
#     b = torch.softmax(torch.randn(batch_size, state_dim), dim=-1)
    
#     b_t = fhmm(logp_o, a, b)
#     assert b_t.shape == b.shape

#     # test value iteration
#     from src.agents.reward import ExpectedFreeEnergy
#     from src.agents.planners import factored_value_iteration, FactoredQMDP
#     R = torch.randn(1, act_dim, act_dim, act_dim, state_dim)
#     B = torch.softmax(fhmm.B, dim=-1)
#     H = 2
#     Q = factored_value_iteration(R, B, H)

#     # test reward
#     obs_model = ConditionalDistribution(obs_dim, state_dim)
#     rwd_model = ExpectedFreeEnergy(fhmm, obs_model)
#     R = rwd_model(B, B)

#     # test planner
#     H = 10
#     planner = FactoredQMDP(fhmm, obs_model, rwd_model, H)
#     Q = planner.plan()
#     pi = planner(b)

#     print("test_factored_hidden_markov_model passed")

def test_factored_hidden_markov_model():
    torch.manual_seed(0)
    from src.distributions.factored_models import FactoredHiddenMarkovModel
    state_dim = 10
    act_dim = 5
    obs_dim = 12
    ctl_dim = 3
    batch_size = 24

    hmm = FactoredHiddenMarkovModel(state_dim, act_dim, ctl_dim)
    print(hmm)
    logp_o = torch.randn(batch_size, state_dim)
    a = torch.softmax(torch.randn([batch_size, ctl_dim, act_dim]), dim=-1)
    b = torch.softmax(torch.randn(batch_size, state_dim), dim=-1)
    b_t = hmm(logp_o, a, b)
    assert list(b_t.shape) == [batch_size, state_dim]
    print("test_factored_hidden_markov_model passed")

def test_factored_conditional_distribution():
    from src.distributions.factored_models import FactoredConditionalDistribution
    state_dim = 10
    act_dim = 5
    obs_dim = 12
    ctl_dim = 3
    batch_size = 24

    ctl_model = FactoredConditionalDistribution(
        ctl_dim, act_dim, dist="mvsn"
    )

    # create synthetic data
    u = torch.randn(batch_size, ctl_dim)
    a = torch.softmax(torch.randn(batch_size, ctl_dim, act_dim), dim=-1)
    logp = ctl_model.log_prob(u)
    samples = ctl_model.sample()
    samples_bma = ctl_model.bayesian_average(a)
    samples_ace = ctl_model.ancestral_sample(a, 1)
    logp_mix = ctl_model.mixture_log_prob(a, u)
    a_post = ctl_model.infer(a, u)
    
    assert list(logp.shape) == [batch_size, act_dim, ctl_dim]
    assert list(samples.shape) == [1, act_dim, ctl_dim]
    assert list(samples_bma.shape) == [1, batch_size, ctl_dim]
    assert list(samples_ace.shape) == [1, batch_size, ctl_dim]
    assert list(logp_mix.shape) == [batch_size, ctl_dim]
    assert list(a_post.shape) == list(a.shape)
    print("test_factored_conditional_distribution passed")

def test_embedded_distributions():
    torch.manual_seed(0)
    from src.distributions.embedded_models import LMDPLayer, EmbeddedHiddenMarkovModel
    
    state_dim = 10
    act_dim = 5
    ctl_dim = 3

    # create synthetic data
    batch_size = 32
    a = torch.softmax(torch.randn(batch_size, ctl_dim, act_dim), dim=-1)
    b = torch.softmax(torch.randn(batch_size, state_dim), dim=-1)
    logp_o = torch.randn(batch_size, state_dim)
    
    # test LMDP 
    lmdp = LMDPLayer(state_dim, act_dim, ctl_dim)
    transition = lmdp(a)
    assert list(transition.shape) == [batch_size, state_dim, state_dim]

    # test embedded hmm
    hmm = EmbeddedHiddenMarkovModel(state_dim, act_dim, ctl_dim)
    b_t = hmm(logp_o, a, b)
    assert list(b_t.shape) == [batch_size, state_dim]

    print("test_embedded_distributions")

def test_skew_normal():
    torch.manual_seed(0)
    import numpy as np
    from src.distributions.distributions import SkewNormal
    from scipy.stats import skewnorm
    
    # compare with scipy 
    batch_size = 32
    loc = torch.randn(batch_size)
    scale = rectify(torch.randn(batch_size))
    skew = torch.randn(batch_size)
    x = torch.randn(batch_size)
    
    sn = SkewNormal(skew, loc, scale)
    sn_scipy = skewnorm(skew.numpy(), loc.numpy(), scale.numpy())
    
    mean = sn.mean.numpy()
    variance = sn.variance.numpy()
    pdf = sn.log_prob(x).exp().numpy()

    mean_scipy = sn_scipy.mean()
    variance_scipy = sn_scipy.var()
    pdf_scipy = sn_scipy.pdf(x.numpy())
    
    assert np.all(np.abs(mean - mean_scipy) < 1e-4)
    assert np.all(np.abs(variance - variance_scipy) < 1e-4)
    assert np.all(np.abs(pdf - pdf_scipy) < 1e-4)
    
    # test multidimensional operations
    act_dim = 5
    ctl_dim = 3
    batch_size = 32
    loc = torch.randn(batch_size, ctl_dim, act_dim)
    scale = rectify(torch.randn(batch_size, ctl_dim, act_dim))
    skew = torch.randn(batch_size, ctl_dim, act_dim)
    x = torch.randn(batch_size, ctl_dim, act_dim)

    sn = SkewNormal(skew, loc, scale)
    
    mean = sn.mean
    variance = sn.variance
    logp = sn.log_prob(x)
    samples = sn.rsample()
    
    # visualize one example
    # import matplotlib.pyplot as plt
    # loc = torch.tensor([-10])
    # scale = torch.tensor([1])
    # skew = torch.tensor([-10])
    # grid = torch.linspace(-20, 0, 200)
    
    # sn = SkewNormal(skew, loc, scale)
    # pdf = sn.log_prob(grid).exp()
    # samples = sn.rsample((100, )).view(-1).numpy()
    
    # plt.plot(grid, pdf)
    # plt.hist(samples, bins="fd", density=True, alpha=0.4)
    # plt.show()

    print("test_skew_normal passed")

if __name__ == "__main__":
    # test_conditional_distribution()
    test_hidden_markov_model()
    # test_poisson_pdf()
    # test_batch_norm_flow()
    # test_conditional_distribution_with_flow()
    # test_generalized_linear_model()
    # test_mixture_density_network()
    # test_factored_hidden_markov_model()
    # test_factored_conditional_distribution()
    # test_embedded_distributions()
    # test_skew_normal()