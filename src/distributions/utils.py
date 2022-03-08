import math
import torch

def make_covariance_matrix(logvar, tril, cholesky=True):
    """ Make full covarance matrix
    
    Args:
        logvar (torch.tensor): log variance vector [batch_size, dim]
        tril (torch.tensor): unmaksed lower triangular matrix [batch_size, dim, dim]
        cholesky (bool, optional): return cholesky decomposition, default=False
    
    Returns:
        L (torch.tensor): scale_tril or cov [batch_size, dim, dim]
    """
    var = torch.exp(logvar.clip(math.log(1e-6), math.log(1e6)))
    L = torch.tril(tril, diagonal=-1)
    L = L + torch.diag_embed(var)
    
    if not cholesky:
        L = torch.bmm(L, L.transpose(-1, -2))
    return L

def poisson_pdf(gamma, K):
    """ 
    Args:
        gamma (torch.tensor): poission arrival rate [batch_size, 1]
        K (int): number of bins

    Returns:
        pdf (torch.tensor): truncated poisson pdf [batch_size, K]
    """
    assert torch.all(gamma > 0)
    Ks = torch.arange(K) + 1
    poisson_dist = torch.distributions.Poisson(gamma)
    pdf = torch.softmax(poisson_dist.log_prob(Ks), dim=-1)
    return pdf