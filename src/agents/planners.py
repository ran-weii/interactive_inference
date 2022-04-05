import torch

def value_iteration(R, B, H):
    """
    Args:
        R (torch.tensor): reward matrix [batch_size, act_dim, state_dim]
        B (torch.tensor): transition matrix [batch_size, act_dim, state_dim, state_dim]
        H (int): planning horizon
        
    Returns:
        Q (torch.tensor): Q value [batch_size, H, act_dim, state_dim]
    """
    Q = [torch.empty(0)] * H
    Q[0] = R
    for h in range(H-1):
        V_next = torch.logsumexp(Q[h], dim=-2, keepdim=True).unsqueeze(-2)
        Q_next = torch.sum(B * V_next, dim=-1)
        Q[h+1] = R + Q_next
    Q = torch.stack(Q).transpose(0, 1)
    return Q