from src.distributions.nn_models import Model

class AbstractAgent(Model):
    def __init__(self, *args, **kwargs):
        super().__init__() 
    
    def reset(self):
        """ Reset internal states for online inference """
        raise NotImplementedError
    
    def init_hidden(self):
        """ Initialize hidden states """
        raise NotImplementedError 

    def forward(self, o, u):
        """ Forward algorithm
        Args:
            o (torch.tensor): 
            u (torch.tensor): 
        """
        raise NotImplementedError
    
    def choose_action_batch(self, o, u, sample_method="", num_samples=1):
        """ Choose action offline for a batch of sequences 
        
        Args:
            o (torch.tensor): observation sequence. size[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size[T, batch_size, ctl_dim]
            sample_method (str, optional): sampling method. Default=""
            num_samples (int, optional): number of samples to draw. Default=1
        """
        raise NotImplementedError

    def choose_action(self, o, u, sample_method="", num_samples=1):
        """ Choose action online for a single time step
        
        Args:
            o (torch.tensor): observation sequence. size[batch_size, obs_dim]
            u (torch.tensor): control sequence. size[batch_size, ctl_dim]
            sample_method (str, optional): sampling method. Default=""
            num_samples (int, optional): number of samples to draw. Default=1
        """
        raise NotImplementedError