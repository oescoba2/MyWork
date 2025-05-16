from torch import nn, Tensor
from torch.distributions import Normal
import torch

class PolicyNetwork(nn.Module):
    """Defines the neural network to learn the stochastic policy of an environment.
    This implements the actor. This class inherits from nn.Module.
    """

    def __init__(self, state_dim:int, action_dim:int, hidden_size:int=64) -> None:
        """Defines the MLP architecture elements that will be used to approximate 
        the stochastic policy.

        Parameters:
            - state_dim (gym.box) : the dimension of the environment's 
                                    state/observation space.
            - action_dim (int) : the dimension of the environment's action space.
            - hidden_size (int) : an int indicating the the number of perceptrons
                                  to use in the hidden layers of the MLP. Default
                                  is 64.
        
        Returns:
            - None
        """

        super().__init__()

        # Architecture: base MLP with two heads/streams
        self.baseMLP = nn.Sequential(nn.Linear(in_features=state_dim, out_features=hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(in_features=hidden_size, out_features=hidden_size),
                                    nn.ReLU(),)
        self.mean_head = nn.Linear(in_features=hidden_size, out_features=action_dim)
        self.logStd_head = nn.Linear(in_features=hidden_size, out_features=action_dim)                                        

    def forward(self, state:Tensor) -> tuple[Tensor, Tensor]:
        """Defines the forward pass of the network and takes the given state
        tensor through the defined archicture as structured below. This com-
        putes the parameters of the probability distribution over the action
        space conditioned on the given state.

        Parameters:
            - state (Tensor) : the state to condition on in order to compute 
                               the probability distribution. This is of shape 
                               (state_dim, ) or (M, state_dim) where M is the
                               batch size.
        
        Return:
            - μ (torch.tensor) : the mean of the probability distribution
                                 for each action condition on the given 
                                 state. This is of shape (action_dim, )
                                 or (M, action_dim) when given a batch
                                 size of states.
                                 
            - σ (torch.tensor) : the standard deviation of the probability
                                 distribution for each action condition on
                                 a given state. This is of shape (action_dim, )
                                 or (M, action_dim) when given a batch size of
                                 states.
        """

        x = self.baseMLP(state)
        μ = self.mean_head(x)
        log_σ = torch.clamp(self.logStd_head(x), min=-20, max=2)

        return μ, torch.exp(log_σ)

    def act(self, state:Tensor) -> tuple[Tensor, Tensor]:
        """Computes the Gaussian distribution for a given state over the 
        action space and samples it in order to select an action to take.

        Parameters:
            - state (Tensor) : the state to condition on in order to 
                               sample the distribution and return an
                               action. This a tensor of shape (state_dim, )
                               or (M, state_dim) when a batch, size M, of 
                               states is given.

        Returns:
            - action (Tensor) : the action to take in the given state 
                                sampled from its distribution. This is of
                                shape (action_dim, ) or (M, state_dim) 
                                when a batch of states is given.
            - action_logprob (Tensor) : The log of the probability of
                                        the sampled action summed across
                                        all possible actions. This is of
                                        shape (1, ) or (M, 1) when a batch
                                        of states is given.
        """

        μ, σ = self.forward(state)
        distribution = Normal(loc=μ, scale=σ)
        action = distribution.sample()
        action_logprob = distribution.log_prob(value=action).sum(dim=-1)

        return action, action_logprob
    
class ValueNetwork(nn.Module):
    """Defines the neural network to learn the state-value function of the
    environment. This implements the critic. It inherits from nn.Module.
    """

    def __init__(self, state_dim:int, hidden_size:int=64) -> None:
        """Defines the MLP architecture elements that will be used to approximate 
        the state value function v(s).

        Parameters:
            - state_dim (int) : the dimension of the environment's state/observation
                                space.
            - hidden_size (int) : an int indicating the the number of perceptrons to 
                                  use in the hidden layers of the MLP. Default is 64.
        
        Returns:
            - None
        """

        super().__init__()

        self.MLP = nn.Sequential(nn.Linear(in_features=state_dim, out_features=hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(in_features=hidden_size, out_features=hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(in_features=hidden_size, out_features=1))
    
    def forward(self, state:Tensor) -> Tensor:
        """Computes the state value of the given state(s).
        
        Parameters:
            - state (Tensor) : the current state(s) of the environment.

        Returns:
            - (Tensor) : the estimated value, V(s), of the given state(s).
        """
        
        return self.MLP(state)