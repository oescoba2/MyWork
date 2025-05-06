from torch import nn, Tensor

class QNetwork(nn.Module):
    """Defines the DeepMind network architecture (i.e. model) to learn a given
       environment. The architecture is modeled after the one proposed in 2015
       with minor modifications. This network can also incorporate dueling 
       architecture.
    """

    def __init__(self, obs_channels:int, act_dim:int, dueling:bool=False):
        """Defines the network building blocks in order to learn 
        a given environment.
        
        Parameters:
            - obs_channels (int) : The number of channels in the observation 
                                   array that the environment emulator returns. 
            - act_dim (int) : the dimension of the action space of the 
                              environment. This is equal to the number of 
                              discrete actions that the agent can take in the
                              emulator.
            - dueling (bool) : whether or not to use the dueling architecture
                              Defaulted to False.

        Returns:
            - None
        """

        super().__init__()

        self.dueling = dueling
        self.conv = nn.Sequential(nn.Conv2d(in_channels=obs_channels, out_channels=32, kernel_size=8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                                  nn.ReLU())
        self.fc1 = nn.Sequential(nn.Flatten(), 
                                 nn.Linear(in_features=64*7*7, out_features=512),
                                 nn.ReLU())
        
        # Create dueling streams
        if self.dueling:
            self.val_stream = nn.Linear(in_features=512, out_features=1)
            self.adv_stream = nn.Linear(in_features=512, out_features=act_dim)

        # Normal architecture
        else:
            self.fc2 = nn.Linear(in_features=512, out_features=act_dim)

    def forward(self, imgs:Tensor) -> Tensor:
        """Defines the forward pass of the network. In particular, the defined
        network, takes in as an input a tensor of states or a single state
        and produces Q-vals for all available actions for each (or all actions
        for a single state).
        
        Parameters:
            - imgs (torch.Tensor) : the stacked images representing the "state"
                                    of the environment. The tensor should be of
                                    shape N x 84 x 84 x m, where N is the batch
                                    size of "states" given, and m is the number
                                    of channels in the "state" image array.
        Returns:
            - q_vals (Tensor) : The approximated Q-values Q(s,a), for the given 
                                "states"/"state" tensor. The tensor is of shape 
                                N x number of discrete actions in action space.
    
        """
        
        x_fc1 = self.fc1(self.conv(imgs))

        # Compute Q(s,a) using A(s,a)
        if self.dueling:

            state_vals = self.val_stream(x_fc1)
            adv_vals = self.adv_stream(x_fc1)
            q_vals = state_vals + (adv_vals - adv_vals.mean(dim=1, keepdim=True))

        else:
            q_vals = self.fc2(x_fc1)

        return q_vals