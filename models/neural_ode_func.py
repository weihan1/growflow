from torch import nn
import torch
import torch.nn.functional as F
import sys
from models.hexplane import HexPlaneField
from models.encoding import get_embedder, exp_se3
import time


class Simple_ODE(nn.Module):
    """
    Takes a ode sample of dimension (feature_dim) and outputs its derivative.
    Note the output dimension must match the input dimension.
    """

    def __init__(
        self,
        feature_dim,
        encoding_network_pos=None,
        encoding_network_time=None,
        hidden_dim=50,
        encoding_name="",
        encoder_output_dim=0,
        time_input_ch = 0,
        xyz_input_ch = 0,
        hidden_depth=2,
        non_linearity_name="relu",
        augment_dim=0,
        concat_remaining =True,
        resnet_init = True,
        feature_out = None,
        use_timenet = False,
        use_skip = False,
        feature_out_output_dim = 64
    ):
        super(Simple_ODE, self).__init__()
        self.device="cuda"
        self.feature_dim = feature_dim #this is the starting feature dimension
        self.augment_dim = augment_dim
        self.encoding_network_pos = encoding_network_pos
        self.encoding_network_time = encoding_network_time
        self.encoder_output_dim = encoder_output_dim
        # self.input_dim = feature_dim + augment_dim + encoder_output_dim
        self.input_dim = encoder_output_dim + augment_dim
        self.output_dim = feature_dim + augment_dim #The output should include the augmented dimension, since we are also augmenting the y0
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.non_linearity_name = non_linearity_name
        self.encoding_name = encoding_name
        self.nfe = 0  # Number of function evaluations
        self.concat_remaining = concat_remaining #whether to concat the remaining parameters when inputting to the neural ode
        self.resnet_init =resnet_init
        self.feature_out = feature_out
        self.use_timenet = use_timenet
        self.use_skip = use_skip
        self.skips = [hidden_depth // 2]
        self.feature_out_output_dim = feature_out_output_dim #only used in hexplanes


        self.use_absolute_encoding=None #only used when training with multiple init conditions
        self.absolute_times = None #absolute times denote the absolute times of the initial conditions (and of their integration steps)

        if self.encoder_output_dim > 0:
            if not concat_remaining:
                self.input_dim -= self.feature_dim - 3 #substract the remaining shapes
        if not encoding_network_pos:
            assert self.encoder_output_dim == 0
            self.input_dim += self.feature_dim
        else:
            assert self.encoder_output_dim > 0

        if non_linearity_name == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        elif non_linearity_name == 'softplus':
            self.non_linearity = nn.Softplus()
        elif non_linearity_name == "silu":
            self.non_linearity = nn.SiLU()
        elif non_linearity_name == "leakyrelu":
            self.non_linearity = nn.LeakyReLU()


        if self.encoding_name == "freq":
            if self.use_timenet:
                self.time_out = 30
                self.timenet = nn.Sequential(
                    nn.Linear(time_input_ch, self.hidden_dim), nn.ReLU(inplace=True), 
                    nn.Linear(self.hidden_dim, self.time_out))
                self.input_dim -= (time_input_ch - self.time_out)

            self.mlp = mlp(input_dim=self.input_dim, output_dim=self.output_dim, hidden_dim=self.hidden_dim, hidden_depth=self.hidden_depth, act=self.non_linearity)
            self.mlp_means = nn.Linear(self.hidden_dim, 3)
            self.mlp_quats = nn.Linear(self.hidden_dim, 4)
            self.mlp_scales = nn.Linear(self.hidden_dim, 3)

        elif self.encoding_name == "hexplane":
            self.mlp_means = nn.Sequential(
                nn.Linear(self.feature_out_output_dim+self.augment_dim, self.feature_out_output_dim),
                nn.ReLU(),
                nn.Linear(self.feature_out_output_dim, 3),
            )
            self.mlp_quats = nn.Sequential(
                nn.Linear(self.feature_out_output_dim+self.augment_dim, self.feature_out_output_dim),
                nn.ReLU(),
                nn.Linear(self.feature_out_output_dim, 4),
            )
            self.mlp_scales = nn.Sequential(
                nn.Linear(self.feature_out_output_dim+self.augment_dim, self.feature_out_output_dim),
                nn.ReLU(),
                nn.Linear(self.feature_out_output_dim, 3),
            )
            if self.augment_dim > 0:
                self.mlp_aug = nn.Sequential(
                    nn.Linear(self.feature_out_output_dim+self.augment_dim, self.feature_out_output_dim),
                    nn.ReLU(),
                    nn.Linear(self.feature_out_output_dim, self.augment_dim),
                )

            if self.resnet_init: 
                init_as_identity(self.mlp_means[-1])
                init_as_identity(self.mlp_quats[-1])
                init_as_identity(self.mlp_scales[-1])
                if self.augment_dim >0:
                    init_as_identity(self.mlp_aug[-1])


        else:
            self.mlp = mlp(input_dim=self.input_dim, output_dim=self.output_dim, hidden_dim=self.hidden_dim, hidden_depth=self.hidden_depth, act=self.non_linearity)
            self.mlp_means = nn.Linear(64, 3)
            self.mlp_quats = nn.Linear(64, 4)
            self.mlp_scales = nn.Linear(64, 3)
                
            # if self.resnet_init: #initialize last layer to zero
            #     print("using resnet last layer init")
            #     self.output_layer.weight.data.fill_(0.0)



    def forward(self, t, x):
        """
        x: All gaussian parameters (N, features)
        The encoded features can go here.
        The gaussians means are always located at x[..., :3]
        NOTE:
        Before integration
        1. The first call to this occurs when in before_integrate when evaluating f0
        2. The second call occurs when estimating f1 in the step size selection, based on an Euler estimate

        """
        self.nfe += 1
        if x.dim() == 3: #x is of shape (T,N,3), only occurs if 
            t = t.expand(x.shape[1], 1)  # (N, 1)
            t = t.unsqueeze(0).expand(x.shape[0], -1, -1)  # (T, N, 1)
            if self.use_absolute_encoding:
                t = t + self.absolute_times
        else: #x is of shape (N,3)
            t = t.expand(x.shape[0], 1)
        if self.encoding_name == "freq":
            means = x[...,:3]
            # remaining = x[..., 3:]
            encoded_means = self.encoding_network_pos(means) #this includes gaussian means
            encoded_time = self.encoding_network_time(t)
            if self.use_timenet:
                encoded_time = self.timenet(encoded_time)
            inp = torch.cat((encoded_means, encoded_time), dim=-1) #default we concat remaining
            if self.augment_dim > 0: 
                aug = torch.zeros(x.shape[0], self.augment_dim).to(self.device)
                inp = torch.cat([inp, aug], 1)
            hidden = self.mlp(inp)

        elif self.encoding_name == "hexplane":
            means = x[..., :3]
            # remaining = x[..., 3:]
            encoded_means = self.encoding_network_pos(means, t) 
            encoded_means = self.feature_out(encoded_means) #mix the hexplane features
            encoded_means = F.relu(encoded_means)
            hidden = encoded_means
            if self.augment_dim>0: #injecting the augmented dimensions alongside the learned features
                augmented_dimensions = x[..., -self.augment_dim:]
                hidden = torch.cat((hidden, augmented_dimensions), dim=-1)
        else:  #no encoding
            inp = x 
            hidden = self.mlp(inp)

        #derivatives
        dmeans = self.mlp_means(hidden)
        dquats = self.mlp_quats(hidden)
        dscales = self.mlp_scales(hidden)
        if self.augment_dim > 0:
            daug = self.mlp_aug(hidden)
            out = torch.cat((dmeans, dquats, dscales, daug), dim=-1)
        else:
            out = torch.cat((dmeans, dquats, dscales), dim=-1)
        return out


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None, act=None):
    """
    Build a simple MLP with some layers and some dimensions.
    """
    if act is None:
        act = nn.ReLU()
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), act]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), act]
        # mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


    
def init_as_identity(layer):
    """Initialize layer to output zeros (identity/residual initialization)"""
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)