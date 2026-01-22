#General imports
import os
import sys
import torch 
import matplotlib.pyplot as plt
from torch import nn
from lib.torchdiffeq import odeint_adjoint, odeint

#neural ode function
from models.neural_ode_func import Simple_ODE
from models.encoding import get_embedder, scale_anything 
from models.hexplane import HexPlaneField
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ODEBlock(nn.Module):
    """Solves ODE defined by odefunc.

    Parameters
    ----------
    device : torch.device

    odefunc : ODEFunc instance or anode.conv_models.ConvODEFunc instance
        Function defining dynamics of system.

    is_conv : bool
        If True, treats odefunc as a convolutional model.

    tol : float
        Error tolerance.

    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    
    event_fn: nn.Module
        Boundary condition
    """
    def __init__(self, odefunc, rtol=1e-4, atol=1e-4, adjoint=False, method="dopri5", event_fn=None):
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.device = "cuda"
        self.odefunc = odefunc
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.event_fn = event_fn

    def forward(self, x, eval_times=None):
        """Solves ODE starting from x.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)

        eval_times : None or torch.Tensor
            If None, returns solution of ODE at final time t=1. If torch.Tensor
            then returns full ODE trajectory evaluated at points in eval_times.

        If augment_dim > 0, append zeros to both the G(0) AND the input to ur velocity network f_theta 
        Otherwise, for other types of encoding, you simply need to encode it in neural_ode_func
        """
        self.odefunc.nfe = 0

        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)

        if self.odefunc.augment_dim > 0:
            # Add augmentation
            aug = torch.zeros(x.shape[0], self.odefunc.augment_dim).to(self.device)
            # Shape (batch_size, data_dim + augment_dim)
            x_aug = torch.cat([x, aug], 1)

        else:
            x_aug = x

        # x_aug = x
        if self.adjoint:
            out = odeint_adjoint(self.odefunc, x_aug, integration_time,
                                 rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = odeint(self.odefunc, x_aug, integration_time,
                                 rtol=self.rtol, atol=self.atol, method=self.method) #(T, N, n_enc_features-3+feature_dim+augment_dim)

        out = out[..., :self.odefunc.feature_dim] 
        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out

    def trajectory(self, x, timesteps):
        """Returns ODE trajectory.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)

        timesteps : int
            Number of timesteps in trajectory.
        """
        integration_time = torch.linspace(0., 1., timesteps)
        return self.forward(x, eval_times=integration_time)





class DynamicalModel(nn.Module):
    """
    A dynamical model which comprises of a neural_ode and a encoder for positions, followed
    by a MLP decoder.
    """

    def __init__(
        self,
        data_dim,
        augment_dim,
        encoding,
        hidden_dim,
        hidden_depth,
        scene_scale,
        max_steps,
        non_linearity_name="relu",
        rtol=1e-3,
        atol=1e-4,
        adjoint=False,
        neural_ode_lr=1e-3,
        encoder_lr=1e-3,
        gamma=0.1,
        event_fn=None,
        concat_remaining=True,
        resnet_init =True,
        adjust_lr_w_scene=True,
        use_timenet = False,
        use_skip = False,
        data_type = "blender",
        multires = [1,2],
        spatial_temp_resolution = [64,64,64,25], #spatial temporal resolution
        method = "dopri5",
        feature_out_output_dim = 64
    ):

        super(DynamicalModel, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.augment_dim = augment_dim
        self.output_dim = data_dim 
        self.atol = atol 
        self.rtol = rtol 
        self.neural_ode_lr = neural_ode_lr
        self.encoder_lr = encoder_lr
        self.gamma = gamma
        self.event_fn = event_fn
        self.encoding = encoding
        self.concat_remaining = concat_remaining
        self.adjust_lr_w_scene = adjust_lr_w_scene
        self.use_timenet = use_timenet
        self.use_skip = use_skip
        self.scene_scale = scene_scale
        self.data_type = data_type
        self.multires = multires
        self.spatial_temp_resolution = spatial_temp_resolution
        self.method = method
        self.feature_out_output_dim = feature_out_output_dim
        assert self.encoding in ["freq", "hexplane", ""], f"encoding must be one of freq, ingp or None, it is set to {self.encoding}"
        print(f"integrating using {self.method}")


        if self.encoding == "hexplane":
            self.bounds = 1.6 #this value is useless we will reset the bounds later in the main code
            self.plane_tv_weight = 0.0001 # TV loss of spatial grid
            self.time_smoothness_weight = 0.01 # TV loss of temporal grid
            self.l1_time_planes = 0.0001  # TV loss of temporal grid
            self.kplanes_config = {
                                'grid_dimensions': 2,
                                'input_coordinate_dim': 4,
                                'output_coordinate_dim': 32,
                                'resolution': self.spatial_temp_resolution
                                }
            self.encoding_network_pos =  HexPlaneField(self.bounds, self.kplanes_config, self.multires)
            hexplane_output_dim = self.encoding_network_pos.feat_dim
            self.feature_out = nn.Linear(hexplane_output_dim, self.feature_out_output_dim) 
            encoding_network_time = None 
            encoder_output_dim = 64 - 3 #the 64 comes from the downsampling
            time_input_ch = 0
            xyz_input_ch = 0


        elif self.encoding == "freq": #encode position and time 
            #Values taken from deformable 3dgs
            x_multires = 10 
            t_multires = 6 #dont increqase
            encoding_network_time, time_input_ch = get_embedder(t_multires, 1)
            self.encoding_network_pos, xyz_input_ch = get_embedder(x_multires, 3)
            encoder_output_dim = xyz_input_ch + time_input_ch  #the reason we subtract 3 here is because we need to account for removing gaussian means  
            self.feature_out = None

        else:
            encoder_output_dim = 0 #just setting it to feature dim 
            time_input_ch = 0
            xyz_input_ch = 0
            self.encoding_network_pos = None
            encoding_network_time = None
            self.feature_out = None
            self.use_timenet = False

        if self.use_skip: 
            assert self.encoding == "freq", "skip only supports freq encoding"

        odefunc = Simple_ODE(
            feature_dim=data_dim,
            hidden_dim=hidden_dim,
            encoding_name = self.encoding,
            encoding_network_pos = self.encoding_network_pos,
            encoding_network_time = encoding_network_time,
            encoder_output_dim=encoder_output_dim,
            xyz_input_ch = xyz_input_ch,
            time_input_ch = time_input_ch,
            augment_dim=augment_dim,
            concat_remaining=concat_remaining, 
            non_linearity_name=non_linearity_name,
            hidden_depth=hidden_depth,
            resnet_init= resnet_init,
            feature_out = self.feature_out,
            use_timenet = self.use_timenet,
            use_skip = self.use_skip,
            feature_out_output_dim = self.feature_out_output_dim
        )

        self.odeblock = ODEBlock(odefunc, atol=atol, rtol=rtol, adjoint=adjoint, method=self.method, event_fn=event_fn)

        if self.encoding == "hexplane":
            self.hexplane_params = list(self.encoding_network_pos.parameters())
            self.neural_ode_params = [p for name, p in self.named_parameters() 
                            if not any(p is param for param in self.encoding_network_pos.parameters())]
            
            if self.adjust_lr_w_scene:
                scaled_encoder_lr = encoder_lr * scene_scale
                scaled_neural_ode_lr = neural_ode_lr * scene_scale
            else:
                scaled_encoder_lr = encoder_lr 
                scaled_neural_ode_lr = neural_ode_lr 
                
            self.hexplane_optimizer = torch.optim.Adam(self.hexplane_params, lr=scaled_encoder_lr)
            self.neural_ode_optimizer = torch.optim.Adam(self.neural_ode_params, lr=scaled_neural_ode_lr)
            #store them as dict so it's easier to save/load
            self.optimizers = {"neural_ode_optimizer": self.neural_ode_optimizer, 
                               "encoder_optimizer":self.hexplane_optimizer}

            hexplane_param_count = sum(p.numel() for p in self.hexplane_params)
            neural_ode_param_count = sum(p.numel() for p in self.neural_ode_params)
            total_param_count = hexplane_param_count + neural_ode_param_count
            
            print(f"HexPlane parameters: {hexplane_param_count:,}")
            print(f"Neural ODE parameters: {neural_ode_param_count:,}")
            print(f"Total parameters: {total_param_count:,}")
            print(f"Hexplane learning rate is set to {scaled_encoder_lr}")
            print(f"Neural ODE learning rate is set to {scaled_neural_ode_lr}")
            self.schedulers = { 
                "encoder_scheduler":torch.optim.lr_scheduler.ExponentialLR(
                    self.hexplane_optimizer, gamma=self.gamma ** (1.0 / max_steps)
                ), 

                "neural_ode_scheduler": torch.optim.lr_scheduler.ExponentialLR(
                    self.neural_ode_optimizer, gamma=self.gamma ** (1.0 / max_steps)
                ), 
            }

        
        else: #for other, there's no trainable encoders, just keep track one set of optimizers 
            self.neural_ode_params = [p for name, p in self.named_parameters()]
            neural_ode_param_count = sum(p.numel() for p in self.neural_ode_params)
            if adjust_lr_w_scene:
                scaled_neural_ode_lr = neural_ode_lr * self.scene_scale
            else:
                scaled_neural_ode_lr = neural_ode_lr  
            self.neural_ode_optimizer = torch.optim.Adam(self.neural_ode_params, lr=scaled_neural_ode_lr)
            self.optimizers = {"neural_ode_optimizer": self.neural_ode_optimizer}
            self.schedulers = { 
                "neural_ode_scheduler": torch.optim.lr_scheduler.ExponentialLR(
                    self.neural_ode_optimizer, gamma=self.gamma ** (1.0 / max_steps)
                ) 
            }
            print(f"using a model size of {neural_ode_param_count}")

    def forward(self, x, t):
        """
        Forward pass
        If using encoding and using augmented dimension, the order is [x, augment_dim]
        """
        pred = self.odeblock(x,eval_times=t) #(T,N,feature_dim+augment_dim)

        return pred

