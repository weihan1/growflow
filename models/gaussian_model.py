import torch
from torch import nn 
from torch import Tensor
import math
import matplotlib.pyplot as plt
from helpers.gsplat_utils import create_splats_with_optimizers
from gsplat import rasterization 
# from gsplat import _rasterization as rasterization #use this pytorch version to understand code
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.cuda._wrapper import spherical_harmonics
from typing import Optional, Tuple, Dict
from copy import deepcopy


class Gaussians(nn.Module):
    """
    3D Gaussians class.
    Initializes the splats and the optimizers.
    """

    def __init__(
        self,
        parser,
        first_mesh_path, #this is only used in blender_pts init
        init_type="sfm",
        app_opt=False,
        packed=False, #NOTE: important to set this to false
        antialiased=False,
        init_num_pts=100_000,
        init_extent=3.0,
        init_opacity=0.1,
        init_scale=1.0,
        scene_scale=1.0,
        sh_degree=3,
        camera_model="pinhole",
        sparse_grad=False,
        visible_adam=False,
        batch_size=1,
        feature_dim=None,
        deformed_params_list=None,
        scale_activation="exp",
        device="cuda",
        learn_mask=False
    ):

        super(Gaussians, self).__init__()
        self.parser = parser
        self.first_mesh_path = first_mesh_path
        self.init_type = init_type
        self.app_opt = app_opt
        self.packed = packed
        self.antialiased = antialiased
        self.init_num_pts = init_num_pts
        self.init_extent = init_extent
        self.init_opacity = init_opacity
        self.init_scale = init_scale
        self.scene_scale = scene_scale
        self.sh_degree = sh_degree
        self.camera_model = camera_model
        self.sparse_grad = sparse_grad
        self.visible_adam = visible_adam
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.device = device
        self.scale_activation = scale_activation
        assert self.scale_activation in ["exp", "softplus"], "scale activation must be one of exp or softplus"
        print(f"using {scale_activation}")
        self.deformed_params_list = deformed_params_list if deformed_params_list is not None else ["means", "scales", "quats"]
        self.learn_mask = learn_mask
        
        #deformed_params_dict is static after initialization
        self.splats, self.optimizers, self.param_feature_dim, self.deformed_params_dict = create_splats_with_optimizers(
                    parser=self.parser,
                    first_mesh_path=self.first_mesh_path,
                    init_type=self.init_type,
                    init_num_pts=self.init_num_pts,
                    init_extent=self.init_extent,
                    init_opacity=self.init_opacity,
                    init_scale=self.init_scale,
                    scene_scale=self.scene_scale,
                    sh_degree=self.sh_degree,
                    sparse_grad=self.sparse_grad,
                    visible_adam=self.visible_adam,
                    batch_size=self.batch_size,
                    feature_dim=self.feature_dim,
                    deformed_params_list=self.deformed_params_list,
                    device=self.device,
                    learn_mask=self.learn_mask
                )


    def activate_params(self, raster_params):
        """
        Activate the parameters in raster_params and return an updated raster_params
        -Apply sigmoid to opacities 
        -Exp the scales
        -And invert camtoworlds
        -No need to normalize quaternions
        -For the spherical harmonics, reshape them from (N, 48) -> (N, 16, 3) and rename the key to colors
        """
        need_activate_params = ["opacities", "scales", "viewmats", "shs"] #shs only appears if we are learning it

        new_params = {k:v for k,v in raster_params.items() if k not in need_activate_params} #those that do not need activation
        N = self.splats.means.shape[0] 
        # Then add activated parameters
        new_params.update({
            "opacities": torch.sigmoid(raster_params["opacities"]),
            "scales": torch.exp(raster_params["scales"]) if self.scale_activation=="exp" else torch.nn.Softplus()(raster_params["scales"]),
            "viewmats": torch.linalg.inv(raster_params["viewmats"]),
        })

        if "colors" not in new_params: #this occurs if we are learning color traj, since raster_params would have shs as key
            new_params.update({
                "colors": raster_params["shs"].reshape(N, -1, 3)
            })
        
        return new_params


    def freeze_splats(self):
        """
        Freeze the splats for dynamic training.
        """
        for k,v in self.splats.items():
            v.requires_grad = False


    def rasterize_splats_custom(
        self,
        means,
        quats,
        scales,
        raster_params,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """
        Rasterize splats with custom gaussian parameters. Use this during debugging phase when you
        mask out gaussians. Also assume these parameters are pre-activated
        Assume scene is 400, 400, sh degree is 3 and near/far planes are 2 and 6
        """
        scales = torch.exp(scales) if self.scale_activation=="exp" else torch.nn.Softplus()(scales) # [N, 3]
        opacities = torch.sigmoid(raster_params["opacities"])  # [N,]
        rasterize_mode = "antialiased" if self.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=raster_params["colors"],
            viewmats=torch.linalg.inv(raster_params["viewmats"]),  # [C, 4, 4]
            Ks=raster_params["Ks"],  # [C, 3, 3]
            width=400,
            height=400,
            packed=self.packed,
            absgrad=( #TODO: try setting this to true, could be helpful for floaters
                self.strategy.absgrad
                if isinstance(self.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.sparse_grad,
            rasterize_mode=rasterize_mode,
            camera_model=self.camera_model,
            sh_degree = 3,
            near_plane = 2.0,
            far_plane = 6.0
        )
        masks = None
        if masks is not None:
            render_colors[~masks] = 0

        C = 1
        sh_degree = 3
        camtoworlds=torch.linalg.inv(raster_params["viewmats"])  # [C, 4, 4]
        colors = raster_params["colors"]
        dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]  # [C, N, 3]
        masks = info["radii"] > 0  # [C, N]
        if colors.dim() == 3:
            # Turn [N, K, 3] into [C, N, K, 3]
            shs = colors.expand(C, -1, -1, -1)  # [C, N, K, 3]
        else:
            # colors is already [C, N, K, 3]
            shs = colors
        colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)
        info.update({"colors": colors})
        return render_colors, render_alphas, info

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """
        Rasterize the splats, if appeareance optimization is on, add the color contribution from the appearance 
        module with the color of the splat.

        Return 
        -Rasterized image
        -Rendered opacity mask 
        -Additional rendering infos
        """
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"]) if self.scale_activation=="exp" else torch.nn.Softplus()(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.packed,
            absgrad=( #TODO: try setting this to true, could be helpful for floaters
                self.strategy.absgrad
                if isinstance(self.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.sparse_grad,
            rasterize_mode=rasterize_mode,
            camera_model=self.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def rasterize_splats_masks(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """
        Follows the implementation of https://github.com/momentum-robotics-lab/deformgs
        """
        assert "masks" in self.splats, "need to be optimizing mask indices too"
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"]) if self.scale_activation=="exp" else torch.nn.Softplus()(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        colors = torch.sigmoid(self.splats["masks"].expand(-1, 3))  #(N,3)

        rasterize_mode = "antialiased" if self.antialiased else "classic"
        rendered_masks, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.packed,
            absgrad=( 
                self.strategy.absgrad
                if isinstance(self.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.sparse_grad,
            rasterize_mode=rasterize_mode,
            camera_model=self.camera_model,
            **kwargs,
        )
        return rendered_masks 

    def rasterize_splats_depths(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """
        """
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"]) if self.scale_activation=="exp" else torch.nn.Softplus()(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
        rasterize_mode = "antialiased" if self.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.packed,
            absgrad=( #TODO: try setting this to true, could be helpful for floaters
                self.strategy.absgrad
                if isinstance(self.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.sparse_grad,
            rasterize_mode=rasterize_mode,
            camera_model=self.camera_model,
            render_mode="RGB+ED",
            **kwargs,
        )
        return render_colors, render_alphas, info

        

    def rasterize_with_dynamic_params(self, pred_param_novel, raster_params, return_meta=False, activate_params=True):
        """
        Dynamically handle rasterization with different predicted parameters.
        
        Args:
            pred_param_novel: The predicted parameter(s) from neural ODE, [T, n_gaussians, n_features]
            deformed_params_dict: Dictionary indicating which parameter is being predicted (e.g., 'means', 'quats', etc.), also includes the shape 
            raster_params: Dictionary containing all fixed parameters.
        
        Returns:
            output image: (b, h, w, 3) and optional meta data
        """
        h, w = raster_params["height"], raster_params["width"]
        n_cams = len(raster_params["viewmats"])

        n_gaussians = pred_param_novel.shape[-2]
        n_features = pred_param_novel.shape[-1]

        if pred_param_novel.dim() == 4: #this occurs if we have init_conditions > 1, in which case we can simply just reshape 
            pred_param_novel = pred_param_novel.reshape(-1, n_gaussians, n_features)

        temp_batch_size = pred_param_novel.shape[0] if pred_param_novel.dim() >= 2 else 1


        final_image = torch.zeros(n_cams, temp_batch_size, h, w, 3, device="cuda") if temp_batch_size > 1 else torch.zeros(n_cams, h, w, 3, device="cuda")
        final_alphas = torch.zeros(n_cams, temp_batch_size, h, w, 1, device="cuda") if temp_batch_size > 1 else torch.zeros(n_cams, h, w, 1, device="cuda")

        #TODO: fix issue where we neeed to conduct ablations
        temp_raster_params = deepcopy(raster_params)
        lst_of_metas = []
        for i in range(temp_batch_size): 
            #We know a priori that means,quat,scales are what we deform
            temp_raster_params["means"] = pred_param_novel[i,..., 0:3]
            temp_raster_params["quats"] = pred_param_novel[i,..., 3:7]
            temp_raster_params["scales"] = pred_param_novel[i,..., 7:10]

            if activate_params:
                new_raster_params = self.activate_params(temp_raster_params) #this activates all, which includes the deformed parameters as well
            else:
                new_raster_params = raster_params
            renders, alphas, meta = rasterization(**new_raster_params) #renders[0] is (C, H, W, 3)
            lst_of_metas.append(meta)
            if temp_batch_size > 1:
                final_image[:, i] = renders
                final_alphas[:, i] = alphas
            else:
                final_image = renders
                final_alphas = alphas

        if return_meta:  
            return final_image, final_alphas, lst_of_metas 
        else:
            return final_image, final_alphas 


    def rasterize_with_dynamic_params_batched(self, pred_param_novel, raster_params, return_meta=False, activate_params=True):
        """
        Dynamically handle rasterization with different predicted parameters.
        
        Args:
            pred_param_novel: The predicted parameter(s) from neural ODE, [T, n_gaussians, n_features]
            deformed_params_dict: Dictionary indicating which parameter is being predicted (e.g., 'means', 'quats', etc.), also includes the shape 
            raster_params: Dictionary containing all fixed parameters.
        
        Returns:
            output image: (b, h, w, 3) and optional meta data
        """
        #TODO: here u need to specify the background color as part of the rasterization argument.
        h, w = raster_params["height"], raster_params["width"]
        n_cams = len(raster_params["viewmats"])

        n_gaussians = pred_param_novel.shape[-2]
        n_features = pred_param_novel.shape[-1]

        if pred_param_novel.dim() == 4: #this occurs if we have init_conditions > 1, in which case we can simply just reshape 
            pred_param_novel = pred_param_novel.reshape(-1, n_gaussians, n_features)

        temp_batch_size = pred_param_novel.shape[0] if pred_param_novel.dim() >= 2 else 1

        temp_raster_params = deepcopy(raster_params)
        #We know a priori that means,quat,scales are what we deform
        temp_raster_params["means"] = pred_param_novel[..., 0:3] #(T,N,3)
        temp_raster_params["quats"] = pred_param_novel[..., 3:7] #(T,N,4)
        temp_raster_params["scales"] = pred_param_novel[..., 7:10] #(T,N,3)

        temp_raster_params["opacities"]= raster_params["opacities"][None].expand(temp_batch_size,-1) #(T,N)
        temp_raster_params["colors"] = raster_params["colors"][None].expand(temp_batch_size, -1,-1,-1) #(T,N,K,3)
        temp_raster_params["viewmats"] = raster_params["viewmats"][None].expand(temp_batch_size,-1,-1,-1) #(T,C,4,4)
        temp_raster_params["Ks"] = raster_params["Ks"][None].expand(temp_batch_size,-1,-1,-1).to(temp_raster_params["viewmats"]) #(T,C,3,3)
        temp_raster_params["backgrounds"] = raster_params["backgrounds"][None].expand(temp_batch_size,-1,-1) #(T,C,3)

        if activate_params:
            new_raster_params = self.activate_params(temp_raster_params) #this activates all, which includes the deformed parameters as well
        else:
            new_raster_params = temp_raster_params 

        #make sure this matches the rasterization params in the static recon
        new_raster_params["packed"] = self.packed #the default packed argument is True
        new_raster_params["absgrad"] = self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False
        new_raster_params["sparse_grad"] = self.sparse_grad
        new_raster_params["camera_model"] = self.camera_model
 
        renders, alphas, lst_of_metas = rasterization(**new_raster_params) #renders is of shape (T,C, H,W,3)
        final_image = renders.transpose(1,0) #(C,T,H,W,3)
        final_alphas = alphas.transpose(1,0)

        if return_meta:  
            return final_image, final_alphas, lst_of_metas 
        else:
            return final_image, final_alphas 

    def rasterize_with_dynamic_params_custom(self, pred_param_novel, raster_params, return_meta=False, activate_params=True):
        """
        Dynamically handle rasterization with different predicted parameters.
        
        Args:
            pred_param_novel: The predicted parameter(s) from neural ODE, [T, n_gaussians, n_features]
            deformed_params_dict: Dictionary indicating which parameter is being predicted (e.g., 'means', 'quats', etc.), also includes the shape 
            raster_params: Dictionary containing all fixed parameters.
        
        Returns:
            output image: (b, h, w, 3) and optional meta data
        """

        n_gaussians = pred_param_novel.shape[-2]
        n_features = pred_param_novel.shape[-1]

        if pred_param_novel.dim() == 4: #this occurs if we have init_conditions > 1, in which case we can simply just reshape 
            pred_param_novel = pred_param_novel.reshape(-1, n_gaussians, n_features)

        temp_batch_size = pred_param_novel.shape[0] if pred_param_novel.dim() >= 2 else 1

        temp_raster_params = deepcopy(raster_params)
        #We know a priori that means,quat,scales are what we deform
        temp_raster_params["means"] = pred_param_novel[..., 0:3] #(T,N,3)
        temp_raster_params["quats"] = pred_param_novel[..., 3:7] #(T,N,4)
        temp_raster_params["scales"] = pred_param_novel[..., 7:10] #(T,N,3)

        temp_raster_params["opacities"]= raster_params["opacities"][None].expand(temp_batch_size,-1) #(T,N)
        temp_raster_params["colors"] = raster_params["colors"][None].expand(temp_batch_size, -1,-1,-1) #(T,N,K,3)
        temp_raster_params["viewmats"] = raster_params["viewmats"][None].expand(temp_batch_size,-1,-1,-1) #(T,C,4,4)
        temp_raster_params["Ks"] = raster_params["Ks"][None].expand(temp_batch_size,-1,-1,-1).to(temp_raster_params["viewmats"]) #(T,C,3,3)

        if activate_params:
            new_raster_params = self.activate_params(temp_raster_params) #this activates all, which includes the deformed parameters as well
        else:
            new_raster_params = temp_raster_params 
        renders, alphas, lst_of_metas = rasterization(**new_raster_params) #renders is of shape (T,C, H,W,3)
        final_image = renders.transpose(1,0) #(C,T,H,W,3)
        final_alphas = alphas.transpose(1,0)

        if return_meta:  
            return final_image, final_alphas, lst_of_metas 
        else:
            return final_image, final_alphas 
    

    

    def rasterize_all_times(self, pred_param , raster_params, iteration, name="debug", cam_index=0):
        """
        Integrate all times using raster_params from canonical_params
        Use this for debugging the learned image trajectory
        """
        import imageio
        import numpy as np
        to8b = lambda x : (255*np.clip(x.detach().cpu().numpy(),0,1)).astype(np.uint8)
        out_img, alphas = self.rasterize_with_dynamic_params(pred_param, raster_params, activate_params=True) 
        #out_img (N, T, H, W, 3)
        num_images = out_img.shape[0] 
        num_timesteps = out_img.shape[1]
        video_duration = 3
        cam_i_image = out_img[cam_index]
        rendered_lst = []
        for image in cam_i_image:
            rendered_lst.append(to8b(image))
        name = f"cam_{cam_index}_all_it{iteration}.mp4" 
        imageio.mimwrite(name, rendered_lst, fps=num_timesteps/video_duration) 
        # plt.imshow(alphas.cpu().numpy())
        # plt.savefig(f"predicted_alphas_{name}_{iteration}")
        # plt.close()


    def down_proj(self, means, viewmats, Ks, covars=None, return_depth=False):
        """
        Project gaussians means from 3D coords down to 2D.
        NOTE: the R is from the viewmatrix not the R from quaternion.
        
        Args:
        -means: (T, N, 3), 3D means in world coords 
        -viewmats: [4,4] world2cam 
        -Ks: intrinsics
        -covars: (T, N, 3,3), 3D covariances in world coords

        Use the 2d conics to compare against the covariances from the rasterization.
        """
        if means.dim() == 2:
            means = means.unsqueeze(0)

        #world -> cam
        R = viewmats[:, :3, :3]  # [1, 3, 3]
        t = viewmats[:, :3, 3]  # [1, 3]
        means_c = torch.einsum("cij,cnj->cni", R, means) + t[:, None, :]  # (T, N, 3)

        #cam -> coords
        tx, ty, tz = torch.unbind(means_c, dim=-1) 
        means2d = torch.einsum("cij,cnj->cni", Ks[:, :2, :3], means_c)  # [T, N, 2]
        means2d = means2d / tz[..., None]  # [T, N, 2]

        if covars is not None:
            raise NotImplementedError
        #     height, width = self.imgs[0].shape[0], self.imgs[0].shape[1]
        #     covars_c = torch.einsum("cij,cnjk,clk->cnil", R, covars, R)  # [T, N, 3, 3]
        #     _, covars2d = _persp_proj(means_c, covars_c, Ks, width, height) #bit inefficient but wtv
        #     eps2d = 0.3
        #     covars2d = covars2d + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d
        #     det = (
        #         covars2d[..., 0, 0] * covars2d[..., 1, 1]
        #         - covars2d[..., 0, 1] * covars2d[..., 1, 0]
        #     )
        #     det = det.clamp(min=1e-10)

        #     conics = torch.stack(
        #         [
        #             covars2d[..., 1, 1] / det,
        #             -(covars2d[..., 0, 1] + covars2d[..., 1, 0]) / 2.0 / det,
        #             covars2d[..., 0, 0] / det,
        #         ],
        #         dim=-1,
        #     )  # [C, N, 3]
        #     return means2d, covars2d, conics
        return means2d
    
    

    def rasterize_quick(self,c2w, K):
        """
        Quickly rasterize, assume image dimensions is (400,400)
        """
        renders, alphas, info = self.rasterize_splats(
            camtoworlds=c2w,
            Ks=K,
            width=400,
            height=400,
            sh_degree=3,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
        )

        return renders

        
    def rasterize_quick_captured(self, c2w, K, height, width):
        """
        Quickly rasterize
        """
        renders, alphas, info = self.rasterize_splats(
            camtoworlds=c2w,
            Ks=K,
            width=width,
            height=height,
            sh_degree=3,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
        )

        return renders, alphas, info
        