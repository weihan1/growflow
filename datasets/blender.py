from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Literal, Union
import imageio.v2 as imageio
import cv2
import numpy as np
import torch
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset
from collections import defaultdict
import random
from copy import deepcopy
import os
from helpers.pc_viz_utils import visualize_point_cloud
from helpers.gsplat_utils import map_cont_to_int

@dataclass
class Static_Dataset:
    data_dir: str
    split: Literal["train", "test", "val"] = "train"

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        transforms_path = self.data_dir / f"transforms_{self.split}.json"
        with transforms_path.open("r") as transforms_handle:
            transforms = json.load(transforms_handle)
        image_ids = []
        cam_to_worlds = []
        images = []
        for frame in transforms["frames"]:
            image_id = frame["file_path"].replace("./", "")
            image_ids.append(image_id)
            file_path = self.data_dir / f"{image_id}.png"
            images.append(imageio.imread(file_path))

            c2w = torch.tensor(frame["transform_matrix"])
            # Convert from OpenGL to OpenCV coordinate system
            c2w[0:3, 1:3] *= -1 #flip sign for y and z
            cam_to_worlds.append(c2w)

        self.image_ids = image_ids
        self.cam_to_worlds = cam_to_worlds
        self.images = images

        # all renders have the same intrinsics
        # see also
        # https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/blender_dataparser.py
        image_height, image_width = self.images[0].shape[:2]
        cx = image_width / 2.0
        cy = image_height / 2.0
        fl = 0.5 * image_width / np.tan(0.5 * transforms["camera_angle_x"])
        self.intrinsics = torch.tensor(
            [[fl, 0, cx], [0, fl, cy], [0, 0, 1]], dtype=torch.float32
        )
        self.image_height = image_height
        self.image_width = image_width

        # compute scene scale (as is done in the colmap parser)
        camera_locations = np.stack(self.cam_to_worlds, axis=0)[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data = dict(
            K=self.intrinsics,
            camtoworld=self.cam_to_worlds[item],
            image=torch.from_numpy(self.images[item]).float(),
            image_id=item,
        )
        return data

class Dynamic_Dataset(Dataset):
    """
    A dataset class for dynamic synthetic blender data.
    Essentially, assume the folder contains T timesteps of N cameras with the usual Blender settings.
    Assume the structure of the dataset is as follows, each camera has SAME number of timesteps.
    --train
    ----r_1
    ----r_2
    ----...
    --test
    ----r_0
    ----r_15
    --transforms_train.json
    Essentially r0,r1,r2,r3,r4... are different folder views and inside they contain all the images for that view. 
    transforms_train.json contains the camera params for all NxT images. All images take from each of the N cameras 
    will have the same parameter but with different time parameter.
    Organize the images as a dictionary, where each key represents a specific view and append all images to that view.
    """
    #TODO: Use multithreading to load images here
    def __init__(
        self,
        data_dir: str,
        split: Literal["train", "test", "val"] = "train",
        is_reverse: bool = False,
        downsample_factor: float = 1,
        prepend_zero=True,
        include_zero=False,
        downsample_eval=True,
        target_shape=(400, 400),
        cam_batch_size=-1,
        half_normalize=False,
        bkgd_color=[0,0,0],
        return_mask=False
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.prepend_zero = prepend_zero #true during training, false during eval
        self.include_zero = include_zero
        self.is_reverse = is_reverse #if this is set to true, reverse the images along the time dimension
        self.every_n = int(1/downsample_factor) #downsampling time
        self.downsample_eval = downsample_eval
        self.cam_batch_size = cam_batch_size
        self.is_monocular = "monocular" in str(self.data_dir)
        self.half_normalize = half_normalize #normalizes time between [0,0.5]
        self.bkgd_color = bkgd_color
        self.return_mask = return_mask
        
        if self.half_normalize: 
            print("normalizing our time from [0,0.5]")  #only useful in learn per frame deformation
        else:
            print("normalizing our time from [0,1]") 

        assert self.every_n >= 1, "downsample_factor must be b/w 0 and 1"
        # print(f"in the {split} dataset, we sample every {self.every_n} frames!")
        transforms_path = self.data_dir / f"transforms_{self.split}.json"
        with transforms_path.open("r") as transforms_handle:
            transforms = json.load(transforms_handle)

        cam_to_worlds_dict = {} #we just need on cam to worlds per view
        images = defaultdict(list) #(camera_idx: T, H, W, C)
        masks = defaultdict(list)#(camera_idx: T, H, W, C)
        alpha_masks = defaultdict(list)
        image_ids_dict = defaultdict(list)
        images_ids_lst = [] #only used for counting
        #TODO: need to use multithreading here to load faster, for now it's fine
        scene = self.data_dir.stem.split("_transparent_final")[0] 
        gt_tracks_path = self.data_dir / "meshes" / f"relevant_{scene}_meshes" / "trajectory_frames.npz"
        gt_tracks = np.load(gt_tracks_path)
        gt_tracks_arr = [gt_tracks[key] for key in sorted(gt_tracks.files)]
        # visualize_point_cloud(gt_tracks_arr[-1])

        # First pass: count frames per camera and collect all times
        frames_per_camera = defaultdict(int)
        all_times = []
        for frame in transforms["frames"]:
            camera_id = frame["file_path"].split("/")[-2]
            frames_per_camera[camera_id] += 1
            if camera_id == list(frames_per_camera.keys())[0]:  # Collect times from first camera only
                all_times.append(frame["time"])

        # Determine which frame indices to keep per camera
        if self.every_n > 1:
            indices_to_keep = {}
            for camera_id, total in frames_per_camera.items():
                indices = set(range(0, total, self.every_n))
                indices.add(total-1) #include the last frame
                indices_to_keep[camera_id] = indices
        else:
            indices_to_keep = None #no subsampling

        # Second pass: load frames with subsampling
        camera_frame_counters = defaultdict(int)
        times_dict = defaultdict(list)
        progress_bar = tqdm(total=len(transforms["frames"]), desc=f"Loading the {scene} dataset")
        for frame in transforms["frames"]: 
            image_ids = frame["file_path"].replace("./", "")
            file_path = self.data_dir / image_ids
            mask_path = self.data_dir / "masks" / image_ids
            camera_id = image_ids.split("/")[-2]
            frame_idx = camera_frame_counters[camera_id]
            # Check if we should skip this frame
            if indices_to_keep is not None and frame_idx not in indices_to_keep[camera_id]:
                camera_frame_counters[camera_id] += 1
                progress_bar.update(1)
                continue
            
            camera_frame_counters[camera_id] += 1
            img = imageio.imread(file_path)
            img = cv2.resize(img, target_shape)
            try:
                mask = imageio.imread(mask_path) 
                mask = cv2.resize(mask, target_shape)
                mask = (mask /255.0).astype(np.float32)
                masks[camera_id].append(mask)
            except FileNotFoundError:
                mask = None
            if img.shape[-1] == 4:  # Has alpha channel
                rgb = img[..., :3] / 255.0 
                alpha = img[..., 3:4] / 255.0  
                norm_img = rgb * alpha + np.array(bkgd_color) * (1 - alpha)
            else:
                norm_img = img[..., :3] / 255.0
            norm_img = np.clip(norm_img, 0, 1) 
            images[camera_id].append(norm_img)
            alpha_masks[camera_id].append(alpha)
            image_ids_dict[camera_id].append(image_ids)
            times_dict[camera_id].append(frame["time"]) 

            c2w = torch.tensor(frame["transform_matrix"])
            c2w[0:3, 1:3] *= -1  # Convert from OpenGL to OpenCV

            cam_to_worlds_dict[camera_id] = c2w
            progress_bar.update(1)
            images_ids_lst.append(image_ids)

        progress_bar.close()

        mesh_path = self.data_dir / "meshes" / "relevant_meshes"
        all_meshes = sorted(os.listdir(mesh_path))
        self.relevant_mesh_path = self.data_dir / "meshes" / f"relevant_{scene}_meshes"
        self.unique_mesh_indices = self.data_dir / "unique_mesh_indices.npy"
        self.gt_tracks = gt_tracks_arr 
        self.first_mesh_path = mesh_path / all_meshes[-1]
        self.images_ids_lst = images_ids_lst 
        self.image_ids_dict = image_ids_dict #have a list that contain all images files
        self.cam_to_worlds_dict = cam_to_worlds_dict
        self.images = images
        self.masks = masks
        self.alpha_masks = alpha_masks
        self.times = times_dict[list(times_dict.keys())[0]]  # Get times from first camera

        # all renders have the same intrinsics
        # see also
        # https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/blender_dataparser.py
        image_height, image_width = self.images[list(self.images.keys())[0]][0].shape[:2] #select the first image from first view
        cx = image_width / 2.0
        cy = image_height / 2.0
        fl = 0.5 * image_width / np.tan(0.5 * transforms["camera_angle_x"])
        self.intrinsics = torch.tensor(
            [[fl, 0, cx], [0, fl, cy], [0, 0, 1]], dtype=torch.float32
        )
        self.image_height = image_height
        self.image_width = image_width

        # compute scene scale (as is done in the colmap parser)
        self.unique_cameras_lst = [v for _, v in self.cam_to_worlds_dict.items()]
        camera_locations = np.stack(self.unique_cameras_lst, axis=0)[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)
        self.num_cams = len(self.unique_cameras_lst)
        assert cam_batch_size <= self.num_cams, "please choose a camera batch size <= to number of cameras"

        print(f"The {self.split} dataset contains {len(images.keys())} cameras and {self.num_timesteps()} timesteps")

    def __len__(self):
        return len(self.images_ids_lst)

    def getfirstcam(self, timesteps: Union[list[int], np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Only output the first camera of all timesteps. Useful for debugging stuff.
        """
        c2w_list = []
        img_list = []
        #Specifically this is the part that will have to change
        available_cameras = [list(self.images.keys())[0]]
        
        if self.include_zero:
            if 0 not in timesteps:
                timesteps.insert(0,0)
        #TODO: this function won't work with monocular setting since self.images[camera_id] won't have timestep t
        for camera_id in available_cameras: #iterate over all cameras
            c2w_list.append(self.cam_to_worlds_dict[camera_id])
            selected_images = [torch.from_numpy(self.images[camera_id][t]).float() for t in timesteps] #(T, H, W, 3)
            img_list.append(torch.stack(selected_images))

                       
        c2ws = torch.stack(c2w_list) #(N, 4, 4)
        gt_images = torch.stack(img_list) #(N, T, H, W, 3)
        cont_t = torch.tensor([self.times[t] for t in timesteps])
        if self.half_normalize:
            cont_t *= 0.5

        if not self.include_zero: #if we included zero earlier we dont include it now.
            if self.prepend_zero:
                inp_t = torch.concat((torch.tensor([0]), cont_t), dim=0)
            else:
                inp_t = cont_t 
        else:
            inp_t = cont_t
        return c2ws, gt_images, inp_t

    def __getitem__(self, timestep: int) -> Dict[str, Any]:
        """
        Retrieve data for a single timestep. 
        NOTE: For single timestep, we output a dictionary, incompatible with raster_params
        """
        data = {
            "K": self.intrinsics,
            "camtoworld": {},
            "image": {},
            "image_id": {}
        }
        if self.return_mask:
            data["mask"] = {}
        for camera_id, img_list in self.images.items():
            if timestep >= len(img_list):
                raise IndexError(f"Timestep {timestep} out of range for camera {camera_id}")
            data["camtoworld"][camera_id] = self.cam_to_worlds_dict[camera_id]
            data["image"][camera_id] = torch.from_numpy(img_list[timestep]).float()
            data["image_id"][camera_id] = self.image_ids_dict[camera_id][timestep]
            if self.return_mask:
                data["mask"][camera_id] = self.masks[camera_id][timestep]
        return data
    
    def __getitems__(self, timesteps: Union[list[int], np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        """ 
        Retrieve data for multiple timesteps as a batch.
        If concat_zero is set to True, prepend inp_t with zero.
        Returns:
            Tuple:
            - `c2ws`: (N, 4, 4) tensor of camera-to-world matrices for all views.
            - `gt_images`: (N, T, H, W, C) tensor of images for all views at selected timesteps.
        """
        c2w_list = []
        img_list = []
        masks_list = []
        #Specifically this is the part that will have to change
        if self.cam_batch_size == -1: #if cam_batch_size = -1, u have all available cameras to use
            available_cameras = list(self.images.keys()) 
        else:
            available_cameras = random.sample(list(self.images.keys()), self.cam_batch_size)    
        
        if self.include_zero:
            if 0 not in timesteps:
                timesteps = [0] + list(timesteps)
        for camera_id in available_cameras: #iterate over all camera indices
            c2w_list.append(self.cam_to_worlds_dict[camera_id])
            selected_images = [torch.from_numpy(self.images[camera_id][t]).float() for t in timesteps] #(T, H, W, 3)
            if self.return_mask:
                selected_masks = [torch.from_numpy(self.masks[camera_id][t]).float() for t in timesteps]
                masks_list.append(torch.stack(selected_masks))
            img_list.append(torch.stack(selected_images))
                       
        c2ws = torch.stack(c2w_list) #(N, 4, 4)
        gt_images = torch.stack(img_list) #(N, T, H, W, 3)
        cont_t = torch.tensor([self.times[t] for t in timesteps])
        if self.half_normalize:
            cont_t *= 0.5

        if not self.include_zero: #if we included zero earlier we dont include it now.
            if self.prepend_zero:
                inp_t = torch.concat((torch.tensor([0]), cont_t), dim=0)
                int_t = torch.concat((torch.tensor([0]), torch.tensor(timesteps)), dim=0)
            else:
                inp_t = cont_t 
                int_t = torch.tensor(timesteps) 
        else:
            inp_t = cont_t
            int_t = torch.tensor(timesteps) 
        if self.return_mask:
            gt_masks = torch.stack(masks_list)
            return c2ws, gt_images, inp_t, int_t, gt_masks
        return c2ws, gt_images, inp_t, int_t, None
        
    def num_timesteps(self) -> int:
        """
        Return the number of timesteps, assuming each cameras has the same number of timesteps.
        """
        if self.is_monocular: #this part a bit ugly but just takes the image number
            return int(self.image_ids_dict["r_1"][0].split("/")[-1].split(".")[0]) + 1
        else:
            first_camera = next(iter(self.images.values()))  # Get the first camera's image list
            return len(first_camera)
    
    def custom_collate_fn(self, batch):
        """
        Custom collate function for the DynamicDataset's __getitems__ method.
        Args:
        -batch: tuple of c2ws, gt_images, inp_t where c2ws is of shape (N, 4, 4) and gt_images is of shape
        (N, T_batch, 400, 400, 4), and inp_t is of shape (T_batch+1,)
        NOTE: if you sort the inp_t, then need to sort the gt images.
        """
        c2ws_batch = batch[0]
        gt_images_batch = batch[1]
        inp_t = batch[2]
        int_t = batch[3]
        if self.return_mask:
            gt_masks_batch = batch[4]

        #Sort inp_t without the first element (which is 0)
        t0, inp_t_to_sort = inp_t[0], inp_t[1:]
        sorted_inp_t, indices = inp_t_to_sort.sort()

        #Sort the image batch
        gt_images_batch = gt_images_batch[:, indices, ...]
        if self.return_mask:
            gt_masks_batch = gt_masks_batch[:, indices, ...]
        int_t0, int_t_to_sort = int_t[0], int_t[1:]
        sorted_int_t = int_t_to_sort[indices]

        #Recreate the new inp_t
        new_inp_t = torch.cat((t0.unsqueeze(0), sorted_inp_t), dim=0)
        new_int_t = torch.cat((int_t0.unsqueeze(0), sorted_int_t), dim=0)
        # int_t = map_cont_to_int(new_inp_t,self.num_timesteps())

        if self.return_mask:
            return c2ws_batch, gt_images_batch, new_inp_t, new_int_t, gt_masks_batch
        else: 
            return c2ws_batch, gt_images_batch, new_inp_t, new_int_t, None 



class SingleTimeDataset(Dataset):
    """
    Custom single time dataset for specific timestep training.
    len() would return the number of cameras for that timestep.
    Indexing into this dataset will return one of the cameras for that timestep.
    Can now combine multiple datasets (e.g., train and test).
    """
    def __init__(self, dataset, index):
        """
        Args:
            datasets: Either a single Dynamic_Dataset or a list of Dynamic_Dataset objects
            index: The timestep index to extract from each dataset
        """
        
        self.dataset = dataset
        self.index = index
        
        # Combine all camera data from all datasets
        self.combined_data = {
            "K": dataset.intrinsics,  # Assuming all datasets have same intrinsics
            "camtoworld": {},
            "image": {},
            "image_id": {}
        }
        if self.dataset.return_mask:
            self.combined_data["mask"] = {}
        
        timestep_data = dataset[index] 
        for cam_id in timestep_data["camtoworld"].keys():
            # Fixed: Use cam_id as key instead of overwriting
            self.combined_data["camtoworld"][cam_id] = timestep_data["camtoworld"][cam_id]
            self.combined_data["image"][cam_id] = timestep_data["image"][cam_id]
            self.combined_data["image_id"][cam_id] = timestep_data["image_id"][cam_id]
            if self.dataset.return_mask:
                self.combined_data["mask"][cam_id] = timestep_data["mask"][cam_id]
        
        self.camera_ids = list(timestep_data["camtoworld"].keys())
    
    def __len__(self):
        return len(self.camera_ids)
    
    def __getitem__(self, idx):
        cam_id = self.camera_ids[idx]
        
        data = dict(
            K=self.combined_data["K"],
            camtoworld=self.combined_data["camtoworld"][cam_id],
            image=self.combined_data["image"][cam_id],
            image_id=idx,  # Or use cam_id if you prefer the actual camera identifier
        )
        if self.dataset.return_mask:
            data["mask"] = self.combined_data["mask"][cam_id]
        
        return data