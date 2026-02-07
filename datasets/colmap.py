import os
import json
from tqdm import tqdm
from typing import Any, Dict, List, Optional
from typing_extensions import assert_never

import cv2
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from pycolmap import SceneManager

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)
import random
import time

from helpers.gsplat_utils import save_point_cloud_to_ply
from typing import Union
import copy
from helpers.gsplat_utils import map_cont_to_int
from helpers.utils import get_divisors
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, mask_dir: str, resized_image_dir: str, resized_mask_dir: str, factor: int) -> str:
    """Resize image folder and corresponding masks."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_image_dir}.")
    os.makedirs(resized_image_dir, exist_ok=True)
    os.makedirs(resized_mask_dir, exist_ok=True)
    
    image_files = _get_rel_paths(image_dir)
    mask_files = _get_rel_paths(mask_dir)
    
    # dont use this in case hidden files creep up
    # assert len(image_files) == len(mask_files), "should have same number of masks and images"
    
    for image_file, mask_file in tqdm(zip(image_files, mask_files)):
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        #make sure everything ends in png
        resized_image_path = os.path.join(
            resized_image_dir, os.path.splitext(image_file)[0] + ".png"
        )
        #COLMAP ends in jpg.png, so need to splitext twice
        resized_mask_path = os.path.join(
            resized_mask_dir, os.path.splitext(os.path.splitext(mask_file)[0])[0] + ".png"
        )
        
        if os.path.isfile(resized_image_path) and os.path.isfile(resized_mask_path):
            continue
        
        if not os.path.isfile(resized_image_path):
            image = imageio.imread(image_path)[..., :3]  # Take only RGB channels
            resized_size = (
                int(round(image.shape[1] / factor)),  # width
                int(round(image.shape[0] / factor)),  # height
            )
            resized_image = np.array(
                Image.fromarray(image).resize(resized_size, Image.BICUBIC)
            )
            imageio.imwrite(resized_image_path, resized_image)
        
        if not os.path.isfile(resized_mask_path):
            mask = imageio.imread(mask_path)
            resized_mask_size = (
                int(round(mask.shape[1] / factor)),  # width
                int(round(mask.shape[0] / factor)),  # height
            )
            resized_mask = np.array(
                Image.fromarray(mask).resize(resized_mask_size, Image.NEAREST)  # Use NEAREST for masks
            )
            imageio.imwrite(resized_mask_path, resized_mask)
    
    return resized_image_dir, resized_mask_dir



class DynamicParser:
    """COLMAP parser for multiple timesteps.
    NOTE: must remember: timestep 0 must be when the plant is fully grown!!! 
    """

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        align_timesteps: bool = False,
        dates: list= ["08-07-2025"],
        use_dense: bool =False,
        subsample_factor: int=1,
        start_from: int=0,
        crop_imgs: bool=False, #for plants, might need to crop stuff
        use_crops: bool=False,
        use_bg_masks: bool=False,
        end_until:int =0,
        include_end: bool=False,
        white_bkgd: bool=False
    ):
        
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.align_timesteps = align_timesteps
        self.use_dense = use_dense
        self.start_from = start_from
        self.use_bg_masks = use_bg_masks
        self.end_until = end_until
        self.include_end = include_end
        self.white_bkgd = white_bkgd

        # Initialize storage for all timesteps
        self.timestep_data = [] #this stores per-timestep data
        self.global_transform = np.eye(4)
        
        # assert dates is not None, "please specify dates"
        if dates is None:
            print("dates are None, using all timelapse dates")
            dates = sorted([f for f in os.listdir(data_dir) if f.startswith("timelapse")])[::-1] #NOTE: very important to invert the list, because we compute the point clouds of fully_grown
            original_dates = deepcopy(dates)
        if subsample_factor > 1: 
            print(f"subsampling the timesteps with factor of  {subsample_factor}")
            dates = dates[::subsample_factor]
            if include_end: #force the last timestep to be part
                assert original_dates[-1] != dates[-1], "only use this when last date doesnt match"
                original_dates.append(dates[-1])
            assert original_dates[-1] == dates[-1], f"the last date doesnt match original. please select subsample factor {get_divisors(len(original_dates) - 1)} "
        if start_from > 0: #start later, actually makes no sense
            dates = dates[start_from:]
        if self.end_until > 0:
            print(f"end_until specified, ending until {-self.end_until}")
            dates = dates[:-self.end_until]
        assert all(dates[i] >= dates[i+1] for i in range(len(dates)-1)), "Dates not in decreasing order"
        #Make sure here u go from latest date to earliest date
        self.num_timesteps = len(dates)
        print(f"[DynamicParser] Processing {self.num_timesteps} timesteps...")

        # Process each timestep
        for t, dir_t in enumerate(dates):
            print(f"[DynamicParser] Processing timestep {t}: {dir_t}")
            full_data_dir = os.path.join(self.data_dir, dir_t)
            timestep_parser = self._parse_single_timestep(full_data_dir, t)
            self.timestep_data.append(timestep_parser)

        all_camtoworlds_lst = []
        all_points_lst = []
        for i in range(len(self.timestep_data)):
            all_camtoworlds_lst.append(self.timestep_data[i]["camtoworlds"])
            all_points_lst.append(self.timestep_data[i]["points"])
        all_camtoworlds_array = np.concatenate(all_camtoworlds_lst, axis=0)
        all_points_array = np.concatenate(all_points_lst, axis=0) #if we are doing the turntable setup, all_points_array will just be all the points for timestep 0

        if self.normalize: #normalize cameras to origin
            print("normalizing across all timesteps")
            T1 = similarity_from_cameras(all_camtoworlds_array)
            camtoworlds = transform_cameras(T1, all_camtoworlds_array)
            # visualize_point_cloud(points)
            points = transform_points(T1, all_points_array)
            # visualize_point_cloud(points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            print("no normalizing, using the original points and cameras")
            camtoworlds = all_camtoworlds_array  # Use original cameras if not normalizing
            points = all_points_array           # Use original points if not normalizing
            transform = np.eye(4)

        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        scene_scale = np.max(dists)

        # Split the transformed data back to individual timesteps
        start_cam = 0
        start_points = 0
        for i in range(len(self.timestep_data)):
            num_camera_i = all_camtoworlds_lst[i].shape[0]
            num_points_i = all_points_lst[i].shape[0]  # Fixed: was all_points_array[i].shape[0]
            
            # Extract the cameras and points for this timestep
            self.timestep_data[i]["camtoworlds"] = camtoworlds[start_cam:start_cam + num_camera_i]
            self.timestep_data[i]["points"] = points[start_points:start_points + num_points_i]
            
            # Store global parameters
            self.timestep_data[i]["transform"] = transform
            self.timestep_data[i]["scene_scale"] = scene_scale
            
            # Update start indices for next timestep
            start_cam += num_camera_i
            start_points += num_points_i

        point_cloud_sequence_untransformed = []
        legend_labels = []
        for t in range(len(self.timestep_data)):
            point_cloud_sequence_untransformed.append(self.timestep_data[t]["points"])
            legend_labels.append(self.timestep_data[t]["data_dir"].split("/")[-1])

        self._create_unified_data()
        print(f"[DynamicParser] Successfully loaded {self.num_timesteps} timesteps")


    def _parse_single_timestep(self, data_dir: str, timestep_idx: int):
        """Parse a single timestep using the original parser logic."""
        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        scene = data_dir.split("/")[-2] 
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= self.factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // self.factor, cam.height // self.factor)
            mask_dict[camera_id] = None

        print(f"[Timestep {timestep_idx}] {len(imdata)} images, taken by {len(set(camera_ids))} cameras.")

        if len(imdata) == 0:
            raise ValueError(f"No images found in COLMAP for timestep {timestep_idx}.")
        if not (type_ == 0 or type_ == 1):
            print(f"Warning: COLMAP Camera is not PINHOLE for timestep {timestep_idx}. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP
        image_names = [imdata[k].name for k in imdata]
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        #no point in using camera_ids cause we use the same camera
        camera_ids = [camera_ids[i] for i in inds]
        image_ids = inds #NOTE: these indices (which are used to index the cameras might NOT be sorted)
        assert sorted(image_names) == image_names, "make sure image_names is sorted"


        # Load extended metadata
        extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                extconf.update(json.load(f))

        # Load bounds
        bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            bounds = np.load(posefile)[:, -2:]

        # Load images
        if self.factor > 1 and not extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{self.factor}"
        else:
            image_dir_suffix = ""

        colmap_image_dir = os.path.join(data_dir, "images_still")
        image_dir = os.path.join(data_dir, "images_still" + image_dir_suffix)
        if self.use_bg_masks:
            masks_dir = os.path.join(data_dir, "masks_bg")
        else:
            masks_dir = os.path.join(data_dir, "masks")


        if self.factor > 1:
            resized_mask_dir = masks_dir + image_dir_suffix+"_png"
        else:
            resized_mask_dir = masks_dir 

        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        #remap colmap images names to images names
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        if self.factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir, mask_dir = _resize_image_folder(  # downsamples the images with png
                image_dir=colmap_image_dir,
                mask_dir=masks_dir,
                resized_image_dir=image_dir + "_png",
                resized_mask_dir = masks_dir+ image_dir_suffix+"_png",
                factor=self.factor,
            )
            image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]
        masks_paths = sorted([os.path.join(resized_mask_dir, f) for f in os.listdir(resized_mask_dir)])
        assert len(masks_paths) != 0, "we need masks,"
        # 3D points
        if timestep_idx == 0: #where we need the actual point clouds
            if self.use_dense: #using dense point cloud for sfm init
                import trimesh
                dense_folder = os.path.join(data_dir, "dense", "fused.ply")
                mesh = trimesh.load(dense_folder) 
                points = mesh.vertices
                points_rgb = mesh.visual.vertex_colors[...,:3] #discard alpha_channel
            else:
                points = manager.points3D.astype(np.float32)
                points_err = manager.point3D_errors.astype(np.float32)
                points_rgb = manager.point3D_colors.astype(np.uint8)
        else: #open the points, but not use them (maybe for viz or smth)
            points = manager.points3D.astype(np.float32)
            points_err = manager.point3D_errors.astype(np.float32)
            points_rgb = manager.point3D_colors.astype(np.uint8)
            

        # points_combined = torch.from_numpy(np.concatenate((points, (points_rgb/255)), axis=-1))
        # save_point_cloud_to_ply(points_combined, "sparse_points.ply")
        point_indices = dict()

        #NOt being used
        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Handle image size correction and undistortion (same as original)
        actual_image = imageio.imread(image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = imsize_dict[camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width 
        
        #NOTE: this mostly doesn't do anything, s_width should be 1
        for camera_id, K in Ks_dict.items(): 
            K[0, :] *= s_width
            K[1, :] *= s_height
            Ks_dict[camera_id] = K
            width, height = imsize_dict[camera_id]
            imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # Undistortion maps
        mapx_dict = dict()
        mapy_dict = dict()
        roi_undist_dict = dict()
        roi_crop_dict = dict()
        
        for camera_id in params_dict.keys():
            params = params_dict[camera_id]
            if len(params) == 0:
                continue
                
            K = Ks_dict[camera_id]
            width, height = imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]

            mapx_dict[camera_id] = mapx
            mapy_dict[camera_id] = mapy
            Ks_dict[camera_id] = K_undist
            roi_undist_dict[camera_id] = roi_undist
            imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            mask_dict[camera_id] = mask

        # Calculate scene scale

        # Return timestep data as a dictionary
        date = data_dir.split("/")[-1]
        return {
            'date': date,
            'data_dir': data_dir,
            'timestep_idx': timestep_idx,
            'image_names': image_names,
            'image_paths': image_paths,
            'masks_paths': masks_paths,
            'camtoworlds': camtoworlds,
            'camera_ids': camera_ids, #if same camera, this would be all ones
            'image_ids': image_ids, #should be using this to index into poses/images
            'Ks_dict': Ks_dict,
            'params_dict': params_dict, #distortion params
            'imsize_dict': imsize_dict, #used in static_render_traj
            'mask_dict': mask_dict,
            'points': points,
            # 'points_err': points_err,
            'points_rgb': points_rgb, #used for initialization
            # 'point_indices': point_indices,
            # 'transform': transform,
            # 'scene_scale': scene_scale,
            'bounds': bounds,
            'extconf': extconf,
            'mapx_dict': mapx_dict, #distortion params
            'mapy_dict': mapy_dict, #distortion params
            'roi_undist_dict': roi_undist_dict, #distortion params
            'num_images': len(image_names),
            'roi_crop_dict': roi_crop_dict
        }

    def _create_unified_data(self):
        """Create unified data structures across all timesteps.
        This is fine, since most parameters will only be used for static reconstruction, which 
        is only done for timestep t0.
        """
        # For backward compatibility, expose the first timestep's data at the top level
        first_timestep = self.timestep_data[0]
        for key, value in first_timestep.items():
            if key != 'timestep_idx':
                setattr(self, key, value)
        
        # Add timestep-specific access methods
        self.all_timesteps = self.timestep_data

    def get_timestep_data(self, timestep_idx: int):
        """Get data for a specific timestep."""
        if timestep_idx >= self.num_timesteps:
            raise IndexError(f"Timestep {timestep_idx} out of range [0, {self.num_timesteps-1}]")
        return self.timestep_data[timestep_idx]

    def get_image_at_timestep(self, timestep_idx: int, image_idx: int):
        """Get a specific image at a specific timestep."""
        timestep_data = self.get_timestep_data(timestep_idx)
        return timestep_data['image_paths'][image_idx]

    def get_cameras_at_timestep(self, timestep_idx: int):
        """Get camera poses for a specific timestep."""
        timestep_data = self.get_timestep_data(timestep_idx)
        return timestep_data['camtoworlds']

    def get_points_at_timestep(self, timestep_idx: int):
        """Get 3D points for a specific timestep."""
        timestep_data = self.get_timestep_data(timestep_idx)
        return timestep_data['points']


class Dynamic_Datasetshared():
    """
    Shared dynamic dataset that loads all images
    """
    def __init__(
        self,
        parser,  # Should be DynamicParser
        debug_data_loading=False,
        apply_mask =False,
    ):
        self.parser = parser
        self.apply_mask = apply_mask
        self.debug_data_loading = debug_data_loading
        self._load_all_data()
        
    
    def _load_all_data(self):
        """Load all images and camera data across all timesteps"""
        self.timestep_images = {}
        self.timestep_poses = {}
        self.timestep_intrinsics = {}
        self.timestep_masks = {}
        self.timestep_image_paths = {}
        
        camera_id = self.parser.timestep_data[0]["camera_ids"][0]
        
        def load_timestep(t):
            """Load all data for a single timestep - returns the data instead of modifying shared state"""
            timestep_data = self.parser.get_timestep_data(t)
            
            images = {}
            poses = {}
            intrinsics = {}
            masks = {}
            image_paths = []
            
            for i, (image_path, image_id, mask_path) in enumerate(zip(
                timestep_data['image_paths'], 
                timestep_data['image_ids'], #NOTE: this might NOT be ordered
                timestep_data["masks_paths"]
            )):
                # Load mask and invert it
                mask = imageio.imread(mask_path)
                if len(mask.shape) == 3:
                    mask = mask[..., 0]
                # inverted_mask = 255 - mask
                inverted_mask = mask 
                inverted_mask = (inverted_mask / 255.0).astype(np.float32)
                
                # Load and process image
                image = imageio.imread(image_path)[..., :3]
                image = (image / 255.0).astype(np.float32)

                # Handle undistortion
                params = timestep_data['params_dict'][camera_id]
                if len(params) > 0:
                    mapx, mapy = (
                        timestep_data['mapx_dict'][camera_id],
                        timestep_data['mapy_dict'][camera_id],
                    )
                    image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
                    inverted_mask = cv2.remap(inverted_mask, mapx, mapy, cv2.INTER_NEAREST)
                    
                    x, y, w, h = timestep_data['roi_undist_dict'][camera_id]
                    image = image[y : y + h, x : x + w]
                    inverted_mask = inverted_mask[y : y + h, x : x + w]
                
                if self.apply_mask:
                    if self.parser.white_bkgd:
                        image = image * inverted_mask[..., np.newaxis] + np.array([1,1,1]) * (1-inverted_mask[...,np.newaxis])
                    else:
                        image = image * inverted_mask[..., np.newaxis]

                images[image_id] = image
                masks[image_id] = inverted_mask
                poses[image_id] = timestep_data['camtoworlds'][i]
                intrinsics[camera_id] = timestep_data['Ks_dict'][camera_id] #NOTE: here the intrinsics have already been undistorted
                image_paths.append(os.path.basename(image_path))
            
            del timestep_data
            
            return {
                't': t,
                'images': images,
                'masks': masks,
                'poses': poses,
                'intrinsics': intrinsics,
                'image_paths': image_paths
            }
        
        # Load all timesteps in parallel
        if self.debug_data_loading:
            for t in tqdm(range(self.parser.num_timesteps)):
                result = load_timestep(t)
                self.timestep_images[t] = result['images']
                self.timestep_masks[t] = result['masks']
                self.timestep_poses[t] = result['poses']
                self.timestep_intrinsics[t] = result['intrinsics']
                self.timestep_image_paths[t] = result['image_paths']
        else:
            max_workers = min(4, os.cpu_count())
            #NOTE: using the multithreading will make it so that the timesteps are not ordered.
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = [executor.submit(load_timestep, t) for t in range(self.parser.num_timesteps)]
                
                # Collect results as they complete
                for future in tqdm(as_completed(futures), total=self.parser.num_timesteps, desc="loading all images"):
                    result = future.result()
                    t = result['t']
                    
                    # Now safely update the instance variables (no threading issues here)
                    self.timestep_images[t] = result['images']
                    self.timestep_masks[t] = result['masks']
                    self.timestep_poses[t] = result['poses']
                    self.timestep_intrinsics[t] = result['intrinsics']
                    self.timestep_image_paths[t] = result['image_paths']
            
        import gc
        gc.collect()
        
        # Validation
        for t in range(self.parser.num_timesteps):
            assert self.timestep_image_paths[t] == sorted(self.timestep_image_paths[t])
        
        print(f"Loaded {self.parser.num_timesteps} timesteps")
        print("")
        print("Printing the first timesteps...")
        for t in range(self.parser.num_timesteps):
            print(f"  Timestep {t}: {len(self.timestep_images[t])} cameras")
    
    
    def get_shared_data(self):
        """
        Return the loaded data in a format that can be shared with another dataset.
        This allows loading once and using for both train and test.
        """
        return {
            'images': self.timestep_images,
            'poses': self.timestep_poses,
            'intrinsics': self.timestep_intrinsics,
            'masks': self.timestep_masks,
            'image_paths': self.timestep_image_paths
        }
    

class Dynamic_Dataset(Dataset):
    """
    Dynamic dataset for captured data
    """
    def __init__(
        self,
        parser,  # Should be DynamicParser
        shared_data,
        split: str = "train",
        is_reverse=False,
        downsample_factor=1, 
        prepend_zero=True,
        include_zero=False, 
        downsample_eval=True,
        cam_batch_size=-1,
        time_normalize_factor=1,
        return_mask=False,
        first_mesh_path=None
    ):
        self.parser = parser
        self.split = split
        self.prepend_zero = prepend_zero
        self.is_reverse = is_reverse
        self.downsample_eval = downsample_eval
        self.cam_batch_size = cam_batch_size
        self.include_zero = include_zero
        self.time_normalize_factor = time_normalize_factor
        self.return_mask = return_mask
        self.first_mesh_path =first_mesh_path #doesn't do anything but ensure compatibility

        if shared_data is not None:
            print("Using shared data (no re-loading)")
            self.timestep_images = shared_data['images']
            self.timestep_poses = shared_data['poses']
            self.timestep_intrinsics = shared_data['intrinsics']
            self.timestep_masks = shared_data['masks']
            self.timestep_image_paths = shared_data['image_paths']
            self.is_shared = True
        else:
            exit("shared_data cannot be null")
        
        self._setup_split() #this part creates a filtered view
        #TODO: for interpolation, we will tweak some stuff here prob
        all_integer_times = np.arange(self.parser.num_timesteps)
        self.times = list(all_integer_times/(self.parser.num_timesteps-1))
    
    def _setup_split(self):
        """
        Setup train/test split based on timesteps.
        When using shared data, we create VIEWS into the shared data rather than copies.
        """
        self.available_timesteps = list(range(self.parser.num_timesteps))
        self.camera_filter = {} #time -> [cam_id]
        
        # Create filtered views based on split
        for timestep in self.available_timesteps:
            if len(self.timestep_images[timestep]) > 0:
                all_cameras = list(self.timestep_images[timestep].keys())
                test_cameras = all_cameras[::self.parser.test_every]  # indices 0, N, 2N, ...
                train_cameras = [cam for cam in all_cameras if cam not in test_cameras]
                
                if self.split == "train":
                    self.camera_filter[timestep] = train_cameras
                elif self.split == "test":
                    self.camera_filter[timestep] = test_cameras
        
        # Print statistics
        print(f"{self.split.capitalize()} split: Using {len(self.camera_filter[0])} cameras (filtered view)")
        print(f"{self.split} set is using image names {self.camera_filter[0]}") #all timesteps share the same cameras
        self.static_indices = self.camera_filter[0] #static_indices are just all camera_ids use in timestep 0
        print("splitting complete")


    def num_timesteps(self) -> int:
        """Return the number of timesteps"""
        return self.parser.num_timesteps
    
    def __len__(self):
        """Return the number of available timesteps"""
        return self.num_timesteps()
    
    def __getitem__(self, timestep: int) -> Dict[str, Any]:
        """
        Retrieve data for a single timestep. 
        NOTE: For single timestep, we output a dictionary, incompatible with raster_params
        NOTE: here we don't take into account the cam batch size and just output all cameras.
        """
        if timestep not in self.timestep_images:
            raise IndexError(f"Timestep {timestep} not found. Available: {list(self.timestep_images.keys())}")
        
        data = {
            "K": {},
            "camtoworld": {},
            "image": {},
            "image_id": {},
            "mask": {}
        }
        
        #For the intrinsics, just set the key to be 1
        data["K"][1] = torch.from_numpy(self.timestep_intrinsics[timestep][1]).float() 

        # Get all cameras available at this timestep
        for i, camera_id in enumerate(self.camera_filter[timestep]):
            data["camtoworld"][camera_id] = torch.from_numpy(self.timestep_poses[timestep][camera_id]).float()
            data["image"][camera_id] = torch.from_numpy(self.timestep_images[timestep][camera_id]).float()
            data["K"][camera_id] = torch.from_numpy(self.timestep_intrinsics[timestep][1]) #intrinsics is always just indexed at 1
            data["image_id"][camera_id] = i #(always just use numbers 0 -> len(images)- 1 as image ids)
         
        return data
    
    def __getitems__(self, timesteps: Union[List[int], np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ 
        Retrieve data for multiple timesteps as a batch.
        Just returns all cameras available for the queried timesteps.
        The only useful stuff for us is 
        1) self.timestep_images (timestep -> images)
        2) self.timestep_poses (timestep -> poses)

        Returns:
            Tuple:
            - `c2ws`: (N, 4, 4) tensor of camera-to-world matrices for all views.
            - `gt_images`: (N, T, H, W, C) tensor of images for all views at selected timesteps.
            - `inp_t`: (T,) tensor of normalized time values
        NOTE: This assumes all timesteps have the same cameras available
        """
        if isinstance(timesteps, np.ndarray):
            timesteps = timesteps.tolist()
        
        
        # Handle include_zero option
        timesteps_to_use = timesteps.copy()
        if self.include_zero and 0 not in timesteps_to_use:
            timesteps_to_use.insert(0, 0)
        
        #NOTE: right now, just to make it simple, we make it only work on temp_batch_size == 1
        selected_timestep = timesteps[0] #single timestep
        available_cameras = self.camera_filter[selected_timestep] #includes all cameras
        if self.cam_batch_size == -1: #select all camera ids
            selected_cameras = available_cameras 
        else:
            #NOTE: this is not sorted
            selected_cameras = random.sample(available_cameras, min(self.cam_batch_size, len(available_cameras)))
        
        c2w_list = []
        img_list = []
        masks_list = []
        
        
        # Collect data for each selected camera
        for camera_id in selected_cameras:
            c2w_list.append(torch.from_numpy(self.timestep_poses[selected_timestep][camera_id]).float())
            img_list.append(torch.from_numpy(self.timestep_images[selected_timestep][camera_id]).float())
            masks_list.append(torch.from_numpy(self.timestep_masks[selected_timestep][camera_id]).float())
            
        
        c2ws = torch.stack(c2w_list)  # (N, 4, 4)
        gt_images = torch.stack(img_list)[:, None]  # (N, T, H, W, C)
        gt_masks = torch.stack(masks_list)[:, None]
        
        #Normalize the timesteps in [0,...,self.time_normalize_factor]
        cont_t = torch.tensor(timesteps, dtype=torch.float32) / (self.num_timesteps() - 1)
        cont_t *= self.time_normalize_factor
        
        # Handle prepend_zero option
        if not self.include_zero:  # if we included zero earlier we don't include it now
            if self.prepend_zero:
                inp_t = torch.cat((torch.tensor([0.0]), cont_t), dim=0)
            else:
                inp_t = cont_t
        else:
            inp_t = cont_t

        if self.return_mask:
            return c2ws, gt_images, inp_t, gt_masks
        else:
            return c2ws, gt_images, inp_t
    
    def get_available_cameras_at_timestep(self, timestep: int) -> List:
        """Return list of available camera IDs at a specific timestep"""
        if timestep in self.timestep_images:
            return self.camera_filter[timestep]
        return []
    
    def get_timestep_range(self) -> tuple[int, int]:
        """Return the range of available timesteps"""
        return (0, self.num_timesteps() - 1)
    
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
        gt_masks_batch = batch[3]

        #Sort inp_t without the first element (which is 0)
        t0, inp_t_to_sort = inp_t[0], inp_t[1:]
        sorted_inp_t, indices = inp_t_to_sort.sort()

        #Sort the image batch and the masks
        gt_images_batch = gt_images_batch[:, indices, ...]
        gt_masks_batch = gt_masks_batch[:, indices, ...]

        #Recreate the new inp_t
        new_inp_t = torch.cat((t0.unsqueeze(0), sorted_inp_t), dim=0)
        int_t = map_cont_to_int(new_inp_t,self.num_timesteps())

        return c2ws_batch, gt_images_batch, new_inp_t, int_t, gt_masks_batch



class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            #this part stores the parameters for undistortion
            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        if factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(
                colmap_image_dir, image_dir + "_png", factor=factor
            )
            image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)




class Static_Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.first_mesh_path = ""

        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        image = image/255.0
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data

class SingleTimeDataset(Dataset):
    def __init__(self, dataset, timestep):
        """
        Custom single time dataset for specific timestep training.
        len() would return the number of cameras (images) for that timestep.
        Indexing into this dataset will return one of the cameras for that timestep.
        """
        self.dataset = dataset
        self.timestep = timestep

    def __len__(self):
        return len(self.dataset.static_indices)


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve data for a single timestep. 
        Here idx stands for a single image.
        NOTE: For single timestep, we output a dictionary, incompatible with raster_params
        NOTE: for statictimedataset, self.timestep = 0
        """
        idx = self.dataset.static_indices[idx]
        data = dict(
            K = torch.from_numpy(self.dataset.timestep_intrinsics[self.timestep][1]).float(),
            camtoworld = torch.from_numpy(self.dataset.timestep_poses[self.timestep][idx]).float(),
            image = torch.from_numpy(self.dataset.timestep_images[self.timestep][idx]).float(),
            image_id = idx,
            mask = torch.from_numpy(self.dataset.timestep_masks[self.timestep][idx]).float()
        ) 
        
        return data



if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Static_Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
