import json
import os
import shutil
import numpy as np
from collections import defaultdict


def extract_subset_json(base_path, target_path, start, end):
    """
    Given a path to a base directory containing train, test, transforms_train.json, transforms_test.json
    Move all the files in base path.
    Get only the frames from [start, end] 
    1. Create train, test, transforms files/folders to target_path and meshes folder
    2. For the train/test folders, copy the files from start to end to target_path
    3. For poses, copy the poses from start to end.
    """
    os.makedirs(os.path.join(target_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(target_path, "test"), exist_ok=True)
    os.makedirs(os.path.join(target_path, "meshes/relevant_meshes"), exist_ok=True)

    #Iterate through all training files 
    training_cameras = sorted(os.listdir(os.path.join(base_path, "train"))) #contains all training cameras
    copied_train_files = []
    for train_camera in training_cameras:
        full_camera_path = os.path.join(target_path, "train", train_camera)
        os.makedirs(full_camera_path, exist_ok=True)
        full_src_camera_path = os.path.join(base_path, "train", train_camera)
        training_images = sorted(os.listdir(full_src_camera_path))
        for i in range(start, end+1):
            copied_train_files.append(training_images[i])
            src_path = os.path.join(base_path, "train", train_camera, training_images[i]) 
            tgt_path = os.path.join(target_path, "train", train_camera, training_images[i])
            shutil.copy(src_path, tgt_path) 
    
    print("all training images copied") 
    #Iterate through all testing images
    test_cameras = sorted(os.listdir(os.path.join(base_path, "test"))) #contains all test cameras
    copied_train_files = []
    for test_camera in test_cameras:
        full_camera_path = os.path.join(target_path, "test", test_camera)
        os.makedirs(full_camera_path, exist_ok=True)
        full_src_camera_path = os.path.join(base_path, "test", test_camera)
        test_images = sorted(os.listdir(full_src_camera_path))
        for i in range(start, end+1):
            copied_train_files.append(test_images[i])
            src_path = os.path.join(base_path, "test", test_camera, test_images[i]) 
            tgt_path = os.path.join(target_path, "test", test_camera, test_images[i])
            shutil.copy(src_path, tgt_path) 
   
    print("all test images copied") 
    #Iterate through all mesh files
    cached_mesh_list = np.load(os.path.join(base_path, "unique_mesh_indices.npy")).astype(np.uint8)
    for i in range(start, end+1):
        mesh_file = os.path.join(base_path, "meshes", "relevant_meshes", f"mesh{cached_mesh_list[i]:04d}.obj")
        shutil.copy(mesh_file, os.path.join(target_path, "meshes", "relevant_meshes", f"mesh{cached_mesh_list[i]:04d}.obj"))        
    
    
    print("all meshes copied") 
    #Copying train stuff
    train_json_path = f"{base_path}/transforms_train.json"    
    new_json_train_path = f"{target_path}/transforms_train.json"
    with open(train_json_path) as f:
        train_json = json.load(f)

    new_json_train = {
        "camera_angle_x": train_json["camera_angle_x"],
        "frames": []
    }
    train_frames = train_json["frames"]
    
    train_cam_dict = defaultdict(list) #need to group frames by cameras
    for frame in train_frames:
        file_path = frame["file_path"]
        # Extract camera ID from file_path (e.g., "r_1" from "./train/r_1/00000.png")
        camera_id = file_path.split('/')[2]
        train_cam_dict[camera_id].append(frame)

    for k,v in train_cam_dict.items():
        for i in range(start, end+1):
            new_json_train["frames"].append(train_cam_dict[k][i])
         
    with open(new_json_train_path, 'w') as f:
        json.dump(new_json_train, f, indent=4)


    print("all train poses copied") 
    #Copying test stuff
    test_json_path = f"{base_path}/transforms_test.json"    
    new_json_test_path = f"{target_path}/transforms_test.json"
    with open(test_json_path) as f:
        test_json = json.load(f)

    new_json_test = {
        "camera_angle_x": test_json["camera_angle_x"],
        "frames": []
    }
    test_frames = test_json["frames"]
    
    test_cam_dict = defaultdict(list) #need to group frames by cameras
    for frame in test_frames:
        file_path = frame["file_path"]
        # Extract camera ID from file_path (e.g., "r_1" from "./test/r_1/00000.png")
        camera_id = file_path.split('/')[2]
        test_cam_dict[camera_id].append(frame)

    for k,v in test_cam_dict.items():
        for i in range(start, end+1):
            new_json_test["frames"].append(test_cam_dict[k][i])
         
    with open(new_json_test_path, 'w') as f:
        json.dump(new_json_test, f, indent=4)
    print("all test poses copied")
    
if __name__ == "__main__":
    # start = 28
    # end = 34
    # for i in range(34): #ends at 33-34
    #     start = i 
    #     end = i+1
    #     scene = "rose_transparent"
    #     print(f"extracting subset renderings for {scene} for {start} - {end}")
    #     base_path = f"/scratch/ondemand28/weihanluo/neural_ode_splatting/data/dynamic/blender/360/multi-view/30_views/{scene}"
    #     target_path = f"/scratch/ondemand28/weihanluo/neural_ode_splatting/data/dynamic/blender/360/multi-view/30_views/{scene}_subset_{start}_{end}"
    #     os.makedirs(target_path, exist_ok=True)
    #     extract_subset_json(base_path, target_path, start, end)
    start = 0
    end = 27
    scene = "rose_transparent"
    print(f"extracting subset renderings for {scene} for {start} - {end}")
    base_path = f"/scratch/ondemand28/weihanluo/neural_ode_splatting/data/dynamic/blender/360/multi-view/30_views/{scene}"
    target_path = f"/scratch/ondemand28/weihanluo/neural_ode_splatting/data/dynamic/blender/360/multi-view/30_views/{scene}_subset_{start}_{end}"
    os.makedirs(target_path, exist_ok=True)
    extract_subset_json(base_path, target_path, start, end)
