import json
import shutil
from pathlib import Path
from collections import defaultdict

def subsample_dataset(
    original_data_dir: str,
    output_data_dir: str,
    every_n: int = 2,
    always_include_last: bool = True
):
    """
    Create a subsampled version of the dataset for training only.
    Copies everything as-is except train/ and transforms_train.json which are subsampled.
    
    Args:
        original_data_dir: Path to original dataset
        output_data_dir: Path to output subsampled dataset
        every_n: Keep every nth frame
        always_include_last: Always include the last frame for each camera
    """
    original_dir = Path(original_data_dir)
    output_dir = Path(output_data_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Subsampling dataset: {original_dir.name}")
    print(f"Output: {output_dir}")
    print(f"Subsampling rate: every_{every_n}")
    print("=" * 60)
    
    # Step 1: Copy everything except train/ and transforms_train.json
    print("\n[1/3] Copying non-training files as-is...")
    for item in original_dir.iterdir():
        if item.name not in ["train", "transforms_train.json"]:
            if item.is_dir():
                print(f"  Copying directory: {item.name}/")
                shutil.copytree(item, output_dir / item.name, dirs_exist_ok=True)
            else:
                print(f"  Copying file: {item.name}")
                shutil.copy2(item, output_dir / item.name)
    
    # Step 2: Subsample transforms_train.json
    print("\n[2/3] Subsampling transforms_train.json...")
    with open(original_dir / "transforms_train.json", 'r') as f:
        train_transforms = json.load(f)
    
    # Group frames by camera
    frames_by_camera = defaultdict(list)
    for frame in train_transforms["frames"]:
        camera_id = frame["file_path"].split("/")[-2]  # e.g., "r_0"
        frames_by_camera[camera_id].append(frame)
    
    # Subsample frames for each camera
    new_frames = []
    frames_to_copy = []  # Track which image files to copy
    
    for camera_id, frames in sorted(frames_by_camera.items()):
        print(f"\n  Camera {camera_id}:")
        print(f"    Original frames: {len(frames)}")
        
        # Determine which indices to keep
        indices_to_keep = set(range(0, len(frames), every_n))
        if always_include_last:
            indices_to_keep.add(len(frames) - 1)
        
        print(f"    Subsampled frames: {len(indices_to_keep)}")
        print(f"    Keeping indices: {sorted(indices_to_keep)}")
        
        # Keep frames at selected indices
        for idx in sorted(indices_to_keep):
            frame = frames[idx]
            new_frames.append(frame)
            frames_to_copy.append(frame["file_path"])
    
    # Create new transforms_train.json with subsampled frames
    new_train_transforms = {
        "camera_angle_x": train_transforms["camera_angle_x"],
        "frames": new_frames
    }
    
    with open(output_dir / "transforms_train.json", 'w') as f:
        json.dump(new_train_transforms, f, indent=4)
    
    print(f"\n  ✓ Created transforms_train.json")
    print(f"    Total original frames: {len(train_transforms['frames'])}")
    print(f"    Total subsampled frames: {len(new_frames)}")
    
    # Step 3: Copy subsampled training images
    print("\n[3/3] Copying subsampled training images...")
    (output_dir / "train").mkdir(exist_ok=True)
    
    for file_path in frames_to_copy:
        # Get camera directory
        camera_id = file_path.split("/")[-2]
        (output_dir / "train" / camera_id).mkdir(exist_ok=True)
        
        # Copy image
        original_image_path = original_dir / file_path.replace("./", "")
        output_image_path = output_dir / file_path.replace("./", "")
        shutil.copy2(original_image_path, output_image_path)
    
    print(f"  ✓ Copied {len(frames_to_copy)} training images")
    
    print("\n" + "=" * 60)
    print("✓ Subsampled dataset created successfully!")
    print(f"  Location: {output_dir}")
    print("=" * 60)

# Usage
if __name__ == "__main__":
    for scene in ["lily"]:
        for factor in [2]:
            subsample_dataset(
                original_data_dir=f"../data/dynamic/blender/360/multi-view/31_views/{scene}_transparent_final_small_vase_70_timesteps",
                output_data_dir=f"../data/dynamic/blender/360/multi-view/31_views/{scene}_transparent_final_small_vase_70_timesteps_subsample_{factor}",
                every_n=factor,
                always_include_last=True
            )