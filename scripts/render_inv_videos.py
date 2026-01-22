from glob import glob
import os
import imageio.v2 as imageio
from argparse import ArgumentParser

def visualize_training_imgs(input_path, split):
    """
    Given a path 
    -train
    -test
    Where they contain individual camera indices, generate video of all frames in a given camera folder
    """
    full_path = os.path.join(input_path, split)
    all_files = [f for f in os.listdir(full_path) if f.startswith("r_")] #camera indices start with r_
    sorted_files = sorted(all_files, key=lambda x: int(x.split('_')[1]))
    for i, folder in enumerate(sorted_files):
        all_imgs = []
        camera_i_path = os.path.join(full_path, folder)
        imgs_in_camera_folder = [f for f in os.listdir(camera_i_path) if f.endswith(".png")]
        for i, img_frames in enumerate(imgs_in_camera_folder):
            full_image_path = os.path.join(camera_i_path, f"{i:05d}.png")
            img = imageio.imread(full_image_path) #reads as uint
            all_imgs.append(img)
            print(f"img {i} has been saved")

        duration = 3
        print(len(all_imgs))
        imageio.mimwrite(
            f"{camera_i_path}/imgs.mp4",
            all_imgs,
            fps = len(all_imgs)/duration
        )

    print("done")
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--base_path", "-b", default="")
    args = parser.parse_args()
    base_path = args.base_path
    splits = ["train", "test"]
    for split in splits:
        input_path = base_path
        try:
            visualize_training_imgs(input_path=input_path, split=split)
        except: #in case training or testing has been skipped
            continue