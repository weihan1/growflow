import os
import subprocess
import numpy as np
import shutil



def flip_images(base_dir):
    """
    Flip images inside the train and test folders. 
    E.g. 00000.png <-> last_image
    For instance if there was 35 images, you move 00000.png <-> 00034.png
    Then for the meshes folder, first open the unique_mesh_indices.npy file, and only keep 
    the mesh indices that you need. Then also flip.
    """
    # paths = ["plant2_transparent", "plant3_transparent", 
    #          "plant4_transparent", "plant5_transparent", "rose_transparent"]
    # paths = ["plant1_transparent"] 
    # paths = ["rose_subset_transparent_31_35", "rose_subset_transparent_0_6", "rose_subset_transparent_0_10"]

    for path in paths:
        print(f"processing {path}")
        full_path = os.path.join(base_dir, path)
        try:
            mesh_indices_file = os.path.join(full_path, "unique_mesh_indices.npy")
            if os.path.exists(mesh_indices_file):
                mesh_indices_folder = np.load(mesh_indices_file).astype(np.uint8)
            else:
                print(f"Warning: unique_mesh_indices.npy not found in {full_path}")
                continue
                
            for subfolder in [f for f in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, f))]:
                subfolder_full_path = os.path.join(full_path, subfolder)
                
                if subfolder in ["train", "test"]:
                    for camera_folder in os.listdir(subfolder_full_path):
                        print(f"processing {camera_folder}")
                        camera_path = os.path.join(subfolder_full_path, camera_folder)
                        if os.path.isdir(camera_path):
                            # Get all PNG files in this directory
                            png_files = sorted([f for f in os.listdir(camera_path) 
                                              if f.endswith(".png") and f[:5].isdigit()])
                            
                            total_files = len(png_files)
                            if total_files == 0:
                                continue
                                
                            # Create temporary directory for the flipping process
                            temp_dir = os.path.join(camera_path, "temp_flip")
                            os.makedirs(temp_dir, exist_ok=True)
                            
                            # First move to temporary location to avoid conflicts
                            for file in png_files:
                                number = int(file[:5])
                                new_number = total_files - number - 1
                                new_file = f"{new_number:05d}.png"
                                shutil.copy2(
                                    os.path.join(camera_path, file),
                                    os.path.join(temp_dir, f"temp_{new_file}")
                                )
                            
                            # Then move from temp to final destination
                            for temp_file in os.listdir(temp_dir):
                                final_name = temp_file[5:]  # Remove "temp_" prefix
                                shutil.move(
                                    os.path.join(temp_dir, temp_file),
                                    os.path.join(camera_path, final_name)
                                )
                            
                            # Remove temporary directory
                            shutil.rmtree(temp_dir)
                            print(f"Flipped {total_files} images in {camera_path}")
                
                elif subfolder == "meshes":
                    relevant_meshes_dir = os.path.join(subfolder_full_path, "relevant_meshes")
                    other_meshes_dir = os.path.join(subfolder_full_path, "other_meshes")
                    
                    # Create directories if they don't exist
                    os.makedirs(relevant_meshes_dir, exist_ok=True)
                    os.makedirs(other_meshes_dir, exist_ok=True)
                    
                    # Write the format information
                    with open(os.path.join(other_meshes_dir, "format.txt"), "w") as f:
                        f.write("These mesh files are NOT reversed!")
                    
                    # Process mesh files
                    mesh_files = [f for f in os.listdir(subfolder_full_path) 
                                 if os.path.isfile(os.path.join(subfolder_full_path, f)) and f.endswith(".obj")]
                    
                    for mesh in mesh_files:
                        mesh_path = os.path.join(subfolder_full_path, mesh)
                        try:
                            mesh_number = int(mesh.split(".")[0][-4:])
                            if mesh_number in mesh_indices_folder:
                                shutil.move(mesh_path, os.path.join(relevant_meshes_dir, mesh))
                            else:
                                shutil.move(mesh_path, os.path.join(other_meshes_dir, mesh))
                        except ValueError:
                            # Handle case where mesh filename doesn't match expected format
                            print(f"Skipping {mesh}: couldn't extract mesh number")
                    print("finished processing meshes")
        
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            continue
    
    print("Image and mesh flipping completed.")



if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (root of your project)
    project_root = os.path.dirname(script_dir)
    base_dir = os.path.join(project_root, "./data/dynamic/blender/360/multi-view/30_views")
    flip_images(base_dir)