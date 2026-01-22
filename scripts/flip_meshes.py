import os
import subprocess
import numpy as np
import shutil
import re

def flip_meshes(base_dir):
    """
    Flip meshes in the relevant_meshes directory.
    Handle mesh file names like 'mesh0050.obj', 'mesh0052.obj', etc.
    Maps first mesh object to last, and generally i → total_length - i - 1
    """
    paths = ["plant1_transparent", "plant2_transparent", "plant3_transparent", 
             "plant4_transparent", "plant5_transparent", "rose_transparent"]
    
    mesh_pattern = re.compile(r'mesh(\d+)\.obj')
    
    for path in paths:
        print(f"Processing {path}")
        full_path = os.path.join(base_dir, path)
        
        for subfolder in [f for f in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, f))]:
            subfolder_full_path = os.path.join(full_path, subfolder)
            
            if subfolder == "meshes":
                relevant_meshes_dir = os.path.join(subfolder_full_path, "relevant_meshes")
                
                # Check if the directory exists
                if not os.path.exists(relevant_meshes_dir):
                    print(f"Directory not found: {relevant_meshes_dir}")
                    continue
                
                # Get all mesh files and extract their numbers using regex
                mesh_files = []
                mesh_numbers = []
                
                for filename in os.listdir(relevant_meshes_dir):
                    match = mesh_pattern.match(filename)
                    if match and os.path.isfile(os.path.join(relevant_meshes_dir, filename)):
                        number = int(match.group(1))
                        mesh_files.append(filename)
                        mesh_numbers.append(number)
                
                # Sort the files by their extracted number
                sorted_pairs = sorted(zip(mesh_files, mesh_numbers), key=lambda x: x[1])
                sorted_files = [pair[0] for pair in sorted_pairs]
                
                total_files = len(sorted_files)
                if total_files == 0:
                    print(f"No mesh files found in {relevant_meshes_dir}")
                    continue
                
                print(f"Found {total_files} mesh files in {relevant_meshes_dir}")
                
                # Create mapping dictionary: original file → new file
                mapping = {}
                for i, filename in enumerate(sorted_files):
                    new_index = total_files - i - 1
                    old_file = sorted_files[i]
                    new_file = sorted_files[new_index]
                    mapping[old_file] = f"temp_{new_file}"
                
                # Create temporary directory for the flipping process
                temp_dir = os.path.join(relevant_meshes_dir, "temp_flip")
                os.makedirs(temp_dir, exist_ok=True)
                
                # First copy to temporary location using our mapping
                for old_file, temp_file in mapping.items():
                    shutil.copy2(
                        os.path.join(relevant_meshes_dir, old_file),
                        os.path.join(temp_dir, temp_file)
                    )
                    print(f"Copying {old_file} → {temp_file[5:]}")
                
                # Then move from temp to final destination
                for temp_file in os.listdir(temp_dir):
                    if temp_file.startswith("temp_"):
                        final_name = temp_file[5:]  # Remove "temp_" prefix
                        shutil.move(
                            os.path.join(temp_dir, temp_file),
                            os.path.join(relevant_meshes_dir, final_name)
                        )
                
                # Remove temporary directory
                shutil.rmtree(temp_dir)
                print(f"Successfully flipped {total_files} mesh files in {relevant_meshes_dir}")

if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (root of your project)
    project_root = os.path.dirname(script_dir)
    base_dir = os.path.join(project_root, "./data/dynamic/blender/360/multi-view/30_views")
    flip_meshes(base_dir)