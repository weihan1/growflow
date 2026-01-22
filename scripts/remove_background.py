from rembg import remove
from PIL import Image
import numpy as np
#NOTE: need to activate different conda env to use code below
# Open the image
input_path = "/scratch/ondemand28/weihanluo/neural_ode_splatting/results/pi_plant5/baseline_longer/full_eval/test/r_0/00000.png"  # Change to your image path
output_path = "output_no_bg.png"  # Output with transparency

# Load image
input_image = Image.open(input_path)

# Remove background
output_image = remove(input_image)

# Save result
output_image.save(output_path)

print(f"Background removed! Saved to {output_path}")
