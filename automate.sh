#!/bin/bash

### COPY OVER FILES ### 
# Declaring target directories
dest_dir=$1 
image_dir="images"

# Compile the transforms.json file 
python Object_Sampler/Assets/Scripts/get_transforms.py

# Copy transforms information to Instant-NGP directory
cp ./Object_Sampler/Assets/Scripts/transforms.json $dest_dir

# If the images directory doesn't exist, create it
dir_name="images"
if [ ! -d "$dest_dir$dir_name" ]; then
  mkdir "$dest_dir$dir_name"
fi
# Copy over images to Instant-NGP directory
cp ./Object_Sampler/Assets/screenshots/*.png $dest_dir$image_dir

### RUN INSTANT NGP ###
# once you copy over the files, run instant ngo to output a mesh file
python $dest_dir/../../../scripts/run.py --mode nerf --scene $dest_dir --network base.json --n_steps 3500 --save_mesh $dest_dir"mesh.obj"

echo "Resulting mesh saved to $dest_dir/mesh.obj"