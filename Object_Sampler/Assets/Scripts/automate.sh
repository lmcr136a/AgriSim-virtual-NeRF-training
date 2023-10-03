#!/bin/bash

### COPY OVER FILES ### 
# Declaring target directories
dest_dir=$1 
image_dir="images"

# Compile the transforms.json file 
python get_transforms.py

# Copy transforms information to Instant-NGP directory
cp ./transforms.json $dest_dir

# If the images directory doesn't exist, create it
dir_name="images"
if [ ! -d "$dest_dir$dir_name" ]; then
  mkdir "$dest_dir$dir_name"
  echo "Directory '$dest_dir$dir_name' created."
else
  echo "Directory '$dest_dir$dir_name' already exists."
fi
# Copy over images to Instant-NGP directory
cp ../screenshots/*.png $dest_dir$image_dir

### RUN INSTANT NGP ###
# once you copy over the files, run instant ngo to output a mesh file
python $dest_dir/../../../scripts/run.py --mode nerf --scene $dest_dir --network base.json --n_steps 3500 --save_mesh $dest_dir"mesh.obj"

echo "Resulting mesh saved to $dest_dir/../../../scripts/test_mesh/mesh.obj"