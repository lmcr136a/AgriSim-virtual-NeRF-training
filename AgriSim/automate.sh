#!/bin/bash

### COPY OVER FILES ### 
# Declaring target directories
dest_dir=$1 
image_dir="images"
screenshot_dir=$2
screenshot_transforms=$3

# Compile the transforms.json file 
python Object_Sampler/Assets/Scripts/get_transforms.py

# Copy transforms information to Instant-NGP directory
cp ./Object_Sampler/Assets/screenshots/transforms.json $dest_dir

# If the images directory doesn't exist, create it
dir_name="images"
if [ ! -d "$dest_dir$dir_name" ]; then
  mkdir "$dest_dir$dir_name"
fi
# Copy over images to Instant-NGP directory
cp ./Object_Sampler/Assets/screenshots/*.png $dest_dir$image_dir

# once you copy over the files, run instant ngo to output a mesh file and screenshots
save_mesh=$4
echo "save_mesh argument is: $save_mesh"
if [ "$save_mesh" = true ]; then
    python ../models/instant-ngp/scripts/run.py --scene $dest_dir --network base.json --n_steps 5000 --save_mesh $screenshot_dir"mesh.obj" --screenshot_dir $screenshot_dir --screenshot_transforms $screenshot_transforms
else
    python ../models/instant-ngp/scripts/run.py --scene $dest_dir --network base.json --n_steps 5000 --screenshot_dir $screenshot_dir --screenshot_transforms $screenshot_transforms
fi

echo "Resulting mesh and screenshots saved to $screenshot_dir"