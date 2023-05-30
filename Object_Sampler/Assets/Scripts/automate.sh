#!/bin/bash

### RUN UNITY SCRIPT ###


### COPY OVER FILES ### 
# declare the target directories
dest_dir=$1 # */instant-ngp/data/nerf/sim_flower
image_dir="images"

# run the python script to get the transforms
python get_transforms.py

# copy over the transforms and image files
cp ./transforms.json $dest_dir
cp ../screenshots/*.png $dest_dir$image_dir

echo "Images and transforms copied over!"

### RUN INSTANT NGP ###
# once you copy over the files, run instant ngo to output a mesh file
python $dest_dir/../../../scripts/run.py --mode nerf --scene $dest_dir --network base.json --n_steps 3500 --save_mesh $dest_dir/../../../scripts/test_mesh/mesh.obj

echo "Resulting mesh saved to $dest_dir/../../../scripts/test_mesh/mesh.obj"