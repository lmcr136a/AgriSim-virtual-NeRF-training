#!/bin/bash

# declare the target directories
dest_dir=$1
image_dir="images"

# run the python script to get the transforms
python get_transforms.py

# copy over the transforms and image files
cp ./transforms.json $dest_dir
cp ../screenshots/*.png $dest_dir$image_dir
