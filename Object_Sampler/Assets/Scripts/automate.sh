#!/bin/bash

# declare the target directories
dest_dir=$1
image_dir="images"

# copy over the transforms and image files
cp ./transforms.json $dest_dir
cp ../screenshots/*.png $dest_dir$image_dir
