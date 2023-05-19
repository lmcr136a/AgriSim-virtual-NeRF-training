## Unity Sampling Pipeline 

### Description

This repo provides a method to generate a NeRF of a 3D object using the Unity Game Engine and Instant-NGP. The pipeline begins with a simple Unity environment, where we have a 3D mesh object simulated with the Unity framework. It lies on a solid, flat surface, and has a camera that captures images of the objects at various view angles. 

Once we have these images, our Unity script will write all of the information needed to generate a NeRF using Instant NGP to a text file. This includes the camera's physical properties, the actual images of the objects, as well as the positions and quaternions associated with each image. Once we have written to the text file, our Python script compiles a transforms.json file that formats all of the information that we collected. This essentially allows us to bypass the use of COLMAP when creating a NeRF.

### Instructions

After cloning this repo, please install the Unity game engine and ensure that it works properly. Then, install the following dependencies.

```
python
scipy
```

After opening up the Unity environment, simply set the ```max_screenshots``` parameter in the CameraRotator script by navigating to the Main Camera's inspector GUI. Click play, and the image sampling phase will begin.

Once the textfiles and screenshots have all been saved, navigate to the Scripts directory and run the following commands.
```sh
Scripts$ python get_transforms.py
Scripts$ ./automate.sh <absolute_file_path_to_instant_ngp_image_folder>
```

The ```automate.sh``` script will copy over the image files and transforms.json file to the instant ngp directory. Finally, build the Instant-NGP NeRF using the instructions given [Instant-NGP Instructions](https://github.com/NVlabs/instant-ngp)
