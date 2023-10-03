## Unity Sampling Pipeline   

### Description

This repository provides a method to generate a NeRF of a 3D object using the Unity Game Engine and Instant-NGP. The pipeline begins with a simple Unity environment, where we have a 3D mesh object simulated with the Unity framework. It lies on a solid, flat surface, and has a camera that captures images of the objects at various view angles. 

![environment_view](https://github.com/QuantuMope/AgriSim/assets/63471459/71346f53-15ce-4e3d-a09b-7ce224e5db41)

Once we have these images, our Unity script will write all of the information needed to generate a NeRF using Instant NGP to a text file. This includes the camera's physical properties, the actual images of the objects, as well as the positions and quaternions associated with each image. Once we have written to the text files, our Python script compiles a transforms.json file that formats all of the information that we collected. This essentially allows us to bypass the use of COLMAP when creating a NeRF.

Finally, we will call the Instant NGP's Python API ```pyngp``` to generate a mesh object of the resulting NeRF.

### Instructions

Clone this repository and install Unity Game Engine. Then, install the following dependencies.

```
python
scipy
opencv
numpy
```

After opening up the Unity environment, set the ```max_screenshots``` parameter in the CameraRotator script by navigating to the Main Camera's inspector GUI.

___**Make sure to enable physical camera in the Unity GUI.**___ This is to ensure that the camera's intrinsic properties written to downstream modules correspond to a physical camera. 

![readme_unity_help](https://github.com/QuantuMope/AgriSim/assets/63471459/600b1250-f88c-490c-baef-9b1427a77083)


Once the textfiles and screenshots have all been saved, make sure you have created a separate directory for the NeRF object
```
<path-to-instant-ngp>/data/nerf/<OBJECT_NAME> # Make sure this directory exists
```

Then, navigate to the Scripts directory and run the following commands.

```sh
AgriSim/Object_Sampler/Assets/Scripts$ ./automate.sh <absolute_file_path_to_instant_ngp_image_folder>
```
For example:
```sh
AgriSim/Object_Sampler/Assets/Scripts$ ./automate.sh /home/brian/Desktop/instant-ngp/data/nerf/sim_flower/
```

The ```automate.sh``` script will perform two operations:
1. Generate the transforms.json file by running the python script ```get_transforms.py```.
2. Copy over the image files and transforms.json field to the instant-ngp directory.
3. Call the Instant-NGP's Python API to train and build a NeRF based on the given training images.

Once you have the resulting mesh, you can view the resulting OBJ file using any rendering software like Meshlab.

### Challenges

A notable challenge that comes with constructing NeRFs from simulated 3D objects is the incompatibility of the coordinate transform structure. Unity and Instant-NGP work in different coordinate systems; Unity uses left-handed, Y-up, whereas Instant-NGP uses right-handed, Y-down (see this [thread](https://github.com/NVlabs/instant-ngp/discussions/153) for more info).
Therefore, the transform matrix obtained with Unity's camera will not be compatible with Instant-NGP's transform matrix convention, and the resulting NeRF will look inaccurate. Here are the steps taken to resolve this issue.
1. Within Unity, save the position and quaternion values for each frame.
2. When compiling the ```transforms.json``` file within ```get_transforms.py```, we will swap the y-axis and z-axis columns of each quaternion and flip the switch of each of four columns, i.e. ```[x y z w ]``` becomes ```[-x -z -y -w]```.
3. Initialize a 3x3 Rotation Matrix represented from the original quaternion data using ```scipy.spatial.transform.Rotation.from_quat```.
4. Now for each frame, we combine the quaternion and position data to construct the 4x4 matrix. The resulting 4x4 matrix should look like 
$$
\left(\begin{array}{cc} 
Q[0,0] & Q[0,1] & Q[0,2] & P[0]\\
Q[1,0] & Q[1,1] & Q[1,2] & P[1]\\
Q[2,0] & Q[2,1] & Q[2,2] & P[2]\\
0 & 0 & 0 & 1
\end{array}\right)
$$
Where $Q$ represents the quaternion data and $P$ represents position data.


