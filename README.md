## Unity Sampling Pipeline   

### Description

This repository provides a Python library that generates a NeRF of a ShapeNet 3D object using the Unity Game Engine and Instant-NGP. The pipeline begins with a simple Unity environment, where we have a 3D mesh object simulated with the Unity framework. It lies on a solid, flat surface, and has a camera that captures images of the objects at various view angles. 

The Python Sampler can download a user-inputted object from the ShapeNet dataset. It then connects to the Unity Game Scene using [peaceful-pie](https://github.com/hughperkins/peaceful-pie) and places that object into the scene. Then, the user can call the Python Sampler to sample from their desired viewpoints. These camera poses are in the form of $(x, y, z, θ, φ)$, where $(x, y, z)$ represent the camera's position and $(θ, φ)$ represent the camera's rotation.

The Sampler allows the user to understand the effects that the additional camera poses have on the overall quality of the NeRF.

![image](https://github.com/QuantuMope/AgriSim/assets/63471459/cd5a9e93-c817-4d1c-a2c0-6b4044499c5f)

Once we have these images, our Unity script will write all of the information needed to generate a NeRF using Instant NGP to a text file. This includes the camera's physical properties, the actual images of the objects, as well as the positions and quaternions associated with each image. Once we have written to the text files, our Python script compiles a transforms.json file that formats all of the information that we collected. This essentially allows us to bypass the use of COLMAP, which takes much longer than our groundtruth method of computing transforms.

Finally, we will call the Instant NGP's Python API ```pyngp``` to generate a mesh object of the resulting NeRF.

### Instructions

Clone this repository and install Unity Game Engine. Then, install the following dependencies.

```
python
scipy
opencv
numpy
datasets
peaceful-pie
```

Please follow the instructions on peaceful-pie's [repository](https://github.com/hughperkins/peaceful-pie) to ensure that it works correctly on both the Python and Unity side.

___**Make sure to enable physical camera in the Unity GUI and do not edit any of the physical parameters.**___ This is to ensure that the camera's intrinsic properties written to downstream modules correspond to a physical camera. 

![readme_unity_help](https://github.com/QuantuMope/AgriSim/assets/63471459/600b1250-f88c-490c-baef-9b1427a77083)


Once the text files and screenshots have all been saved, make sure you have created a separate directory for the NeRF object
```
<path-to-instant-ngp>/data/nerf/<OBJECT_NAME> # Make sure this directory exists
```

Additionally, make sure that you are in the AgriSim directory when running the Sampler. An example file utilizing the Sampler has been provided

```sh
AgriSim$ python test_sampler.py
```

#### Viewing Results
Once Instant-NGP is finished with training, it will save a mesh file and screenshots of the resulting NeRF. The user can specify the destination paths of these files before running the Sampler.

In order to specify where to take the resulting screenshots from, the user can input a list of camera poses into the ```screenshot_transforms``` parameter when running the Python Sampler.

Once you have the resulting mesh, you can view the resulting OBJ file using any rendering software like Meshlab.


### Challenges

A notable challenge that comes with constructing NeRFs from simulated 3D objects is the incompatibility of the coordinate transform structure. Unity and Instant-NGP work in different coordinate systems; Unity uses left-handed, Y-up, whereas Instant-NGP uses right-handed, Y-down (see this [thread](https://github.com/NVlabs/instant-ngp/discussions/153) for more info).
Therefore, the transform matrix obtained with Unity's camera will not be compatible with Instant-NGP's transform matrix convention, and the resulting NeRF will look inaccurate. Here are the steps taken to resolve this issue.
1. Within Unity, save the position and quaternion values for each frame.
2. When compiling the ```transforms.json``` file within ```get_transforms.py```, we will swap the y-axis and z-axis columns of each quaternion and flip the switch of each of four columns, i.e. ```[x y z w ]``` becomes ```[-x -z -y -w]```.
3. Initialize a 3x3 Rotation Matrix represented from the original quaternion data using ```scipy.spatial.transform.Rotation.from_quat```.
4. Now for each frame, we combine the quaternion and position data to construct the 4x4 matrix. The resulting 4x4 matrix should look like 
$`
\left(\begin{array}{cc} 
Q[0,0] & Q[0,1] & Q[0,2] & P[0]\\
Q[1,0] & Q[1,1] & Q[1,2] & P[1]\\
Q[2,0] & Q[2,1] & Q[2,2] & P[2]\\
0 & 0 & 0 & 1
\end{array}\right)
`$
Where $Q$ represents the quaternion data and $P$ represents position data.

Another challenge was the problem that our groundtruth method was not matching the quality of NeRFs generated through COLMAP. To fix this, we used the intrinsic matrix values that COLMAP calculated. Since we know that intrinsic values will not change if we use the same physical camera, we hard-coded these values into our groundtruth method. With this change, the GT NeRF quality is equal to the quality of the COLMAP NeRFs.


