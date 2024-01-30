from dataclasses import dataclass
from peaceful_pie.unity_comms import UnityComms
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import shutil
import os
import zipfile
import subprocess
import json
import numpy as np

CAMERA_PROPERTIES = [1.29, 0.88, 567, 567, 0, 0, 0, 0, 425, 268, 850, 531, 8]

@dataclass
class CameraViewpoint:
    x: float
    y: float
    z: float
    theta: float
    phi: float

class Sampler:
    def __init__(self, unity_port=9000):
        self.unity_comms = UnityComms(port=unity_port)
        self.object_path = None

    def generate_nerf(self, object_family, 
                      object_id, 
                      nerf_dest, 
                      result_poses,
                      screenshot_transforms, 
                      screenshot_dir, 
                      additional_viewpoints=None, 
                      produce_mesh=False):
        """
        Samples the desired object and produces a NeRF using Instant-NGP
    
        Args:
            object_family (str): Object family 
            object_id (str): ID of the desired object
            nerf_dest (str): Absolute file path to the instant-ngp object directory
            result_poses (List[Tuple[float]]): Tells Instant-NGP API where to take pictures of the resulting NeRF
            screenshot_transforms (str): transforms.json file describing the resulting screenshots
            additional_viewpoints (List[CameraViewpoint]): List of camera viewpoints to sample from.
                (x,y,z,theta,phi), where xyz is the position and theta, phi represents the viewing direction of the camera
            produce_mesh (bool): If True, this function will return a mesh file of the resulting nerf
    
        Returns:
            Snapshots and mesh file of the resulting NeRF
        """

        # Load in the object from HuggingFace into Unity scene
        self._load_object(object_family, object_id)
        self._place_object_in_scene(self.object_path)

        # Call Unity's function to sample from default viewpoints
        self.unity_comms.sample_default_viewpoints(max_poses=50)
        
        # Call Unity's function to sample from additional viewpoints
        for pose in additional_viewpoints:
            self.unity_comms.sample_additional_viewpoints(pose=pose)

        # Write camera pose information to files
        self.unity_comms.sampling_cleanup()

        # Call automate script to generate a NeRF
        produce_mesh_arg = "true" if produce_mesh else "false"
        self._generate_screenshot_transforms(result_poses, screenshot_dir)
        subprocess.run(["sh", "./automate.sh", nerf_dest, screenshot_dir, screenshot_transforms, produce_mesh_arg])

    def _load_object(self, object_family, object_id):
        """
        Loads in the desired object from ShapeNet
    
        Args:
            object_id (str): The desired object
        """

        # Download object family from Hugging Face hub
        hub_filepath = object_family + ".zip"
        cache_path = hf_hub_download(repo_id="ShapeNet/ShapeNetCore", filename=hub_filepath, repo_type="dataset")
        cache_dir = os.path.dirname(cache_path)

        print(f"Everything in cache directory: {os.listdir(cache_dir)}, and the current cache path is {cache_path}")
        
        # Unzip the source_path within the cache
        with zipfile.ZipFile(cache_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)

        # Get the object_id and copy it into the Unity Assets folder
        source_path = "/".join([cache_dir, object_family, object_id])
        print(f"The source path is {source_path}")

        # NOTE: Make sure the current directory we run this script from is AgriSim 
        destination_path = "./Object_Sampler/Assets/shapenet_objects/" + object_id
        shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        self.object_path = "/shapenet_objects/" + object_id + "/models/model_normalized.obj"

    
    def _place_object_in_scene(self, object_path):
        """
        Loads in the desired object from ShapeNet
    
        Args:
            object_path (str): The file path of the desired object
        """

        # Pass in the object_path into the JsonRpcMethod to place the Object into Unity Scene
        print(f"The desired object path is {object_path}")
        self.unity_comms.spawn_object(obj_path=object_path)


    def _delete_files_in_directory(self, directory_path):
        """
        Deletes the contents of a directory
    
        Args:
            directory_path (str): Path to the directory
        """

        try:
            files = os.listdir(directory_path)
            for file in files:
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except OSError:
            print("Error occurred while deleting files.")


    def _generate_screenshot_transforms(self, poses, screenshot_dir):
        """
        Generate a transforms.json file that tells instant-ngp api where to take screenshots of the resulting NeRF.
    
        Args:
            poses (List[Tuple[float]]): Camera angles to take screenshots from
            screenshot_dir (str): Directory path to save screenshots
        """
        dictionary = {
            "camera_angle_x": CAMERA_PROPERTIES[0],
            "camera_angle_y": CAMERA_PROPERTIES[1],
            "fl_x": CAMERA_PROPERTIES[2],
            "fl_y": CAMERA_PROPERTIES[3],
            "k1": CAMERA_PROPERTIES[4],
            "k2": CAMERA_PROPERTIES[5],
            "p1": CAMERA_PROPERTIES[6],
            "p2": CAMERA_PROPERTIES[7],
            "cx": CAMERA_PROPERTIES[8],
            "cy": CAMERA_PROPERTIES[9],
            "w": CAMERA_PROPERTIES[10],
            "h": CAMERA_PROPERTIES[11],
            "aabb_scale": CAMERA_PROPERTIES[12],
            "frames": self._camera_poses_to_transform_matrices(poses)
        }
        json_object = json.dumps(dictionary, indent=4)
        # Writing to sample.json
        with open(f"{screenshot_dir}screenshot_transforms.json", "w") as f:
            f.write(json_object)
    
    def _camera_poses_to_transform_matrices(self, poses):
        """
        Convert the camera poses into transformation matrices
    
        Args:
            poses (List[Tuple[float]]): Camera angles to take screenshots from
        Returns:
            Array of transformation matrices
        """

        res = []
        for i, pose in enumerate(poses):
            # Pose will be in the form (x, y, z, theta, phi)
            curr_transform = np.ones((4, 4))
            # Fill in translation values
            curr_transform[:3, 3] = pose[:3]
            # Calculate the 3x3 Rotation matrix using theta and phi
            theta, phi = pose[3], pose[4]
            R = np.array([
                [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)],
                [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)],
                [-np.sin(phi), np.cos(phi), 0]
            ])

            curr_transform[:3, :3] = R
            frames_dict = {
                "file_path": f"screenshot{i}.png",
                "sharpness": 100,
                "transform_matrix": curr_transform,
                "transform_matrix_start": curr_transform
            }
        return res
