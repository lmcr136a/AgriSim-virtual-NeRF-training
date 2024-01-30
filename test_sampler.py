from unity_sampler import Sampler, CameraViewpoint
import json    

if __name__ == "__main__":
    # Instantiate Python Sampler
    sampler = Sampler()

    # Declare sample ShapeNet object IDs
    object_family="02843684"
    object_id="1b73f96cf598ef492cba66dc6aeabcd4"

    # Declare additional camera poses, in the form of (x, y, z, theta, phi)
    view1, view2 = CameraViewpoint(5.0, 5.0, 5.0, 90.0, 90.0), CameraViewpoint(10.0, 5.0, 12.0, 69.0, 62.0)
    test=[view1, view2]

    # Declare destination directory paths
    nerf_dest = "/home/brian/Desktop/instant-ngp/data/nerf/python_module_test/"
    screenshot_dir = "/home/brian/Desktop/AgriSim/result_images/"
    screenshot_transforms = "./screenshot_transforms.json"

    # Declare camera poses to take screenshots from (x, y, z, theta, phi)
    poses = [(5, 5, 5, 90, 90), (10, 5, 12, 69, 62)]

    # Run the sampler
    sampler.generate_nerf(object_family=object_family, 
                        object_id=object_id, 
                        nerf_dest=nerf_dest,                    
                        result_poses=poses,     
                        screenshot_transforms=screenshot_transforms,
                        screenshot_dir=screenshot_dir,
                        additional_viewpoints=test, 
                        produce_mesh=True)
