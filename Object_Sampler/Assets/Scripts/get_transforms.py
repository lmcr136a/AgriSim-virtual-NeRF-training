import cv2 
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R


# A constant used to convert Unity matrix to Instant-NGP convention.
ROTATION_MATRIX = np.array([[  1.0000000,  0.0000000,  0.0000000],
   [0.0000000,  0.0000000, -1.0000000],
   [0.0000000,  1.0000000,  0.0000000 ]])

SCREENSHOTS_DIR = "./Object_Sampler/Assets/screenshots/"


def get_camera_properties():
    """
    Parses the intrinsic camera properties obtained from Unity. 
    """

    properties = []
    with open(f"{SCREENSHOTS_DIR}camera_properties.txt") as f:
        contents = f.read()
        properties = contents.split(sep='\n')[:-1]
        properties = [float(p) for p in properties]
    return properties


def compute_sharpness(img_path): 
    """
    Computes the sharpness of an image.
    """

    # Load the image from a file
    img = cv2.imread(img_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Compute the sharpness metric as the variance of the Laplacian
    sharpness = laplacian.var()

    return sharpness


# read in the data and store it in a list of lists
def read_data(file_path):
    """
    Reads in the data and stores it in a list of lists.
    """
    res = []
    with open(file_path) as f:
        contents = f.read()
        positions = contents.split(sep='\n')[:-1]
        for pos in positions:
            res.append(pos[1:-1])
    # transforms = convert_to_instantngp_convention(transforms_list[1:])
    res = np.asarray(res)
    return res


def write_data(data, file_path):
    """
    Writes data into file_path.
    """

    with open(file_path, "w") as f:
        for i in range(len(data)):
            f.write(str(data[i]) + "\n")


def convert_to_nerf_convention(rotation_matrix, original_matrices):
    """
    Converts matrix into Instant-NGP Convention

            Parameters:
                    rotation_matrix: matrix used to rotate original matrix
                    original_matrices: List of original Unity convention matrices

            Returns:
                    new_matrices (list): List of Instant-NGP convention matrices
    """

    new_matrices = []
    for original_matrix in original_matrices:
        copy_og = original_matrix.copy()
        R = original_matrix[:3, :3]
        new_R = R @ rotation_matrix
        copy_og[:3, :3] = new_R
        new_matrices.append(copy_og)
    return new_matrices


def get_correct_transforms(position_source, quaternion_source, position_dest, quaternion_dest):
    """
    Given the raw data of Unity positions/quaternions, convert that into a list of 
    transform matrices that can be passed through to Instant-NGP

            Parameters:
                    position_source (str): File path of position data
                    quaternion_source (str): File path of quaternion data
                    position_dest (str): File path of new position data
                    quaternion_dest (str): File path of new quaternion data

            Returns:
                    predicted_nerf_matrices (list): List of 4x4 transform matrices
    """

    # Read and write position/quaternion data into a more readable format
    position_data = read_data(position_source)
    quaternion_data = read_data(quaternion_source)
    write_data(position_data, position_dest)
    write_data(quaternion_data, quaternion_dest)
    np_position_data = np.loadtxt(position_dest, delimiter=",")
    np_quaternion_data = np.loadtxt(quaternion_dest, delimiter=",")

    # Left to Right hand conversions
    # First, swap the y-axis and z-axis
    orig_y_vals = np_position_data[:, 1].copy()
    np_position_data[:, 1] = np_position_data[:, 2]
    np_position_data[:, 2] = orig_y_vals

    # Flip the sign of each column
    orig_y_quaternions = np_quaternion_data[:, 1].copy()
    np_quaternion_data[:, 1] = -1 * np_quaternion_data[:, 2]
    np_quaternion_data[:, 2] = -1 * orig_y_quaternions
    np_quaternion_data[:, 0] = -1 * np_quaternion_data[:, 0]

    num_matrices = np_position_data.shape[0]

    transform_matrices = np.zeros((num_matrices, 4, 4))

    # Combine position and quaternion data to construct a 4x4 transform matrix
    for i in range(num_matrices):
        # Initialize a 3D Rotation matrix represented from the quaternion data
        r = R.from_quat(np_quaternion_data[i])
        T = np.zeros((4, 4))
        T[:3, :3] = r.as_matrix()
        T[-1, -1] = 1
        T[0, -1] = np_position_data[i, 0]
        T[1, -1] = np_position_data[i, 1]
        T[2, -1] = np_position_data[i, 2]
        transform_matrices[i] = T
    predicted_nerf_matrices = convert_to_nerf_convention(ROTATION_MATRIX, transform_matrices)
    predicted_nerf_matrices = [predicted_nerf_matrices[i].tolist() for i in range(len(predicted_nerf_matrices))]

    return predicted_nerf_matrices # returns a list of 4x4 lists 


def img_to_transform_matrix(img_path):
    """
    Obtain corresponding transform matrix from the image path

            Parameters:
                    img_path (str): File path containing all images.

            Returns:
                    transform_matrix (np.ndarray): A 4x4 transform matrix
    """
        
    # get a list of the transform matrices
    transforms = get_correct_transforms(f"{SCREENSHOTS_DIR}positions.txt", f"{SCREENSHOTS_DIR}quaternions.txt", f"{SCREENSHOTS_DIR}positions2.txt", f"{SCREENSHOTS_DIR}quaternions2.txt")
    # get the index of the image
    img_index = int(img_path.split(sep='pic')[1].split(sep='.')[0]) 
    return transforms[img_index]


def create_frames_arr():
    """
    Creates a list of dictionaries that contain information about each frame

            Returns:
                    frames_arr (List): List of dictionaries of each frame
    """


    # Get all files in the directory
    all_files = os.listdir(SCREENSHOTS_DIR)

    # Filter the files to only include PNG files
    png_files = [SCREENSHOTS_DIR + file for file in all_files if file.endswith(".png")]

    frames_arr = []

    for img in png_files:
        # add the file path, sharpness, and transform matrix to the dictionary
        paths = img.split("/")
        file_path = "./images/" + paths[2]
        frames_dict = {
            "file_path": file_path,
            "sharpness": compute_sharpness(img_path=img),
            "transform_matrix": img_to_transform_matrix(img_path=img)
        }
        frames_arr.append(frames_dict)
    return frames_arr


def get_image_shape(img_path):
    """
    Returns the shape of an image

            Parameters:
                    img_path (str): File path of image

            Returns:
                    img_shape (np.ndarray): Shape of image
    """
    img = cv2.imread(img_path)
    return img.shape


def create_transforms_json():
    """
    Compiles the transforms.json file needed to construct a NeRF
    """
    dictionary = {
        "camera_angle_x": get_camera_properties()[0],
        "camera_angle_y": get_camera_properties()[1],
        "fl_x": get_camera_properties()[2],
        "fl_y": get_camera_properties()[3],
        "k1": get_camera_properties()[4],
        "k2": get_camera_properties()[5],
        "p1": get_camera_properties()[6],
        "p2": get_camera_properties()[7],
        "cx": get_camera_properties()[8],
        "cy": get_camera_properties()[9],
        "w": get_image_shape(f"{SCREENSHOTS_DIR}pic1.png")[1],
        "h": get_image_shape(f"{SCREENSHOTS_DIR}pic1.png")[0],
        "aabb_scale": 2,
        "frames": create_frames_arr()
    }
    json_object = json.dumps(dictionary, indent=4)
    # Writing to sample.json
    with open(f"{SCREENSHOTS_DIR}transforms.json", "w") as f:
        f.write(json_object)


if __name__ == "__main__":
    create_transforms_json()