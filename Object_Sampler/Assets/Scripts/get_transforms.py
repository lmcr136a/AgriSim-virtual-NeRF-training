import cv2 
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_camera_properties():
    properties = []
    with open("../screenshots/camera_properties.txt") as f:
        contents = f.read()
        properties = contents.split(sep='\n')[:-1]
        properties = [float(p) for p in properties]
    return properties

def compute_sharpness(img_path): 
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
    with open(file_path, "w") as f:
        for i in range(len(data)):
            f.write(str(data[i]) + "\n")

# constant rotation matrix to rotate our Unity matrix to instant ngp's convention
rotation_matrix = np.array([[  1.0000000,  0.0000000,  0.0000000],
   [0.0000000,  0.0000000, -1.0000000],
   [0.0000000,  1.0000000,  0.0000000 ]])

def convert_to_nerf_convention(rotation_matrix, original_matrices):
    new_matrices = []
    for original_matrix in original_matrices:
        copy_og = original_matrix.copy()
        R = original_matrix[:3, :3]
        new_R = R @ rotation_matrix
        copy_og[:3, :3] = new_R
        new_matrices.append(copy_og)
    return new_matrices

def get_correct_transforms(position_source, quaternion_source, position_dest, quaternion_dest):
    position_data = read_data(position_source)
    quaternion_data = read_data(quaternion_source)
    write_data(position_data, position_dest)
    write_data(quaternion_data, quaternion_dest)
    np_position_data = np.loadtxt(position_dest, delimiter=",")
    np_quaternion_data = np.loadtxt(quaternion_dest, delimiter=",")

    # Left to Right hand conversions
    orig_y_vals = np_position_data[:, 1].copy()
    np_position_data[:, 1] = np_position_data[:, 2]
    np_position_data[:, 2] = orig_y_vals

    orig_y_quaternions = np_quaternion_data[:, 1].copy()
    np_quaternion_data[:, 1] = -1 * np_quaternion_data[:, 2]
    np_quaternion_data[:, 2] = -1 * orig_y_quaternions
    np_quaternion_data[:, 0] = -1 * np_quaternion_data[:, 0]

    num_matrices = np_position_data.shape[0]

    transform_matrices = np.zeros((num_matrices, 4, 4))

    for i in range(num_matrices):
        r = R.from_quat(np_quaternion_data[i])
        T = np.zeros((4, 4))
        T[:3, :3] = r.as_matrix()
        T[-1, -1] = 1
        T[0, -1] = np_position_data[i, 0]
        T[1, -1] = np_position_data[i, 1]
        T[2, -1] = np_position_data[i, 2]
        transform_matrices[i] = T
    predicted_nerf_matrices = convert_to_nerf_convention(rotation_matrix, transform_matrices)
    predicted_nerf_matrices = [predicted_nerf_matrices[i].tolist() for i in range(len(predicted_nerf_matrices))]
    return predicted_nerf_matrices # returns a list of 4x4 lists 

# obtain the corresponding transform matrix from the image path
def img_to_transform_matrix(img_path):
    # get a list of the transform matrices
    transforms = get_correct_transforms("../screenshots/positions.txt", "../screenshots/quaternions.txt", "../screenshots/positions2.txt", "../screenshots/quaternions2.txt")
    # get the index of the image
    img_index = int(img_path.split(sep='pic')[1].split(sep='.')[0]) 
    return transforms[img_index]

# return a list of dictionaries. each dictionary will have keys of "file_path", "sharpness", and "transform_matrix"
def create_frames_arr():
    # Define the directory where the PNG files are located
    directory = "../screenshots/"

    # Get all files in the directory
    all_files = os.listdir(directory)

    # Filter the files to only include PNG files
    png_files = [directory + file for file in all_files if file.endswith(".png")]

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

# a function that returns the shape of our images
def get_image_shape(img_path):
    img = cv2.imread(img_path)
    return img.shape

def create_transforms_json():
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
        "w": get_image_shape("../screenshots/pic0.png")[1],
        "h": get_image_shape("../screenshots/pic0.png")[0],
        "aabb_scale": 2,
        "frames": create_frames_arr()
    }
    json_object = json.dumps(dictionary, indent=4)
    # Writing to sample.json
    with open("transforms.json", "w") as f:
        f.write(json_object)
    with open("transforms.json", "r") as f:
        json_data = f.read()
        print(json_data)

if __name__ == "__main__":
    create_transforms_json()