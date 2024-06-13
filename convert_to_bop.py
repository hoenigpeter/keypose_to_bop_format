# parts of the code from: https://github.com/google-research/google-research/tree/master/keypose/objects

import os
import json
import shutil
from glob import glob
from pathlib import Path
import re
import numpy as np
import cv2
import png
from tqdm import tqdm
import argparse
import OpenEXR
import Imath
import data_pb2 as pb
from google.protobuf import text_format
import utils

def read_exr(file_path):
    # Open the EXR file
    exr_file = OpenEXR.InputFile(file_path)
    
    # Get the header to extract information about the channels and size
    header = exr_file.header()
    data_window = header['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    
    # Get the channel names
    channels = header['channels'].keys()
    
    # Read the data for each channel
    channel_data = {}
    for channel in channels:
        # Read raw data from the channel
        raw_data = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
        
        # Convert the raw data to a numpy array
        channel_data[channel] = np.frombuffer(raw_data, dtype=np.float32).reshape((height, width))
    
    # Return the width, height, and channel data
    return width, height, channel_data

def parse_pbtxt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    transform = []
    camera = {}

    in_transform = False
    in_camera = False

    for line in lines:
        if 'kp_target {' in line:
            in_transform = True
        if 'camera {' in line:
            in_camera = True
        if in_transform and 'element:' in line:
            transform.append(float(line.split('element:')[1].strip()))
        if in_camera:
            match = re.match(r'\s*(\w+):\s*([0-9.]+)', line)
            if match:
                key, value = match.groups()
                camera[key] = float(value)
        if '}' in line:
            if in_transform:
                in_transform = False
            if in_camera:
                in_camera = False
                break

    return transform, camera

def save_depth(path, im):
    """Saves a depth image (16-bit) to a PNG file.

    :param path: Path to the output depth image file.
    :param im: ndarray with the depth image to save.
    """
    if str(path).split('.')[-1].lower() != 'png':
        raise ValueError('Only PNG format is currently supported.')

    im_uint16 = np.round(im).astype(np.uint16)

    # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
    w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
    with open(path, 'wb') as f:
        w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))

def convert_dataset(input_folder, output_folder, depth_type, copy_images, mesh, obj_id):
    input_path = Path(input_folder)
    output_folder = Path(output_folder)

    rgb_folder = output_folder / 'rgb'
    depth_folder = output_folder / 'depth'
    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(depth_folder, exist_ok=True)

    scene_gt = {}
    scene_camera = {}

    image_counter = 0
    subfolders = sorted([subfolder for subfolder in input_path.iterdir() if subfolder.is_dir()])
    total_files = sum(len(list(subfolder.glob("*_L.pbtxt"))) for subfolder in subfolders)
    
    print(f"Converting dataset from {input_folder} to {output_folder}...")
    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        for subfolder in subfolders:
            for image_id in range(10000):  # Assuming there won't be more than 10,000 images
                image_id_str = f"{image_id:06d}"
                
                # Process left image
                pbtxt_file_left = subfolder / f"{image_id_str}_L.pbtxt"
                if pbtxt_file_left.exists():

                    targs_pb = utils.read_target_pb(pbtxt_file_left)

                    kps_pb = targs_pb.kp_target
                    camera = targs_pb.kp_target.camera

                    keys_uvd_l, to_world_l, visible_l = utils.get_keypoints(kps_pb)
                    
                    q_matrix_camera = utils.q_matrix_from_camera(camera)
                    p_matrix_camera = utils.p_matrix_from_camera(camera)
                    xyzw_camera = utils.project_np(q_matrix_camera, keys_uvd_l.T)

                    kpts_to_mesh = obj.project_to_uvd(xyzw_camera, p_matrix_camera)

                    R = kpts_to_mesh[:3, :3]
                    t = kpts_to_mesh[:3, 3] * 1000

                    scene_gt[image_counter] = [{"cam_R_m2c": R.tolist(), "cam_t_m2c": t.tolist(), "obj_id": int(obj_id)}]

                    scene_camera[image_counter] = {
                        "cam_K": [camera.fx, 0, camera.cx, 0, camera.fy, camera.cy, 0, 0, 1],
                        "depth_scale": 1.0
                    }

                    if copy_images:
                        # Copy left image files
                        src_file = subfolder / f"{image_id_str}_L.png"
                        if src_file.exists():
                            dest_file = rgb_folder / f"{image_counter:06d}.png"
                            shutil.copy(src_file, dest_file)

                        # Copy and convert depth file
                        depth_ext = 'Do' if depth_type == 'opaque' else 'Dt'
                        src_file = subfolder / f"{image_id_str}_{depth_ext}.exr"
                        if src_file.exists():
                            # Read the EXR file
                            _, _, channel_data = read_exr(str(src_file))

                            depth_image = channel_data['D'] * 1000
                            assert depth_image is not None, 'Cannot open %s' % str(src_file)

                            if depth_image is None:
                                print(f"Warning: Depth image {src_file} could not be read. Skipping.")
                            else:
                                dest_file = depth_folder / f"{image_counter:06d}.png"
                                save_depth(dest_file, depth_image)

                    image_counter += 1
                    pbar.update(1)

    with open(output_folder / 'scene_gt.json', 'w') as f:
        json.dump(scene_gt, f, indent=4)

    with open(output_folder / 'scene_camera.json', 'w') as f:
        json.dump(scene_camera, f, indent=4)

    print(f"Conversion complete. Output saved to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset to BOP format.")
    parser.add_argument("input_folder", type=str, help="Path to the images and annotation files.")
    parser.add_argument("mesh_folder", type=str, help="Path to the meshes.")
    parser.add_argument("--depth_type", type=str, choices=['opaque', 'transparent'], default='opaque', help="Type of depth images to use (opaque or transparent).")
    parser.add_argument("--copy_images", action='store_true', help="Option to copy images.")

    args = parser.parse_args()
    
    data_folders = sorted(os.listdir(args.input_folder + "/data"))
    output_dir = args.input_folder + "/test"
    print(data_folders)
    print(output_dir)

    for obj_id, obj_name in enumerate(data_folders, start=1):

        print("obj_id: ", obj_id)
        print("obj_name: ", obj_name)

        obj_output_dir = output_dir + f"/{obj_id:06}"
        print("obj_output_dir: ", obj_output_dir)
        os.makedirs(obj_output_dir, exist_ok=True)

        mesh_file = args.mesh_folder + "/" + obj_name + ".obj"
        obj = utils.read_mesh(mesh_file, num=300)
        obj.large_points = False

        convert_dataset(args.input_folder + "/data/" + obj_name, obj_output_dir, args.depth_type, args.copy_images, obj, obj_id)
