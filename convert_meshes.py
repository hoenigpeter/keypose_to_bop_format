import os
import trimesh
import numpy as np
import argparse

def process_mesh_files(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for index, filename in enumerate(sorted(os.listdir(input_folder)), start=1):
        if filename.endswith(".ply") or filename.endswith(".obj"):
            # Construct full file path
            file_path = os.path.join(input_folder, filename)
            
            # Load the mesh using trimesh
            mesh = trimesh.load(file_path)
            
            # Extract vertices and normals
            vertices = mesh.vertices * 1000
            normals = mesh.vertex_normals

            # Create a new mesh with the combined vertex data
            new_mesh = trimesh.Trimesh(vertices=vertices, vertex_normals=normals, faces=mesh.faces, process=False)

            # Save the new mesh to the output folder in ASCII format
            new_file_path = os.path.join(output_folder, f'obj_{index:06d}.ply')
            new_mesh.export(new_file_path, file_type='ply', encoding='ascii')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset to BOP format.")
    parser.add_argument("input_folder", type=str, help="Path to the mesh files.")
    args = parser.parse_args()

    output_folder = "./output_meshes"
    
    process_mesh_files(args.input_folder, output_folder)
