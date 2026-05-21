import torch  # Import PyTorch for tensor operations and deep learning functionality
from pytorch3d.io import load_objs_as_meshes  # Import function to load .obj files as PyTorch3D mesh objects
from pytorch3d.renderer import TexturesVertex  # Import TexturesVertex for applying vertex colors as textures
from pytorch3d.structures.meshes import join_meshes_as_scene  # Import function to join multiple meshes into one scene
import os  # Import os module for operating system related functions
import plotly.graph_objects as go  # Import Plotly's graph objects for interactive 3D visualization
import argparse  # Import argparse for parsing command-line arguments

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Set the device to GPU if available, otherwise CPU

parser = argparse.ArgumentParser()  # Create an argument parser object
parser.add_argument('--folder', type=str, help='Folder containing .obj files')  # Add a command-line argument for the folder path containing .obj files
selected_folder = parser.parse_args().folder  # Parse the arguments and store the provided folder path in selected_folder

def mesh_apply_colors(mesh, device):  # Define a function to apply random colors to the vertices of a mesh
    verts = mesh.verts_packed()  # Retrieve the packed vertices from the mesh
    num_verts = verts.shape[0]  # Get the number of vertices in the mesh
    random_color = torch.rand((1, 3)).to(device)  # Generate a random color (RGB) and move it to the specified device
    random_color = random_color.repeat(num_verts, 1)  # Repeat the random color for each vertex
    textures = TexturesVertex(verts_features=random_color.unsqueeze(0))  # Create a TexturesVertex object with the vertex colors
    mesh.textures = textures  # Assign the textures (vertex colors) to the mesh
    return mesh  # Return the colored mesh

def visualize_mesh(meshes, device):  # Define a function to visualize a list of meshes using Plotly
    colored_meshes = [mesh_apply_colors(mesh, device) for mesh in meshes]  # Apply random colors to each mesh in the list
    combined_mesh = join_meshes_as_scene(colored_meshes)  # Combine all colored meshes into a single scene

    verts = combined_mesh.verts_packed().cpu().numpy()  # Get the combined mesh's vertices as a NumPy array (on CPU)
    faces = combined_mesh.faces_packed().cpu().numpy()  # Get the combined mesh's face indices as a NumPy array (on CPU)
    colors = combined_mesh.textures.verts_features_packed().cpu().numpy()  # Get the vertex colors as a NumPy array (on CPU)

    # Normalize colors to range [0, 1]
    colors = colors.clip(0, 1)  # Clip the colors to ensure values are within [0, 1]

    # Create Plotly mesh
    mesh = go.Mesh3d(
        x=verts[:, 0],  # Set x-coordinates of vertices
        y=verts[:, 1],  # Set y-coordinates of vertices
        z=verts[:, 2],  # Set z-coordinates of vertices
        i=faces[:, 0],  # Set first vertex indices for each face
        j=faces[:, 1],  # Set second vertex indices for each face
        k=faces[:, 2],  # Set third vertex indices for each face
        vertexcolor=colors,  # Assign vertex colors to the mesh
        name='Mesh'  # Set the name of the mesh for the plot legend
    )

    fig = go.Figure(data=[mesh])  # Create a Plotly Figure with the mesh as data
    fig.update_layout(scene_aspectmode='data')  # Update the layout to maintain aspect ratio based on data
    fig.show()  # Display the interactive 3D plot

def main(selected_folder):  # Define the main function that processes the folder of .obj files
    meshes = []  # Initialize an empty list to store loaded meshes
    mesh_names = [f for f in os.listdir(selected_folder) if f.endswith('.obj')]  # List all .obj files in the selected folder
    for mesh_name in mesh_names:  # Loop through each .obj file
        mesh = load_objs_as_meshes([os.path.join(selected_folder, mesh_name)], device=device)  # Load the .obj file as a mesh, specifying the device
        meshes.append(mesh)  # Append the loaded mesh to the meshes list
    visualize_mesh(meshes, device)  # Call the visualize_mesh function to display the combined scene of meshes

main(selected_folder)  # Execute the main function with the folder provided via command-line argument
