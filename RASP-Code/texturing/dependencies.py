from tqdm import tqdm  # Import tqdm for progress bars
import os  # Import os module for file and directory operations
import sys  # Import sys module for system-specific parameters and functions
import torch  # Import PyTorch for tensor computations and deep learning functionalities
import pytorch3d  # Import pytorch3d for 3D deep learning and rendering utilities
from pytorch3d.io import load_objs_as_meshes, load_obj  # Import functions to load .obj files as meshes
from pytorch3d.renderer import (  # Import various renderer components from pytorch3d
    look_at_view_transform,  # Function to compute camera extrinsics (rotation and translation)
    FoVPerspectiveCameras,  # Class for perspective cameras with field-of-view specification
    FoVOrthographicCameras,  # Class for orthographic cameras with field-of-view specification
    PointLights,  # Class for point light sources
    DirectionalLights,  # Class for directional light sources
    Materials,  # Class for material properties used in rendering
    RasterizationSettings,  # Class for configuring rasterization (e.g., image size, blur radius)
    MeshRenderer,  # Class for mesh rendering pipeline
    MeshRasterizer,  # Class for rasterizing meshes into pixel fragments
    SoftPhongShader,  # Shader class implementing soft Phong shading
    TexturesUV,  # Class for applying textures based on UV coordinates
    TexturesVertex,  # Class for applying vertex colors as textures
    SoftSilhouetteShader  # Shader class for rendering soft silhouettes
)
import matplotlib.pyplot as plt  # Import matplotlib's pyplot module for plotting
import numpy as np  # Import numpy for numerical operations and array handling
from pytorch3d.vis.plotly_vis import plot_scene  # Import function to visualize 3D scenes using Plotly
from pytorch3d.structures.meshes import Meshes, join_meshes_as_scene  # Import Meshes class and function to join multiple meshes
import math  # Import math module for mathematical functions
from pytorch3d.loss import (  # Import various mesh loss functions from pytorch3d
    chamfer_distance,  # Function to compute the Chamfer distance between point clouds
    mesh_edge_loss,  # Function to compute loss based on mesh edge lengths
    mesh_laplacian_smoothing,  # Function for mesh Laplacian smoothing loss
    mesh_normal_consistency  # Function to compute loss for mesh normal consistency
)
from pysdf import SDF  # Import SDF class from pysdf for signed distance function computations
import plotly.graph_objects as go  # Import Plotly graph objects for interactive visualization
from pytorch3d.transforms import quaternion_apply, euler_angles_to_matrix, axis_angle_to_quaternion  # Import transformation functions for applying rotations

def set_device(id=0):  # Define a function to set the computation device, default id is 0
    if id == 'cpu':  # Check if the provided id is 'cpu'
        device = torch.device("cpu")  # Set device to CPU
        print("Using device: cpu")  # Print confirmation message
        return device  # Return the CPU device
    device_str = "cuda:{}".format(id)  # Format the device string for CUDA with the given id
    if torch.cuda.is_available():  # Check if CUDA is available
        device = torch.device(device_str)  # Set device to the specified CUDA device
        torch.cuda.set_device(device)  # Explicitly set the current CUDA device
        print("Using device: {}".format(device_str))  # Print confirmation message with CUDA device
    else:  # If CUDA is not available
        device = torch.device("cpu")  # Fallback to CPU
        print("Using device: cpu")  # Print confirmation message for CPU
    return device  # Return the selected device
