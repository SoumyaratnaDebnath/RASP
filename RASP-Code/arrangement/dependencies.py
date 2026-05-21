from tqdm import tqdm  # Import tqdm for progress bars
import os  # Import os for operating system dependent functionality
import sys  # Import sys for system-specific parameters and functions
import torch  # Import torch for PyTorch functionalities
import pytorch3d  # Import pytorch3d for 3D deep learning utilities
from pytorch3d.io import load_objs_as_meshes, load_obj  # Import functions to load OBJ files as meshes
from pytorch3d.renderer import (  # Import various rendering utilities from pytorch3d
    look_at_view_transform,  # Function to compute camera view transformations
    FoVPerspectiveCameras,  # Class for perspective camera with field-of-view specification
    FoVOrthographicCameras,  # Class for orthographic cameras with field-of-view specification
    PointLights,  # Class for point light sources
    DirectionalLights,  # Class for directional light sources
    Materials,  # Class for material properties
    RasterizationSettings,  # Settings for the rasterizer
    MeshRenderer,  # Class for rendering meshes
    MeshRasterizer,  # Class for rasterizing meshes
    SoftPhongShader,  # Shader class for soft Phong shading
    TexturesUV,  # Class for UV texture mapping
    TexturesVertex,  # Class for vertex texture mapping
    SoftSilhouetteShader  # Shader class for soft silhouette rendering
)
import matplotlib.pyplot as plt  # Import matplotlib for plotting graphs and images
import numpy as np  # Import numpy for numerical operations
from pytorch3d.vis.plotly_vis import plot_scene  # Import function to plot 3D scenes using Plotly
from pytorch3d.structures.meshes import Meshes, join_meshes_as_scene  # Import Meshes class and function to join meshes
import math  # Import math module for mathematical functions
from pytorch3d.loss import (  # Import various loss functions from pytorch3d
    mesh_edge_loss,  # Loss function based on mesh edges
    mesh_laplacian_smoothing,  # Loss function for Laplacian smoothing of meshes
    mesh_normal_consistency,  # Loss function for ensuring normal consistency across mesh faces
)
from pysdf import SDF  # Import SDF class from pysdf for signed distance functions
import plotly.graph_objects as go  # Import graph objects from Plotly for advanced visualizations
from pytorch3d.transforms import quaternion_apply, euler_angles_to_matrix, axis_angle_to_quaternion  # Import transformation functions for rotations
from pytorch3d.loss import chamfer_distance  # Import chamfer_distance loss function from pytorch3d
from torch.optim.lr_scheduler import StepLR  # Import learning rate scheduler for optimizer adjustments

def set_device(id=0):
    if id == 'cpu':  # Check if the id is 'cpu'
        device = torch.device("cpu")  # Set device to CPU
        print("Using device: cpu")  # Print message indicating CPU usage
        return device  # Return the CPU device
    device_str = "cuda:{}".format(id)  # Create a CUDA device string based on the given id
    if torch.cuda.is_available():  # Check if CUDA is available
        device = torch.device(device_str)  # Set the device to the specified CUDA device
        torch.cuda.set_device(device)  # Set the current CUDA device
        print("Using device: {}".format(device_str))  # Print message indicating CUDA usage
    else:
        device = torch.device("cpu")  # Fallback: set device to CPU if CUDA is not available
        print("Using device: cpu")  # Print message indicating CPU usage
    return device  # Return the chosen device

discrete_colors = [  # Define a list of discrete RGB color values
    [255, 0, 0],     # red
    [0, 0, 255],     # blue
    [0, 255, 0],     # green
    [255, 255, 0],   # yellow
    [128, 0, 128],   # purple
    [128, 128, 128], # gray
    [255, 165, 0],   # orange
    [255, 192, 203], # pink
    [165, 42, 42],   # brown
    [0, 255, 255],   # cyan
    [255, 0, 255],   # magenta
    [0, 255, 128],   # lime
    [0, 128, 128],   # teal
    [128, 128, 0],   # olive
    [255, 215, 0],   # gold
]

pastel_colors = ["#FFB3BA","#AEC6CF","#FFDFBA","#FFFFBA","#B2FBA5","#B39EB5","#FF9AA2","#FFB7B2","#FFDAC1","#E2F0CB","#B3E5FC","#D1C4E9","#C5CAE9","#B2EBF2","#B2DFDB","#F8BBD0","#FFCCBC","#D7CCC8","#FFF9C4","#DCEDC8","#F0F4C3","#FFECB3","#E1BEE7","#CE93D8","#FFAB91"]
# Define a list of pastel color hex codes

def chamfer_distance_custom(S1, S2, N1, N2):
    S1 = S1.unsqueeze(-2).expand(-1, -1, N2, -1)  # Expand S1 tensor dimensions to match S2 for pairwise distance calculation
    S2 = S2.unsqueeze(1).expand(-1, N1, -1, -1)  # Expand S2 tensor dimensions to match S1 for pairwise distance calculation
    return torch.min(torch.sum((S1 - S2) ** 2, dim=-1), dim=-1)[0]  # Compute the minimum squared Euclidean distance along the last dimension

def chamfer_loss(Y, Y_cap, N1, N2, threshold=1.732, k=10.0):  # Define function chamfer_loss with parameters for inputs, threshold, and k
    xy_dis = chamfer_distance_custom(Y, Y_cap, N1, N2)[0]  # Compute Chamfer distance from Y to Y_cap and take the first output element
    yx_dis = chamfer_distance_custom(Y_cap, Y, N2, N1)[0]  # Compute Chamfer distance from Y_cap to Y and take the first output element
    mask_xy = torch.sigmoid(k * (threshold - xy_dis))  # Apply sigmoid to the difference between threshold and xy_dis scaled by k
    mask_yx = torch.sigmoid(k * (threshold - yx_dis))  # Apply sigmoid to the difference between threshold and yx_dis scaled by k
    loss = torch.sum(mask_xy) + torch.sum(mask_yx)  # Sum the masked distances from both directions to compute the loss
    return loss  # Return the final loss value

def chamfer_loss_weighted(Y, Y_cap, N1, N2, w1, w2, threshold=1.732, k=10.0):  # Define function chamfer_loss_weighted with additional weights w1 and w2
    xy_dis = chamfer_distance_custom(Y, Y_cap, N1, N2)  # Compute the custom Chamfer distance from Y to Y_cap
    yx_dis = chamfer_distance_custom(Y_cap, Y, N2, N1)  # Compute the custom Chamfer distance from Y_cap to Y
    mask_xy = torch.sigmoid(k * (threshold - xy_dis))  # Generate a sigmoid mask for the distance from Y to Y_cap
    mask_yx = torch.sigmoid(k * (threshold - yx_dis))  # Generate a sigmoid mask for the distance from Y_cap to Y
    loss = torch.sum(mask_xy * w1) + torch.sum(mask_yx * w2) + torch.sum(mask_xy * w2) + torch.sum(mask_yx * w1)  # Compute the weighted sum of the masks using weights w1 and w2 symmetrically
    return loss  # Return the computed weighted Chamfer loss