from dependencies import *  # Import everything from the dependencies module
import torch.nn.functional as F  # Import functional API from PyTorch for loss functions and other operations
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import pdb  # Import pdb, Python debugger, for debugging purposes
from PIL import Image  # Import Image module from PIL for image processing

# Define the Losses class containing various loss functions for mesh processing
class Losses:
    def get_intersection_loss(environment, mesh_fields, mode):  # Method to compute the intersection loss given an environment, list of mesh fields, and a mode
        environment.get_intersections(mesh_fields, mode)  # Call the environment's method to compute intersections using the specified mode
        loss = environment.intersection_value  # Retrieve the intersection value computed by the environment
        return loss  # Return the intersection loss

    def get_distance_loss(mesh_fields):  # Method to compute a distance loss across mesh fields
        loss = 0.0  # Initialize the loss to zero
        for mesh in mesh_fields:  # Loop over each mesh in the mesh fields
            loss += abs(mesh.mesh.get_mesh_verts_faces(0)[0].sum())  # Accumulate absolute sum of all vertex coordinates from the mesh
        return loss  # Return the total distance loss

    def get_silhouette_loss(renderer, mesh_fields, target_silhouette, use_bce_loss=False):  # Method to compute the silhouette loss comparing projected silhouettes to a target silhouette; can use BCE or MSE
        silhouette = renderer.project_silhouette([mesh.mesh for mesh in mesh_fields])  # Project silhouettes for all meshes using the renderer
        silhouette = silhouette.clamp(0, 1)  # Clamp silhouette values between 0 and 1
        if use_bce_loss:  # If using Binary Cross-Entropy (BCE) loss
            # Compute binary cross-entropy loss for each channel separately
            loss_r = F.binary_cross_entropy(silhouette[:, :, 0], target_silhouette[:, :, 0])  # Compute BCE loss for red channel
            loss_g = F.binary_cross_entropy(silhouette[:, :, 1], target_silhouette[:, :, 1])  # Compute BCE loss for green channel
            loss_b = F.binary_cross_entropy(silhouette[:, :, 2], target_silhouette[:, :, 2])  # Compute BCE loss for blue channel
            loss = (loss_r + loss_g + loss_b) / 3  # Average the losses from all channels
        else:
            # Compute mean squared error (MSE) loss for each channel separately
            loss_r = ((silhouette[:, :, 0] - target_silhouette[:, :, 0]) ** 2).mean()  # Compute MSE loss for red channel
            loss_g = ((silhouette[:, :, 1] - target_silhouette[:, :, 1]) ** 2).mean()  # Compute MSE loss for green channel
            loss_b = ((silhouette[:, :, 2] - target_silhouette[:, :, 2]) ** 2).mean()  # Compute MSE loss for blue channel
            loss = (loss_r + loss_g + loss_b) / 3  # Average the losses from all channels
        return loss  # Return the computed silhouette loss

    def get_environment_boundings_loss(mesh_fields, environment):  # Method to compute a loss based on how much the mesh points extend beyond environment bounds
        loss = 0.0  # Initialize the loss to zero
        bound = environment.N  # Retrieve the boundary value from the environment
        for mesh in mesh_fields:  # Loop over each mesh in the mesh fields
            loss += torch.sum(torch.abs(mesh.sdf_points) > bound).float()  # Accumulate loss by summing the number of SDF points whose absolute value exceeds the bound, cast to float
        return loss  # Return the total bounding loss
