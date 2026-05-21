from dependencies import *  # Import all necessary dependencies from a custom module
import torch.nn.functional as F  # Import functional operations for neural networks from PyTorch
import torch.nn as nn  # Import the neural network module from PyTorch

# Define a class to encapsulate different loss functions
class Losses:
    # Compute intersection loss based on the environment's intersection values
    def get_intersection_loss(environment, mesh_fields):
        environment.get_intersections(mesh_fields)  # Compute intersections between environment and mesh fields
        loss = environment.intersection_value  # Retrieve the intersection loss value
        return loss  # Return the computed loss

    # Compute bounding loss for mesh fields based on environment constraints
    def get_environment_boundings_loss(mesh_fields, environment, target_dim):
        # Initialize loss as a tensor with value 0.0, placed on the same device as the environment
        loss = torch.tensor(0.0, dtype=torch.float32, device=environment.device)
        
        # Iterate through each mesh field to compute the smooth signed distance sum
        for mesh in mesh_fields:
            loss += environment.smooth_signed_distance_sum(
                mesh.mesh.verts_packed(),  # Get packed vertex positions of the mesh
                target_dim[0],  # Target bounding dimension along x-axis
                target_dim[1],  # Target bounding dimension along y-axis
                target_dim[2]   # Target bounding dimension along z-axis
            )
        return loss  # Return the computed loss

    # Compute silhouette loss by comparing projected mesh silhouettes to a target silhouette
    def get_silhouette_loss(renderer, mesh_fields, target_silhouette):
        silhouette = renderer.project_silhouette([mesh.mesh for mesh in mesh_fields])  # Project mesh silhouettes
        loss = ((silhouette - target_silhouette)**2).sum()  # Compute squared difference and sum over all pixels
        return loss  # Return the computed silhouette loss