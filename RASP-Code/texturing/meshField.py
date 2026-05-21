from dependencies import *  # Import all necessary dependencies from the dependencies module

# Define the meshField class to handle mesh operations and SDF computations
class meshField:
    def __init__(self, device=set_device()):     # Constructor with an optional device parameter (default set via set_device())
        self.device = device   # Store the device (e.g., CPU or GPU)
        self.mesh = None  # Initialize the mesh attribute to None
        self.sdf_points = None  # Initialize the SDF points attribute to None
        self.sdf_values = None  # Initialize the SDF values attribute to None

    def populate_Mesh(self, vertices, faces, scaling_factor=1.0, environment_factor=1.0):  # Method to create and populate the mesh from vertices and faces
        faces_idx = faces.verts_idx  # Extract face indices from the faces object
        vertices = vertices  # Assign the vertices (no modification done)

        num_verts = vertices.shape[0]  # Determine the number of vertices
        color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=self.device)  # Create a white color tensor
        color = color.repeat(num_verts, 1)  # Repeat the color for each vertex
        textures = TexturesVertex(verts_features=color.unsqueeze(0))  # Create a vertex texture using the repeated color

        self.mesh = Meshes(  # Create a Meshes object with vertices, face indices, and textures
            verts=[vertices],
            faces=[faces_idx],
            textures=textures
        ).to(self.device)  # Move the mesh to the specified device

    def transform_mesh(self, color, rotate, translate):  # Method to apply transformation and color update to the mesh
        rotate = axis_angle_to_quaternion(rotate)  # Convert rotation from axis-angle representation to quaternion
        self.mesh.verts_list()[0] = quaternion_apply(rotate, self.mesh.verts_list()[0])  # Apply the quaternion rotation to the mesh vertices
        self.mesh.verts_list()[0] += translate  # Translate the mesh vertices by the given translation vector
        color = color.repeat(self.mesh.verts_list()[0].shape[0], 1)  # Repeat the input color for each vertex in the mesh
        self.mesh.textures = TexturesVertex(verts_features=color.unsqueeze(0))  # Update the mesh textures with the new vertex colors

    def colorize_mesh(self, color):  # Method to update the mesh with a new color
        # color = color.repeat(self.mesh.verts_list()[0].shape[0], 1)  (This line is commented out)
        color = torch.stack(color, dim=0)  # Stack the color components into a tensor
        self.mesh.textures = TexturesVertex(verts_features=color.unsqueeze(0))  # Update the mesh textures with the new color tensor
 
    def clone(self):  # Method to create a copy of the current meshField object
        new_meshField = meshField()  # Instantiate a new meshField object
        new_meshField.mesh = self.mesh.clone()  # Clone the mesh and assign it to the new object
        return new_meshField  # Return the cloned meshField
