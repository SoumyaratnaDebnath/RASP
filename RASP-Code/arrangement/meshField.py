from dependencies import *  # Import all necessary dependencies from the custom module

# Define a class to represent a mesh field, including its structure and transformations
class meshField:
    def __init__(self, device=set_device()):  # Initialize the meshField class with a device for computation
        self.device = device  # Store the device (CPU or GPU)
        self.mesh = None  # Placeholder for the mesh object
        self.sdf_points = None  # Placeholder for SDF (Signed Distance Function) points
        self.sdf_values = None  # Placeholder for SDF values

    # Function to create and populate a mesh from given vertices and faces
    def populate_Mesh(self, vertices, faces, scaling_factor=1.0):
        faces_idx = faces  # Assign the face indices
        vertices = vertices  # Assign the vertex positions
        center = vertices.mean(0)  # Compute the center of the vertices
        vertices = vertices - center  # Translate vertices to the origin
        vertices = vertices * scaling_factor  # Scale the vertices

        # Create a mesh object using PyTorch3D and assign it to the class
        self.mesh = Meshes(
            verts=[vertices],
            faces=[faces_idx]
        ).to(self.device)  # Move the mesh to the specified device

    # Function to create a scale-consistent mesh
    def populate_Mesh_scale_consistent(self, vertices, faces, scaling_factor=1.0, environment_factor=1.0, consistency_factor=None):
        faces_idx = faces  # Assign face indices
        vertices = vertices  # Assign vertex positions
        center = vertices.mean(0)  # Compute the center of the vertices
        vertices = vertices - center  # Translate vertices to the origin

        vertices = vertices.to(self.device)  # Move vertices to the specified device
        faces_idx = faces_idx.to(self.device)  # Move face indices to the specified device

        # Determine the scale factor based on max absolute coordinate if no consistency factor is provided
        if consistency_factor is None:
            scale = max(vertices.abs().max(0)[0])
        else:
            scale = consistency_factor  # Use provided consistency factor

        vertices = vertices / scale  # Normalize vertices to the scale
        vertices = vertices * scaling_factor * environment_factor  # Apply scaling factors

        # Create the mesh object and move it to the specified device
        self.mesh = Meshes(
            verts=[vertices],
            faces=[faces_idx]
        ).to(self.device)
        
        return scale  # Return the computed or provided scale factor

    # Function to create a textured scale-consistent mesh
    def populate_textured_Mesh_scale_consistent(self, vertices, faces, scaling_factor=1.0, environment_factor=1.0, consistency_factor=None, texture=None):
        faces_idx = faces  # Assign face indices
        vertices = vertices  # Assign vertex positions
        center = vertices.mean(0)  # Compute the center of the vertices
        vertices = vertices - center  # Translate vertices to the origin

        vertices = vertices.to(self.device)  # Move vertices to the specified device
        faces_idx = faces_idx.to(self.device)  # Move face indices to the specified device

        # Determine the scale factor based on max absolute coordinate if no consistency factor is provided
        if consistency_factor is None:
            scale = max(vertices.abs().max(0)[0]).to(self.device)
        else:
            scale = consistency_factor.to(self.device)  # Use provided consistency factor

        vertices = vertices / scale  # Normalize vertices
        vertices = vertices * scaling_factor * environment_factor  # Apply scaling factors

        # Create a textured mesh object and move it to the specified device
        self.mesh = Meshes(
            verts=[vertices],
            faces=[faces_idx],
            textures=texture  # Assign texture to the mesh
        ).to(self.device)
        
        return scale  # Return the computed or provided scale factor

    # Function to compute the Signed Distance Function (SDF) for environment points
    def populate_SDF(self, environment_points):
        self.sdf_points, self.sdf_values = self.get_SDF(environment_points=environment_points)  # Compute SDF

    # Function to apply rotation and translation to the mesh
    def transform_mesh(self, rotate, translate):
        rotate = axis_angle_to_quaternion(rotate)  # Convert rotation from axis-angle to quaternion
        self.mesh.verts_list()[0] = quaternion_apply(rotate, self.mesh.verts_list()[0])  # Apply rotation
        self.mesh.verts_list()[0] += translate  # Apply translation

    # Function to compute the Signed Distance Function (SDF) for given environment points
    def get_SDF(self, environment_points):
        verts, faces = (  # Get mesh vertices and faces and convert to NumPy arrays
            self.mesh.get_mesh_verts_faces(0)[0].detach().cpu().numpy(),
            self.mesh.get_mesh_verts_faces(0)[1].detach().cpu().numpy()
        )
        environment_points = environment_points.cpu().numpy()  # Convert environment points to NumPy
        sdf = SDF(verts, faces)  # Compute SDF using a predefined function

        sdf_pts = []  # List to store points inside the SDF
        sdf_vals = []  # List to store corresponding SDF values

        # Iterate through each environment point to compute its SDF value
        for point in environment_points:
            if sdf.contains(point):  # Check if the point is inside the SDF
                sdf_pts.append(point)  # Store the point
                sdf_vals.append(sdf(point))  # Store the computed SDF value
        
        # Convert lists to tensors and move them to the specified device
        sdf_pts = torch.tensor(np.array(sdf_pts), dtype=torch.float32, device=self.device)
        sdf_vals = torch.tensor(np.array(sdf_vals), dtype=torch.float32, device=self.device)

        return sdf_pts, sdf_vals  # Return computed SDF points and values

    # Function to apply rotation, translation, and scaling to the SDF points
    def transform_SDF(self, rotate, translate, scale):
        rotate = axis_angle_to_quaternion(rotate)  # Convert rotation from axis-angle to quaternion
        self.sdf_points = torch.round(self.sdf_points)  # Round SDF points
        new_sdf_pts = quaternion_apply(rotate, self.sdf_points)  # Apply rotation
        new_sdf_pts += translate  # Apply translation
        self.sdf_points = new_sdf_pts  # Store transformed SDF points

    # Function to create a deep copy of the meshField object
    def clone(self):
        new_meshField = meshField()  # Create a new meshField object
        new_meshField.mesh = self.mesh.clone()  # Clone the mesh
        new_meshField.sdf_points = self.sdf_points.clone()  # Clone the SDF points
        new_meshField.sdf_values = self.sdf_values.clone()  # Clone the SDF values
        return new_meshField  # Return the cloned object

    # Function to get the dimensions of the mesh by computing its bounding box
    def get_dimensions(self):
        return (
            self.mesh.get_mesh_verts_faces(0)[0].detach().cpu().numpy().max(0) -
            self.mesh.get_mesh_verts_faces(0)[0].detach().cpu().numpy().min(0)
        )  # Compute the difference between max and min vertex coordinates to get dimensions
