from dependencies import *  # Import all functions and classes from the dependencies module

class Environment:
    def __init__(self, num: int, device=set_device()):  # Initialize the Environment class with a number and device (default set by set_device())
        self.device = device  # Store the device in the instance variable

        n = int(num ** (1 / 3))  # Compute the cube root of num and convert it to an integer
        if n%2 != 0: n = n - 1  # If n is odd, decrement it by 1 to make it even
        
        x_points = torch.linspace(-n/2, n/2, n+1)  # Create linearly spaced x coordinates from -n/2 to n/2 with n+1 points
        y_points = torch.linspace(-n/2, n/2, n+1)  # Create linearly spaced y coordinates from -n/2 to n/2 with n+1 points
        z_points = torch.linspace(-n/2, n/2, n+1)  # Create linearly spaced z coordinates from -n/2 to n/2 with n+1 points
        
        xx, yy, zz = torch.meshgrid(x_points, y_points, z_points, indexing='ij')  # Create a meshgrid for the 3D space with 'ij' indexing
        environment_points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # Stack the meshgrid tensors and reshape to a list of 3D points

        self.N = n/2  # Store half the value of n as an instance variable N (boundary limit)
        self.environment = environment_points  # Save the environment points in the instance variable
        self.intersection_value = torch.tensor(0.0, dtype=torch.float32, device=self.device)  # Initialize the intersection_value tensor on the specified device
        self.boundary_points = self.get_boundary_points(self.environment)  # Compute and store the boundary points from the environment points

    def get_boundary_points(self, points):  # Define a method to extract boundary points from a set of points
        n = self.N  # Retrieve the boundary limit stored in the instance variable
        boundary_points = torch.empty((0, 3), dtype=torch.float32, device=self.device)  # Initialize an empty tensor for boundary points on the specified device
        for point in points:  # Iterate over each point in the input tensor
            x, y, z = point  # Unpack the x, y, z coordinates of the point
            if (-n <= x <= n) and (-n <= y <= n) and (-n <= z <= n):  # Check if the point lies within the boundaries defined by n
                boundary_points = torch.cat((boundary_points, torch.tensor([[x, y, z]], device=self.device)))  # Concatenate the point to the boundary_points tensor if within bounds
        return boundary_points  # Return the tensor containing all boundary points
    

    def smooth_signed_distance_sum(self, points, nx, ny, nz):  # Define a method to compute a smooth signed distance sum for given points and boundary extents
        def smooth_abs(x):  # Define an inner function to compute a smooth absolute value
            return torch.sqrt(x ** 2 + 1e-6)  # Return the smooth absolute value with a small epsilon for numerical stability
        # Separate the x, y, z coordinates
        x, y, z = points[:, 0], points[:, 1], points[:, 2]  # Extract x, y, z components from the points tensor
        # Compute the distance for each coordinate
        dx = smooth_abs(x) - nx  # Compute the distance from the smooth absolute x to nx
        dy = smooth_abs(y) - ny  # Compute the distance from the smooth absolute y to ny
        dz = smooth_abs(z) - nz  # Compute the distance from the smooth absolute z to nz
        # Outside distance for points outside the cuboid
        outside_distance = torch.sqrt(torch.relu(dx) ** 2 + torch.relu(dy) ** 2 + torch.relu(dz) ** 2)  # Calculate the Euclidean distance for points outside the cuboid using ReLU to ensure non-negativity
        # Inside distance for points inside the cuboid
        inside_distance = -torch.min(nx - smooth_abs(x), torch.min(ny - smooth_abs(y), nz - smooth_abs(z)))  # Compute the negative minimum distance for points inside the cuboid
        # Smooth transition between inside and outside distances
        signed_distances = torch.where((dx <= 0) & (dy <= 0) & (dz <= 0), inside_distance, outside_distance)  # Choose inside_distance for points inside and outside_distance for points outside
        # Sum of all signed distances
        total_signed_distance = torch.sum(signed_distances)  # Sum up all signed distances to get a total measure
        return total_signed_distance  # Return the total signed distance sum

    def desc_environment(self):  # Define a method to describe the environment
        environment_points = self.environment.cpu().numpy()  # Move environment points to CPU and convert to a NumPy array
        print('Environment points shape:', environment_points.shape, 'Environment points length:', len(environment_points))  # Print the shape and length of the environment points array

    def get_intersections(self, mesh_fields):  # Define a method to compute intersections between mesh fields
        device = mesh_fields[0].mesh.device  # Assume all mesh fields are on the same device; retrieve the device from the first mesh
        loss = torch.tensor(0.0, device=device)  # Initialize a loss tensor on the retrieved device to accumulate intersection values
        num_meshes = len(mesh_fields)  # Get the number of mesh fields

        for i in range(num_meshes):  # Iterate over each mesh field by index i
            m1 = mesh_fields[i]  # Get the first mesh field m1
            for j in range(i + 1, num_meshes):  # Iterate over each subsequent mesh field by index j
                m2 = mesh_fields[j]  # Get the second mesh field m2
                chamfer_dist = chamfer_loss(  # Compute the weighted Chamfer loss between m1 and m2 using their sdf_points and sdf_values
                    m1.sdf_points.unsqueeze(0),  # Unsqueeze m1's sdf_points to add a batch dimension
                    m2.sdf_points.unsqueeze(0),  # Unsqueeze m2's sdf_points to add a batch dimension
                    m1.sdf_points.shape[0],  # Pass the number of points in m1
                    m2.sdf_points.shape[0],  # Pass the number of points in m2
                )
                loss += chamfer_dist  # Accumulate the chamfer loss into the total loss

        self.intersection_value = loss  # Store the computed intersection loss in the instance variable
