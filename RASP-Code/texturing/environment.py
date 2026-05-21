from dependencies import *  # Import all functions and classes from the dependencies module

# Define the Environment class to represent a 3D environment and compute mesh intersections
class Environment:
    def __init__(self, num: int, device=set_device()):  # Constructor takes a number (size parameter) and a computation device
        self.device = device  # Save the device for later use

        n = int(num ** (1 / 3))  # Compute cube root of 'num' to determine grid resolution
        if n % 2 != 0: n = n - 1  # Ensure 'n' is even; if odd, reduce by one

        x_points = torch.linspace(-n/2, n/2, n+1)  # Create linearly spaced points along x-axis from -n/2 to n/2
        y_points = torch.linspace(-n/2, n/2, n+1)  # Create linearly spaced points along y-axis from -n/2 to n/2
        z_points = torch.linspace(-n/2, n/2, n+1)  # Create linearly spaced points along z-axis from -n/2 to n/2

        xx, yy, zz = torch.meshgrid(x_points, y_points, z_points, indexing='ij')  # Create a 3D grid of points with 'ij' indexing
        environment_points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # Stack grid arrays and reshape to a list of 3D points

        self.N = n/2  # Set a scaling factor N as half of n (used for environment dimensions)
        self.environment = environment_points.to(self.device)  # Store the environment points on the specified device
        self.intersection_points = torch.empty((0, self.environment.shape[1]), dtype=torch.float32, device=self.device)  # Initialize an empty tensor for intersection points
        self.intersection_value = torch.tensor(0.0, dtype=torch.float32, device=self.device)  # Initialize the intersection value (e.g., count or sum) as 0

    def desc_environment(self):  # Method to describe the environment by printing its properties
        environment_points = self.environment.cpu().numpy()  # Move environment points to CPU and convert to numpy array
        print('Environment points shape:', environment_points.shape, 'Environment points length:', len(environment_points))  # Print shape and length of environment points

    def get_intersections(self, meshFields: list, mode="counting"):  # Method to compute intersections given a list of meshFields and a mode
        switch_case = {  # Define a dictionary mapping mode names to corresponding functions
            "counting": self.get_intersections_counting,
            "multiplex": self.get_intersections_multiplex,
        }

        if mode in switch_case:  # Check if provided mode is valid
            return switch_case[mode](meshFields)  # Call the corresponding function with meshFields
        else:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are: {', '.join(switch_case.keys())}")  # Raise error if mode is invalid

    def get_intersections_counting(self, meshFields: list):  # Method to compute intersections by counting mesh points present in the environment
        env_points_set = set(map(tuple, self.environment.cpu().numpy()))  # Convert environment points to a set of tuples for fast lookup

        mesh_points_all = []  # List to collect mesh points from each meshField
        mesh_values_all = []  # List to collect corresponding SDF values from each meshField

        for mesh in meshFields:  # Iterate over each meshField
            mesh.sdf_points = mesh.sdf_points.to(self.device)  # Ensure mesh SDF points are on the correct device
            mesh_points_all.append(mesh.sdf_points)  # Append mesh's SDF points to list
            mesh_values_all.append(mesh.sdf_values)  # Append mesh's SDF values to list

        # Concatenate all mesh points and values into one tensor
        all_mesh_points = torch.cat(mesh_points_all, dim=0)

        # Convert environment points to set again (redundant but ensures consistency)
        env_points_set = set(map(tuple, self.environment.cpu().numpy()))
        
        # Find unique mesh points among all collected points
        unique_mesh_points, unique_indices = torch.unique(all_mesh_points, dim=0, return_inverse=True)
        
        # Create a mask to select unique mesh points that are present in the environment
        valid_mask = torch.tensor([tuple(p) in env_points_set for p in unique_mesh_points.cpu().detach().numpy()], device=self.device)
        
        valid_mesh_points = unique_mesh_points[valid_mask]  # Filter unique mesh points using the mask

        # Initialize a tensor to count occurrences of each valid mesh point
        point_counts = torch.zeros(valid_mesh_points.size(0), dtype=torch.int32, device=self.device)
        for mesh in meshFields:  # Iterate over each meshField
            # Create a mask for mesh's SDF points that are in the environment set
            mesh_mask = torch.tensor([tuple(p) in env_points_set for p in mesh.sdf_points.cpu().detach().numpy()], device=self.device)
            mesh_points_valid = mesh.sdf_points[mesh_mask]  # Filter the mesh's SDF points using the mask
            # Find indices where valid_mesh_points equal mesh_points_valid; nonzero returns indices for matching points
            idxs = torch.nonzero((valid_mesh_points.unsqueeze(1) == mesh_points_valid.unsqueeze(0)).all(dim=2), as_tuple=False)
            point_counts[idxs[:, 0]] += 1  # Increment the count for each matching valid mesh point

        # Identify valid mesh points that are intersected by more than one mesh (count > 1)
        multi_intersection_mask = point_counts > 1
        final_intersection_points = valid_mesh_points[multi_intersection_mask]  # Filter valid mesh points with multiple intersections

        self.intersection_points = final_intersection_points  # Store the final intersection points in the object
        self.intersection_value = torch.tensor(final_intersection_points.size(0), dtype=torch.float32, device=self.device)  # Set the intersection value as the count of intersection points

    def get_intersections_multiplex(self, meshFields: list):  # Method to compute intersections with multiplexing approach
        intersection_value = 0.0  # Initialize a variable to accumulate the intersection value

        env_points_set = set(map(tuple, self.environment.cpu().numpy()))  # Convert environment points to a set of tuples

        mesh_points_all = []  # List to collect mesh points from all meshFields
        mesh_values_all = []  # List to collect mesh SDF values from all meshFields

        for mesh in meshFields:  # Iterate over each meshField
            mesh.sdf_points = mesh.sdf_points.to(self.device)  # Ensure SDF points are on the correct device
            mesh_points_all.append(mesh.sdf_points)  # Append mesh's SDF points
            mesh_values_all.append(mesh.sdf_values)  # Append mesh's SDF values

        # Concatenate all mesh points into a single tensor
        all_mesh_points = torch.cat(mesh_points_all, dim=0)

        # Convert environment points to set for fast lookup
        env_points_set = set(map(tuple, self.environment.cpu().numpy()))
        
        # Identify unique mesh points among all collected points
        unique_mesh_points, unique_indices = torch.unique(all_mesh_points, dim=0, return_inverse=True)
        
        # Create a mask to find unique mesh points present in the environment
        valid_mask = torch.tensor([tuple(p) in env_points_set for p in unique_mesh_points.cpu().detach().numpy()], device=self.device)
        
        valid_mesh_points = unique_mesh_points[valid_mask]  # Filter unique mesh points using the valid mask

        # Initialize a tensor to count occurrences of each valid mesh point
        point_counts = torch.zeros(valid_mesh_points.size(0), dtype=torch.int32, device=self.device)
        for mesh in meshFields:  # Iterate over each meshField
            # Create a mask to filter mesh SDF points that are present in the environment
            mesh_mask = torch.tensor([tuple(p) in env_points_set for p in mesh.sdf_points.cpu().detach().numpy()], device=self.device)
            mesh_points_valid = mesh.sdf_points[mesh_mask]  # Filter mesh's SDF points using the mask
            # Find indices where valid_mesh_points equal mesh_points_valid; returns indices of matching points
            idxs = torch.nonzero((valid_mesh_points.unsqueeze(1) == mesh_points_valid.unsqueeze(0)).all(dim=2), as_tuple=False)
            point_counts[idxs[:, 0]] += 1  # Increment count for each matching valid mesh point

        # Filter valid mesh points that have more than one intersection
        multi_intersection_mask = point_counts > 1
        final_intersection_points = valid_mesh_points[multi_intersection_mask]

        # For each intersection point, compute a cumulative value from each mesh's SDF values
        for point in final_intersection_points:
            value = 0.0  # Initialize value accumulator for this point
            for mesh in meshFields:  # Iterate over each meshField
                idx = torch.all(mesh.sdf_points == point, dim=1)  # Find indices where mesh SDF points equal the current point
                value += mesh.sdf_values[idx].sum().item()  # Sum the SDF values at these indices and add to value
            intersection_value += value  # Accumulate the intersection value

        self.intersection_points = final_intersection_points  # Store the intersection points
        self.intersection_value = torch.tensor(intersection_value, dtype=torch.float32, device=self.device)  # Store the cumulative intersection value

    def get_intersections_counting_proximity(self, meshFields: list, tol=2):  # Method to compute intersections using counting with proximity tolerance
        # Initialize a tensor to collect intersection counts for each environment point
        intersection_counts = torch.zeros(self.environment.size(0), dtype=torch.int32, device=self.device)

        # For each meshField, compare its SDF points with the environment points
        for mesh in meshFields:
            mesh.sdf_points = mesh.sdf_points.to(self.device)  # Ensure mesh SDF points are on the correct device

            # Expand the environment and mesh SDF points to compute pairwise differences
            env_expanded = self.environment.unsqueeze(1)  # Shape: (num_env_points, 1, point_dims)
            mesh_expanded = mesh.sdf_points.unsqueeze(0)  # Shape: (1, num_mesh_points, point_dims)

            # Determine which pairs of points are within the tolerance in all dimensions
            close = torch.abs(env_expanded - mesh_expanded) < tol  # Boolean tensor indicating closeness
            is_intersection = torch.any(torch.all(close, dim=2), dim=1)  # Determine if any mesh point is within tolerance for each environment point

            # Update the intersection counts by adding the boolean result (converted to int)
            intersection_counts += is_intersection.int()

        # Identify environment points that have been intersected by more than one meshField
        valid_intersections = intersection_counts > 1

        # Gather these environment points and sum their intersection counts
        self.intersection_points = self.environment[valid_intersections]
        self.intersection_points = self.intersection_points.to(self.device)
        self.intersection_value = intersection_counts[valid_intersections].float().sum().to(self.device)

    def get_intersections_multiplex_proximity(self, meshFields: list, tol=2):  # Method to compute intersections using multiplexing with proximity tolerance
        intersection_points = torch.empty((0, self.environment.shape[1]), dtype=torch.float32, device=self.device)  # Initialize empty tensor for intersection points
        intersection_value = 0.0  # Initialize cumulative intersection value

        for point in self.environment:  # Iterate over each environment point
            count = 0  # Initialize counter for how many meshFields intersect this point
            for mesh in meshFields:  # Iterate over each meshField
                # Check if the point is within tolerance of more than 2 points in the mesh's SDF points
                if torch.sum(torch.all(torch.abs(mesh.sdf_points - point) < tol, dim=1)) > 2:
                    count += 1  # Increment count if condition is met
                if count > 1:  # If point is intersected by more than one meshField
                    intersection_points = torch.cat((intersection_points, point.unsqueeze(0)), dim=0)  # Append the point to intersection_points
                    value = 0.0  # Initialize value accumulator for this point
                    for m in meshFields:  # Iterate over each meshField
                        idx = torch.all(torch.abs(m.sdf_points - point) < tol, dim=1)  # Identify indices where the point is within tolerance
                        value += m.sdf_values[idx].sum().item()  # Sum SDF values for these indices and add to value
                    intersection_value += value  # Accumulate the intersection value for this point
                    break  # Break out of loop once intersection is confirmed for this point

        self.intersection_value = torch.tensor(intersection_value, dtype=torch.float32, device=self.device)  # Store the cumulative intersection value as a tensor
        self.intersection_points = intersection_points.to(self.device)  # Store the intersection points on the specified device
