from dependencies import *  # Import everything from the dependencies module
import imageio  # Import imageio for creating and saving GIFs
from pytorch3d.io import save_obj  # Import save_obj to save meshes as .obj files
from PIL import Image  # Import Image class from PIL for image processing

# Define the Visualizer class that contains various visualization and saving methods
class Visualizer:
    def visualize_silhouette(self, silhouette, name: str = 'silhouette'):  # Method to visualize a single silhouette image
        plt.imshow(silhouette.squeeze().cpu().numpy())  # Display the silhouette image using matplotlib (squeeze removes singleton dimensions)
        plt.axis('off')  # Turn off the axis for a clean image
        plt.imsave(f'{name}.png', silhouette.squeeze().cpu().numpy())  # Save the silhouette image as a PNG file with the given name

    def visualize_silhouette_multiple(self, silhouettes, name: str = 'silhouettes'):  # Method to visualize multiple silhouette images in one figure
        silhouette_imgs = [silhouette.squeeze().cpu().numpy() for silhouette in silhouettes]  # Convert each silhouette tensor to a numpy array
        n_silhouettes = len(silhouette_imgs)  # Get the number of silhouettes (fixed variable name from context: silhouette_imgs)
        height, width = silhouettes[0].shape[1], silhouettes[0].shape[0]  # Get height and width from the first silhouette tensor
        aspect_ratio = width / height  # Calculate the aspect ratio of the silhouette images
        fig_width = min(100, n_silhouettes * aspect_ratio * 4)  # Determine figure width with an upper bound of 100
        fig_height = min(100, 4)  # Set figure height with an upper bound of 4
        fig, axs = plt.subplots(1, n_silhouettes, figsize=(fig_width, fig_height))  # Create a subplot grid with 1 row and n_silhouettes columns
        if n_silhouettes == 1:  # If there is only one silhouette, wrap the single axis in a list
            axs = [axs]
        for i, silhouette in enumerate(silhouette_imgs):  # Loop over each silhouette image
            axs[i].imshow(silhouette)  # Display the silhouette image in the subplot
            axs[i].axis('off')  # Turn off axis lines for the subplot
            axs[i].title.set_text(f'Camera {i+1}')  # Set the title of the subplot to indicate the camera number
        plt.savefig(f'{name}.png')  # Save the entire figure as a PNG file with the given name
        plt.close()  # Close the figure to free up resources

    def visualize_silhouette_multiple_black(self, silhouettes, name: str = 'silhouettes', dpi: int = 300):  # Method to visualize multiple silhouettes with a black background
        silhouette_imgs = [silhouette.squeeze().cpu().numpy() for silhouette in silhouettes]  # Convert each silhouette tensor to a numpy array
        n_silhouettes = len(silhouette_imgs)  # Get the number of silhouettes
        height, width = silhouettes[0].shape[1], silhouettes[0].shape[0]  # Get the height and width from the first silhouette tensor
        aspect_ratio = width / height  # Calculate the aspect ratio
        fig_width = min(100, n_silhouettes * aspect_ratio * 4)  # Determine figure width with an upper limit
        fig_height = min(100, 4)  # Determine figure height with an upper limit
        fig, axs = plt.subplots(1, n_silhouettes, figsize=(fig_width, fig_height), dpi=dpi)  # Create subplots with specified dpi
        fig.patch.set_facecolor('black')  # Set the figure background color to black
        if n_silhouettes == 1:  # If only one subplot exists, wrap it in a list for consistency
            axs = [axs]
        for i, silhouette in enumerate(silhouette_imgs):  # Loop over each silhouette image
            axs[i].imshow(silhouette, cmap='gray', interpolation='nearest')  # Display the silhouette in grayscale with nearest interpolation
            axs[i].axis('off')  # Turn off the axis lines for the subplot
            axs[i].set_facecolor('black')  # Set the subplot background color to black
        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust spacing between subplots
        plt.savefig(f'{name}.png', bbox_inches='tight', pad_inches=0.1, facecolor='black')  # Save the figure with a tight layout and black background
        plt.close()  # Close the figure

    def visualize_3d(self, meshes, name='scene'):  # Method to visualize a 3D scene composed of meshes using Plotly
        combined_meshes = join_meshes_as_scene(meshes)  # Combine multiple meshes into a single scene mesh
        fig = plot_scene({  # Create a Plotly figure of the scene
            name: {
                "mesh": combined_meshes  # Add the combined mesh to the scene
            }
        })
        fig.write_html(f"{name}.html")  # Save the interactive 3D plot as an HTML file

    def visualize_environment(self, environment, name: str = 'environment', env_color='gray', intersection_color='red'):  # Method to visualize an environment and its intersections
        environment_points = environment.environment.cpu().detach().numpy()  # Convert environment points to a numpy array
        fig = go.Figure(data=[go.Scatter3d(  # Create a 3D scatter plot for the environment points
            x=environment_points[:, 0],  # Set x-coordinates
            y=environment_points[:, 1],  # Set y-coordinates
            z=environment_points[:, 2],  # Set z-coordinates
            mode='markers',  # Use markers for plotting points
            marker=dict(
                size=1,  # Set marker size to 1
                color=env_color,  # Set marker color to env_color (default gray)
                opacity=0.2  # Set marker opacity to 0.2
            )
        )])
        intersection_points = environment.intersection_points.cpu().detach().numpy()  # Convert intersection points to a numpy array
        fig.add_trace(go.Scatter3d(  # Add a new trace to the figure for intersection points
            x=intersection_points[:, 0],  # Set x-coordinates
            y=intersection_points[:, 1],  # Set y-coordinates
            z=intersection_points[:, 2],  # Set z-coordinates
            mode='markers',  # Use markers for plotting
            marker=dict(
                size=2,  # Set marker size to 2 for intersections
                color=intersection_color,  # Set marker color for intersections (default red)
                opacity=1  # Set full opacity for intersections
            )
        ))
        fig.update_layout(scene=dict(  # Update the layout of the 3D scene
                            xaxis_title='X',  # Label x-axis
                            yaxis_title='Y',  # Label y-axis
                            zaxis_title='Z'),  # Label z-axis
                            margin=dict(l=0, r=0, b=0, t=0))  # Remove margins around the plot
        # save the plot
        fig.write_html(name + '.html')  # Save the 3D visualization as an HTML file

    def visualize_SDF(self, sdf_pts: list, sdf_vals: list, environment_points, name: str = 'SDF'):  # Method to visualize SDF values over environment points using Plotly
        sdf_pts = [instance.cpu().numpy() for instance in sdf_pts]  # Convert each SDF points tensor to a numpy array
        sdf_vals = [instance.cpu().numpy() for instance in sdf_vals]  # Convert each SDF values tensor to a numpy array
        environment_points = environment_points.cpu().numpy()  # Convert environment points tensor to a numpy array

        fig = go.Figure(data=[go.Scatter3d(  # Create a 3D scatter plot for the environment points
            x=environment_points[:, 0],
            y=environment_points[:, 1],
            z=environment_points[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color='gray',
                opacity=0.2
            )
        )])

        if len(sdf_pts[0]) > 1:  # Check if there is more than one SDF point per instance
            for instance in range(len(sdf_pts)):  # Loop over each SDF instance
                fig.add_trace(go.Scatter3d(  # Add a new 3D scatter trace for each SDF instance
                    x=sdf_pts[instance][:, 0],
                    y=sdf_pts[instance][:, 1],
                    z=sdf_pts[instance][:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=sdf_vals[instance].flatten(),  # Use flattened SDF values as colors
                        colorscale='viridis',  # Apply the 'viridis' colorscale
                        opacity=1
                    )
                ))
                fig.update_layout(scene=dict(  # Update layout settings for the scene
                                    xaxis_title='X',
                                    yaxis_title='Y',
                                    zaxis_title='Z'),
                                    margin=dict(l=0, r=0, b=0, t=0))
    
        fig.write_html(name + '.html')  # Save the SDF visualization as an HTML file

    def visualize_curves(self, curves, name: str = 'loss'):  # Method to visualize loss curves using Plotly
        for curve in curves:  # Loop over each curve provided in the curves list
            fig = go.Figure()  # Create a new Plotly figure
            fig.add_trace(go.Scatter(y=curve["values"], mode='lines', name=curve["name"]))  # Add a line plot trace for the curve values
            fig.update_layout(xaxis_title='Iterations', yaxis_title='Loss', title='Losses')  # Update the layout with axis labels and title
            fig.write_html(name + '_' + curve["name"] + '.html')  # Save the curve visualization as an HTML file

    def save_obj(self, meshes, name='scene'):  # Method to save a combined mesh as an .obj file
        combined_meshes = join_meshes_as_scene(meshes)  # Combine multiple meshes into a single scene mesh
        verts = combined_meshes.verts_list()[0]  # Retrieve the vertices from the first mesh in the scene
        faces = combined_meshes.faces_list()[0]  # Retrieve the faces from the first mesh in the scene
        save_obj(name + '.obj', verts, faces)  # Save the combined mesh as an .obj file with the specified name

    def save_gif(self, image_folder, gif_name):  # Method to create a GIF from images in a folder
        images = []  # Initialize an empty list to store images
        for file_name in sorted(os.listdir(image_folder)):  # Iterate over sorted file names in the image folder
            if file_name.endswith(".png"):  # Check if the file is a PNG image
                file_path = os.path.join(image_folder, file_name)  # Construct the full file path
                images.append(imageio.imread(file_path))  # Read the image and append it to the list

        imageio.mimsave(gif_name, images, duration=0.5)  # Save the list of images as a GIF with a frame duration of 0.5 seconds
        print(f"Created gif {gif_name}")  # Print confirmation that the GIF has been created

    def save_indivisual_objs(self, meshes, name='scene'):  # Method to save individual mesh objects as separate .obj files
        os.makedirs(name, exist_ok=True)  # Create a directory with the given name if it does not already exist
        for i, mesh in enumerate(meshes):  # Loop over each mesh with its index
            verts = mesh.verts_list()[0]  # Get the vertices of the mesh
            faces = mesh.faces_list()[0]  # Get the faces of the mesh
            save_obj(name + '/' + str(i).zfill(10) + '.obj', verts, faces)  # Save the mesh as an .obj file with a zero-padded filename

    def render_from_multiple_directions(self, rendering, object_mesh, distance, save_dir):  # Method to render a mesh from multiple directions
        renderer = rendering.renderer  # Get the renderer from the rendering object
        lights = rendering.light  # Get the light source from the rendering object
        import matplotlib.pyplot as plt  # Import matplotlib.pyplot for plotting
        from pytorch3d.transforms import Rotate, axis_angle_to_matrix  # Import transformation functions
        import torch  # Import torch (redundant if already imported)
        import numpy as np  # Import numpy for numerical operations
        from PIL import Image  # Import Image from PIL for image processing

        # Set a constant camera view
        elev = 0  # Set the elevation angle for the camera
        azim = 0  # Set the azimuth angle for the camera
        R, T = look_at_view_transform(dist=distance, elev=elev, azim=azim)  # Compute camera rotation and translation for the specified distance and view angles
        R = R.to(object_mesh.device)  # Move the rotation matrix to the object's device
        T = T.to(object_mesh.device)  # Move the translation vector to the object's device

        # Define the rotation matrix for 1 degree around the Y-axis
        rotation_matrix = axis_angle_to_matrix(torch.tensor([0, 1, 0], device=object_mesh.device) * torch.deg2rad(torch.tensor(1.0, device=object_mesh.device)))  # Create a rotation matrix for a 1 degree rotation about the Y-axis

        # Extract the initial vertices
        vertices = object_mesh.verts_packed().clone()  # Clone the original vertices of the object_mesh
        
        # Define lemon yellow background color
        lemon_yellow = np.array([255, 255, 255], dtype=np.uint8)  # Define the background color as white (lemon yellow can be adjusted; here it's set to white)

        for angle in range(360):  # Loop over 360 degrees to rotate the object fully
            # Apply rotation to the vertices and update object_mesh
            rotated_vertices = torch.matmul(vertices, rotation_matrix)  # Rotate the vertices using the rotation matrix
            object_mesh = object_mesh.update_padded(rotated_vertices[None, :, :])  # Update the mesh with the new rotated vertices

            # Render the image with the rotated object
            images = renderer(meshes_world=object_mesh, R=R, T=T, lights=lights).to(object_mesh.device)  # Render the mesh with current camera parameters and lights
            image = images[0][..., :3]  # Extract the RGB channels from the rendered image
            image = image.clamp(0, 1)  # Clamp the image values to the range [0, 1]
            image = image.detach().cpu().numpy()  # Detach the image tensor and convert to a numpy array
            
            # Convert the rendered image to uint8
            image = (image * 255).astype(np.uint8)  # Scale the image to 0-255 and convert to unsigned 8-bit integers
            
            # Create a lemon yellow background
            background = np.full(image.shape, lemon_yellow, dtype=np.uint8)  # Create a background image filled with the lemon_yellow color
            
            # Composite the object over the lemon yellow background
            alpha = images[0][..., 3].detach().cpu().numpy()  # Extract the alpha channel from the rendered image
            alpha = np.clip(alpha, 0, 1)  # Clip alpha values to [0, 1]
            image_with_background = (image * alpha[..., None] + background * (1 - alpha[..., None])).astype(np.uint8)  # Composite the image with the background using alpha blending
            
            # Save the final image
            img = Image.fromarray(image_with_background)  # Create a PIL Image from the composited numpy array
            img.save(f"{save_dir}/view_{angle:03d}.png")  # Save the image with a filename indicating the view angle

            # Update vertices for the next rotation step
            vertices = rotated_vertices  # Set the current rotated vertices as the new vertices for subsequent rotations
