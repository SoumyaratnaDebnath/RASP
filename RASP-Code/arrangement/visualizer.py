from dependencies import *  # Import all symbols from the dependencies module
import imageio  # Import imageio library for reading and writing image files and creating GIFs
from pytorch3d.io import save_obj  # Import the save_obj function from pytorch3d.io to save mesh objects in OBJ format

class Visualizer:  # Define the Visualizer class to handle various visualization tasks
    def visualize_silhouette(self, silhouette, name:str='silhouette'):
        plt.imshow(silhouette.squeeze().cpu().numpy())  # Display the silhouette image after squeezing dimensions and moving it to CPU as a NumPy array
        plt.axis('off')  # Turn off the axis for a cleaner image display
        plt.imsave(f'{name}.png', silhouette.squeeze().cpu().numpy())  # Save the silhouette image to a PNG file with the given name

    def visualize_silhouette_multiple(self, silhouettes, name: str = 'silhouettes', dpi: int = 300):
        silhouette_imgs = [silhouette.squeeze().cpu().numpy() for silhouette in silhouettes]  # Convert each silhouette tensor to a NumPy array after squeezing and moving to CPU
        n_silhouettes = len(silhouettes)  # Count the number of silhouettes
        height, width = silhouettes[0].shape[1], silhouettes[0].shape[0]  # Get height and width from the first silhouette tensor
        aspect_ratio = width / height  # Calculate the aspect ratio of the silhouettes
        fig_width = min(100, n_silhouettes * aspect_ratio * 4)  # Determine the figure width based on number of silhouettes and aspect ratio, with an upper limit
        fig_height = min(100, 4)  # Determine the figure height with an upper limit
        fig, axs = plt.subplots(1, n_silhouettes, figsize=(fig_width, fig_height), dpi=dpi)  # Create subplots with specified DPI and figure size
        if n_silhouettes == 1:
            axs = [axs]  # Ensure axs is iterable when there is only one subplot
        for i, silhouette in enumerate(silhouette_imgs):  # Loop over each silhouette image with its index
            axs[i].imshow(silhouette, cmap='gray', interpolation='nearest')  # Display the silhouette image in grayscale with nearest interpolation
            axs[i].axis('off')  # Turn off the axis for this subplot
            axs[i].set_facecolor('black')   # Set the background color of the subplot to black
        plt.subplots_adjust(wspace=0.1, hspace=0.1)   # Adjust the spacing between subplots
        plt.savefig(f'{name}.png', bbox_inches='tight', pad_inches=0.1, facecolor='black')  # Save the figure with tight bounding box, small padding, and black facecolor
        plt.close()  # Close the plot to free up resources

    def visualize_3d(self, meshes, name='scene'):
        combined_meshes = join_meshes_as_scene(meshes)  # Combine multiple mesh objects into a single scene mesh
        fig = plot_scene({
            name: {
                "mesh": combined_meshes  # Create a dictionary with the scene name and the combined mesh for plotting
            }
        })
        fig.write_html(f"{name}.html")  # Save the interactive 3D scene as an HTML file

    def visualize_environment(self, environment, name:str='environment', env_color='gray', intersection_color='red'):
        environment_points = environment.environment.cpu().detach().numpy()  # Convert environment points to a NumPy array after detaching and moving to CPU
        fig = go.Figure(data=[go.Scatter3d(
            x=environment_points[:, 0],  # Set x-coordinates from the environment points
            y=environment_points[:, 1],  # Set y-coordinates from the environment points
            z=environment_points[:, 2],  # Set z-coordinates from the environment points
            mode='markers',  # Use markers to represent points
            marker=dict(
                size=1,  # Set the marker size
                color=env_color,  # Set the marker color for environment points
                opacity=0.2  # Set the marker opacity
            )
        )])
        fig.update_layout(scene=dict(
                            xaxis_title='X',  # Label the x-axis
                            yaxis_title='Y',  # Label the y-axis
                            zaxis_title='Z'),  # Label the z-axis
                            margin=dict(l=0, r=0, b=0, t=0))  # Remove margins from the layout
        fig.write_html(name + '.html')  # Save the 3D environment visualization as an HTML file

    def visualize_SDF(self, sdf_pts:list, sdf_vals:list, environment_points, name:str='SDF'):
        sdf_pts = [instance.cpu().numpy() for instance in sdf_pts]  # Convert each SDF points tensor to a NumPy array after moving to CPU
        sdf_vals = [instance.cpu().numpy() for instance in sdf_vals]  # Convert each SDF values tensor to a NumPy array after moving to CPU
        environment_points = environment_points.cpu().numpy()  # Convert environment points to a NumPy array after moving to CPU

        fig = go.Figure(data=[go.Scatter3d(
            x=environment_points[:, 0],  # Set x-coordinates from the environment points
            y=environment_points[:, 1],  # Set y-coordinates from the environment points
            z=environment_points[:, 2],  # Set z-coordinates from the environment points
            mode='markers',  # Use markers for display
            marker=dict(
                size=1,  # Set marker size for environment points
                color='gray',  # Set marker color for environment points
                opacity=0.2  # Set marker opacity
            )
        )])

        if len(sdf_pts[0]) > 1:  # Check if there is more than one SDF point in the first instance
            for instance in range(len(sdf_pts)):  # Loop over each SDF instance
                fig.add_trace(go.Scatter3d(
                    x=sdf_pts[instance][:, 0],  # Set x-coordinates for the current SDF instance
                    y=sdf_pts[instance][:, 1],  # Set y-coordinates for the current SDF instance
                    z=sdf_pts[instance][:, 2],  # Set z-coordinates for the current SDF instance
                    mode='markers',  # Use markers for display
                    marker=dict(
                        size=2,  # Set marker size for SDF points
                        color=sdf_vals[instance].flatten(),  # Set marker color based on flattened SDF values
                        colorscale='viridis',  # Use the Viridis colormap for SDF values
                        opacity=1  # Set full opacity for SDF points
                    )
                ))
                fig.update_layout(scene=dict(
                                    xaxis_title='X',  # Label the x-axis
                                    yaxis_title='Y',  # Label the y-axis
                                    zaxis_title='Z'),  # Label the z-axis
                                    margin=dict(l=0, r=0, b=0, t=0))  # Remove margins from the layout
    
        fig.write_html(name + '.html')  # Save the SDF visualization as an HTML file

    def visualize_curves(self, curves, name:str='loss'):
        for curve in curves:  # Loop over each curve dictionary in the curves list
            fig = go.Figure()  # Create a new Plotly figure
            fig.add_trace(go.Scatter(y=curve["values"], mode='lines', name=curve["name"]))  # Add a line trace for the curve values with the specified name
            fig.update_layout(xaxis_title='Iterations', yaxis_title='Loss', title='Losses')  # Set the layout titles for axes and the figure
            fig.write_html(name + '_' + curve["name"] + '.html')  # Save the curve visualization as an HTML file with a name based on the curve

    def save_obj(self, meshes, name='scene'):
        combined_meshes = join_meshes_as_scene(meshes)  # Combine multiple mesh objects into a single scene mesh
        verts = combined_meshes.verts_list()[0]  # Extract the vertices from the first mesh in the combined scene
        faces = combined_meshes.faces_list()[0]  # Extract the faces from the first mesh in the combined scene
        save_obj(name+'.obj', verts, faces)  # Save the combined mesh as an OBJ file with the specified name

    def save_gif(self, image_folder, gif_name):
        images = []  # Initialize an empty list to store images
        for file_name in sorted(os.listdir(image_folder)):  # Loop over the sorted list of file names in the image folder
            if file_name.endswith(".png"):  # Check if the file name ends with .png
                file_path = os.path.join(image_folder, file_name)  # Construct the full file path
                images.append(imageio.imread(file_path))  # Read the image and append it to the images list

        imageio.mimsave(gif_name, images, duration=0.5)  # Save the list of images as a GIF with the specified duration between frames
        print(f"Created gif {gif_name}")  # Print a confirmation message with the GIF name

    def save_indivisual_objs(self, meshes, name='scene'):
        os.makedirs(name, exist_ok=True)  # Create a directory with the given name if it doesn't exist
        for i, mesh in enumerate(meshes):  # Loop over each mesh with its index
            verts = mesh.verts_list()[0]   # Extract the vertices from the mesh
            faces = mesh.faces_list()[0]   # Extract the faces from the mesh
            save_obj(name+'/'+str(i).zfill(10)+'.obj', verts, faces)  # Save each mesh as an OBJ file with a zero-padded file name

    def visualize_image_silhouette_multiple(self, images, name="scene"):
        # Check if the images are CUDA tensors and move them to CPU
        images = [image.cpu().numpy() if hasattr(image, 'is_cuda') and image.is_cuda else image for image in images]  # Convert CUDA tensors to NumPy arrays if necessary
        
        # Handle single image case by ensuring axs is iterable
        if len(images) == 1:
            fig, axs = plt.subplots(1, 1, figsize=(4, 4))  # Create a single subplot for one image
            axs.imshow(images[0])  # Display the image
            axs.axis('off')  # Turn off axis display
        else:
            fig, axs = plt.subplots(1, len(images), figsize=(4*len(images), 4))  # Create subplots for multiple images with an appropriate figure size
            for i, image in enumerate(images):  # Loop over each image and its index
                axs[i].imshow(image)  # Display the image in the corresponding subplot
                axs[i].axis('off')  # Turn off axis display for the subplot
        
        # Save figure to file
        plt.savefig(f'{name}.png')  # Save the combined image figure as a PNG file with the specified name
        plt.close()  # Close the plot to free up memory
