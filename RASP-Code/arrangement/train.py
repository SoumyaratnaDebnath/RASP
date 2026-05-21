import warnings  # Import warnings module to manage warning messages
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch3d")  # Suppress specific UserWarnings from pytorch3d
import sys  # Import sys module for system-specific parameters and functions
import argparse  # Import argparse module for parsing command-line arguments
import importlib  # Import importlib for dynamic module import using strings
import os  # Import os module for operating system-related functions
import shutil  # Import shutil for high-level file operations (e.g., copying, removing directories)
from dependencies import *  # Import all necessary dependencies from the dependencies module
from meshField import meshField  # Import the meshField class from the meshField module
from environment import Environment  # Import the Environment class from the environment module
from visualizer import Visualizer  # Import the Visualizer class from the visualizer module
from rendering import Rendering  # Import the Rendering class from the rendering module
from losses import *  # Import all loss functions and classes from the losses module
from PIL import Image  # Import Image class from PIL for image processing

parser = argparse.ArgumentParser(description='parse command-line arguments.')  # Create an argument parser with a description
parser.add_argument('--config', type=str, help='config file')  # Add a command-line argument for the config file
parser.add_argument('--save_mode', type=str, help='save mode [new/override]')  # Add an argument to specify the save mode (new or override)
parser.add_argument('--device', type=str, help='device [cpu/id]')  # Add an argument to specify the computation device (cpu or GPU id)

ARG_config = parser.parse_args().config  # Parse and store the config file argument
ARG_save_mode = parser.parse_args().save_mode  # Parse and store the save_mode argument
hyperparameters = importlib.import_module(ARG_config)  # Dynamically import the configuration module specified by the user
ARG_device = parser.parse_args().device  # Parse and store the device argument

# validate the arguments
assert ARG_save_mode in ['new', 'override'], "Invalid save mode. Choose from [new, override]"  # Ensure save_mode is valid
assert ARG_device.isdigit() or ARG_device == 'cpu', "Invalid device. Choose from [cpu, id]"  # Ensure device is either a digit (GPU id) or 'cpu'

# folder setup for saving the results
if not os.path.exists('results'): os.makedirs('results')  # Create the 'results' folder if it doesn't exist
_folder = os.listdir('results')  # List all items in the 'results' folder
_num = 0  # Initialize a variable to store the highest run number
for i in _folder:  # Loop through each item in the results folder
    _n = i.split('_')[-1]  # Extract the numeric part from the folder name (after the last underscore)
    if _n.isdigit(): _num = max(_num, int(_n))  # Update _num if a higher numeric value is found

if ARG_save_mode == 'new':  # If the save mode is new
    RUNS_FOLDER = 'results/runs_' + str(_num+1).zfill(4)  # Create a new run folder with incremented number (zero-padded)
elif ARG_save_mode == 'override' and _num > 0:  # If save mode is override and there is an existing run folder
    RUNS_FOLDER = 'results/runs_' + str(_num).zfill(4)  # Use the latest run folder

if os.path.exists(RUNS_FOLDER): shutil.rmtree(RUNS_FOLDER), os.makedirs(RUNS_FOLDER)  # If RUNS_FOLDER exists, remove it then recreate it
else: os.makedirs(RUNS_FOLDER)  # Otherwise, create RUNS_FOLDER

os.mkdir(os.path.join(RUNS_FOLDER, 'silhouettes'))  # Create a subdirectory for silhouettes in the run folder
os.mkdir(os.path.join(RUNS_FOLDER, 'metadata'))  # Create a subdirectory for metadata in the run folder
os.mkdir(os.path.join(RUNS_FOLDER, 'report'))  # Create a subdirectory for reports in the run folder
os.mkdir(os.path.join(RUNS_FOLDER, 'report/structures'))  # Create a subdirectory for mesh structures inside the report folder

# copy the hyperparameters file to the results folder
hyperparam_file_path = os.path.join('config', ARG_config.split('.')[-1] + '.py')  # Build the path to the hyperparameters file in the config folder
shutil.copy(hyperparam_file_path, RUNS_FOLDER + '/metadata/params.txt')  # Copy the hyperparameters file to the metadata folder as params.txt

# set the device
device = set_device(int(ARG_device))  # Set the computation device based on the provided device argument
# get the visualizer
visualizer = Visualizer()  # Instantiate the Visualizer object

# create the environment
environment = Environment(hyperparameters.ENVIRONMENT_SIZE, device=device)  # Create the Environment object with the specified size and device

# create the renderers
renderers = []  # Initialize an empty list to store renderer objects
assert len(hyperparameters.CAMERA_TYPES) == hyperparameters.CAMERA_COUNT, "Number of camera transforms should be equal to the number of cameras"  # Check that the number of camera types matches the number of cameras
for i in range(hyperparameters.CAMERA_COUNT):  # Loop over the number of cameras
        camera_transform = hyperparameters.CAMERA_TRANSFORMS[i]  # Retrieve the camera transform for the current index
        renderer = Rendering(camera_transform, 
                             camera_type=hyperparameters.CAMERA_TYPES[i], 
                             image_size=hyperparameters.IMAGE_SIZE, 
                             sigma=1e-4, 
                             faces_per_pixel=hyperparameters.FACES_PER_PIXEL, 
                             environment_factor=environment.N,
                             device=device,)  # Instantiate a Rendering object with specified parameters
        renderers.append(renderer)  # Append the renderer to the renderers list

# create the target silhouettes
target_silhouettes = []  # Initialize an empty list for target silhouettes
        
for i in range(hyperparameters.CAMERA_COUNT):  # Loop over each camera
    # open target image in grayscale
    trg_image = Image.open(hyperparameters.TARGET_IMAGES[i]).convert('L')  # Open the target image and convert it to grayscale ('L' mode)
    trg_image = trg_image.resize((hyperparameters.IMAGE_SIZE, hyperparameters.IMAGE_SIZE))  # Resize the image to the specified dimensions
    trg_image = np.array(trg_image)  # Convert the image to a NumPy array
    trg_image = np.where(trg_image > 250, 0, 1)  # Apply a threshold inversion: pixels > 250 become 0, else 1
    trg_image = torch.from_numpy(trg_image).float()  # Convert the NumPy array to a float tensor
    trg_image = trg_image.to(device)  # Move the tensor to the specified device
    target_silhouettes.append(trg_image)  # Append the processed target silhouette to the list

# save the target silhouettes
visualizer.visualize_image_silhouette_multiple(target_silhouettes, name=RUNS_FOLDER + '/metadata/target_silhouettes')  # Visualize and save the target silhouettes

# create the source objects
source_objects = hyperparameters.SOURCE_OBJECTS  # Retrieve the list of source object file paths from hyperparameters
__sourceMeshFields__ = []  # Initialize an empty list to store meshField objects for source objects
for idx, obj_path in enumerate(source_objects):  # Loop over each source object with its index
    vertices, faces, _ = load_obj(obj_path)  # Load the object file to get vertices, faces, and additional data
    mesh = meshField(device=device)  # Create a meshField object with the specified device

    mesh.populate_Mesh(vertices, faces.verts_idx, scaling_factor=hyperparameters.SOURCE_SCALES[idx], )  # Populate the mesh with vertices and face indices, applying a scaling factor
    
    mesh.populate_SDF(environment.environment)  # Compute and populate the Signed Distance Function (SDF) for the mesh based on the environment points
    
    # perform random transformations
    if hyperparameters.RANDOM_INITIAL_TRANSFORMATIONS:  # Check if random initial transformations are enabled
        random_translation = (torch.rand(3, device=device) - 0.5) * environment.N  # Generate a random translation vector scaled by environment size
        random_rotation = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)  # Create a tensor for rotation (here set to zero)
        mesh.transform_mesh(rotate=random_rotation, translate=random_translation)  # Apply the random transformation to the mesh
        mesh.transform_SDF(rotate=random_rotation, translate=random_translation, scale=environment.N)  # Apply the same transformation to the mesh's SDF

    __sourceMeshFields__.append(mesh)  # Append the processed mesh to the list of source mesh fields

source_silhouettes = []  # Initialize an empty list for source silhouettes
for i in range(hyperparameters.CAMERA_COUNT):  # Loop over each camera
    renderer = renderers[i]  # Retrieve the renderer corresponding to the current camera
    silhouette = renderer.project_silhouette([mesh.mesh for mesh in __sourceMeshFields__])  # Project silhouettes for all source meshes
    source_silhouettes.append(silhouette)  # Append the computed silhouette to the list
visualizer.visualize_3d([i.mesh for i in __sourceMeshFields__], name=RUNS_FOLDER + '/metadata/source_objects')  # Visualize the source objects in 3D and save the visualization
visualizer.visualize_SDF([mesh.sdf_points for mesh in __sourceMeshFields__],
                        [mesh.sdf_values for mesh in __sourceMeshFields__],
                        environment.environment, name=RUNS_FOLDER + '/metadata/source_SDF')  # Visualize the SDF of source objects and save the result

visualizer.visualize_silhouette_multiple(source_silhouettes, name=RUNS_FOLDER + '/metadata/source_silhouettes')  # Visualize multiple source silhouettes and save the result
environment.get_intersections(__sourceMeshFields__)  # Compute intersections between the environment and the source objects
visualizer.visualize_environment(environment, name=RUNS_FOLDER + '/metadata/environment_source')  # Visualize the environment along with the objects

# setup weights for training
losses = {  
            "silhouette": {"weight": hyperparameters.WEIGHT_SILHOUETTE, "values": []},  # Set weight and initialize list for silhouette loss values
            "intersection": {"weight": hyperparameters.WEIGHT_INTERSECTION, "values": []},   # Set weight and initialize list for intersection loss values
            "bounding": {"weight": hyperparameters.WEIGHT_BOUNDING, "values": []},  # Set weight and initialize list for bounding loss values
         }

# setup the transformations tensors
transformations = []  # Initialize an empty list to store transformation tensors
for i in range(hyperparameters.NUM_OBJECTS):  # Loop over each object to be transformed
    transformations.append(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device, requires_grad=True))  # Append a tensor for rotation (requires gradient)
    transformations.append(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device, requires_grad=True))  # Append a tensor for translation (requires gradient)

optimizer = torch.optim.Adam(transformations, lr=hyperparameters.LEARNING_RATE)  # Initialize the Adam optimizer with the transformation tensors
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1)  # Set up a learning rate scheduler to decay the learning rate every 400 steps

# training loop
iterations = tqdm(range(hyperparameters.NUM_EPOCHS))  # Create a progress bar for the number of training epochs

__deformedMeshFields__ = []  # Initialize an empty list to store deformed (transformed) mesh fields for the current iteration
__iteration__ = 0  # Initialize iteration counter

try:
    for __iteration__ in iterations:  # Loop over each training epoch
        optimizer.zero_grad()  # Zero out gradients before computing the new gradients

        __deformedMeshFields__ = []  # Reset the list of deformed mesh fields for the current epoch
        # transform the source objects
        for instance in range(hyperparameters.NUM_OBJECTS):  # Loop over each source object
            src_meshField = __sourceMeshFields__[instance].clone()  # Clone the source mesh to avoid in-place modifications
            # upscaling the rotations and translations to force the params between -1 and 1
            R = (transformations[2 * instance]) * 180.0  # Scale the rotation parameters (assumed to be in a normalized range)
            T = transformations[2 * instance + 1] * environment.N  # Scale the translation parameters by the environment size
            src_meshField.transform_mesh(rotate=R, translate=T)  # Apply the transformation (rotation and translation) to the mesh
            src_meshField.transform_SDF(rotate=R, translate=T, scale=environment.N)  # Apply the same transformation to the mesh's SDF
            __deformedMeshFields__.append(src_meshField)  # Append the transformed mesh to the deformed mesh fields list

        # setup the losses
        loss = {k: torch.tensor(0.0, device=device) for k in losses}  # Initialize a dictionary of loss values on the specified device

        # compute the losses
        for renderer, __targetSilhouette__ in zip(renderers, target_silhouettes):  # Loop over each renderer and its corresponding target silhouette
            loss["silhouette"] += Losses.get_silhouette_loss(renderer, __deformedMeshFields__, __targetSilhouette__)  # Accumulate the silhouette loss
        loss["intersection"] = Losses.get_intersection_loss(environment, __deformedMeshFields__)  # Compute the intersection loss
        loss["bounding"] = torch.tensor(0.0, device=device)  # Set the bounding loss to zero (bypassing its computation)

        # compute the net loss
        net_loss = torch.tensor(0.0, device=device)  # Initialize the net loss as a zero tensor
        for k in losses:  # Loop over each loss type
            net_loss += losses[k]["weight"] * loss[k]  # Add the weighted loss to the net loss
            losses[k]["values"].append(losses[k]["weight"] * loss[k].item())  # Record the weighted loss value for later visualization

        iterations.set_description("Silhouette: %.3f, Intersection: %.3f, Bounding: %.3f" % (losses["silhouette"]["values"][-1], losses["intersection"]["values"][-1], losses["bounding"]["values"][-1]))  # Update the progress bar with the current loss values

        # backpropagate the loss
        net_loss.backward()  # Compute gradients via backpropagation
        optimizer.step()  # Update the transformation tensors based on the gradients
        scheduler.step()  # Update the learning rate according to the scheduler

        # visualize the results
        if hyperparameters.PROGRESSIVE_RESULTS:  # Check if progressive result visualization is enabled
            _silhouettes = []  # Initialize a temporary list for silhouettes
            for i in range(hyperparameters.CAMERA_COUNT):  # Loop over each camera
                renderer = renderers[i]  # Retrieve the renderer for the current camera
                silhouette = renderer.project_silhouette([mesh.mesh for mesh in __deformedMeshFields__])  # Compute the silhouette for the deformed meshes
                silhouette = silhouette.detach().cpu()  # Detach the silhouette tensor and move it to CPU
                _silhouettes.append(silhouette)  # Append the silhouette to the temporary list
            visualizer.visualize_silhouette_multiple(_silhouettes, name=RUNS_FOLDER + '/silhouettes')  # Visualize and save the silhouettes

        if (__iteration__ + 1) % hyperparameters.PLOT_PERIOD == 0:  # Check if it's time to save periodic visualizations
            _silhouettes = []  # Initialize a new list for silhouettes
            for i in range(hyperparameters.CAMERA_COUNT):  # Loop over each camera
                renderer = renderers[i]  # Get the renderer for the current camera
                silhouette = renderer.project_silhouette([mesh.mesh for mesh in __deformedMeshFields__])  # Project the silhouette for deformed meshes
                silhouette = silhouette.detach().cpu()  # Detach and move the silhouette to CPU
                _silhouettes.append(silhouette)  # Append to the silhouettes list
                visualizer.visualize_silhouette_multiple(_silhouettes, name=RUNS_FOLDER + '/silhouettes/image' + '_' + str(__iteration__ + 1).zfill(6))  # Visualize and save the silhouette image for the current iteration
                visualizer.save_obj([i.mesh for i in __deformedMeshFields__], name=RUNS_FOLDER + '/report/structures/structure_' + str(__iteration__ + 1))  # Save the current deformed mesh structures

            # visualizer.visualize_3d([i.mesh for i in __deformedMeshFields__], name=RUNS_FOLDER+'/structures/structure_' + str(__iteration__+1))  # (Commented out alternative 3D visualization)

            if hyperparameters.VISUALIZE:  # If detailed visualization is enabled
                sdf_points = [mesh.sdf_points.detach().cpu() for mesh in __deformedMeshFields__]  # Collect SDF points from deformed meshes and move to CPU
                sdf_values = [mesh.sdf_values.detach().cpu() for mesh in __deformedMeshFields__]  # Collect SDF values from deformed meshes and move to CPU
                visualizer.visualize_SDF(sdf_points, sdf_values, environment.environment, name=RUNS_FOLDER + '/report/SDF_' + str(__iteration__ + 1))  # Visualize and save the SDF visualization
                environment.get_intersections(__deformedMeshFields__)  # Compute intersections with the deformed meshes
                visualizer.visualize_environment(environment, name=RUNS_FOLDER + '/report/intersection_' + str(__iteration__ + 1))  # Visualize and save the environment intersections

except KeyboardInterrupt:
    print("Training interrupted")  # Print a message if training is interrupted manually

finally:
    # visualize the final results
    if not hyperparameters.VISUALIZE:  # If visualization is not enabled during training
        sdf_points = [mesh.sdf_points.detach().cpu() for mesh in __deformedMeshFields__]  # Gather final SDF points from deformed meshes
        sdf_values = [mesh.sdf_values.detach().cpu() for mesh in __deformedMeshFields__]  # Gather final SDF values from deformed meshes
        visualizer.visualize_SDF(sdf_points, sdf_values, environment.environment, name=RUNS_FOLDER + '/report/SDF_' + str(__iteration__ + 1))  # Visualize and save the final SDF
        environment.get_intersections(__deformedMeshFields__)  # Compute final intersections for deformed meshes
        visualizer.visualize_environment(environment, name=RUNS_FOLDER + '/report/intersection_' + str(__iteration__ + 1))  # Visualize and save the final environment intersections
        
    # plot the losses
    visualizer.visualize_curves([
        {"values": losses["silhouette"]["values"], "name": "silhouette"},  # Prepare silhouette loss values for plotting
        {"values": losses["intersection"]["values"], "name": "intersection"},  # Prepare intersection loss values for plotting
        {"values": losses["bounding"]["values"], "name": "bounding"},  # Prepare bounding loss values for plotting
    ], name=RUNS_FOLDER + '/report/loss')  # Visualize and save the loss curves

    # save the transformations in a text file
    save_file = os.path.join(RUNS_FOLDER, 'report', 'transformations.csv')  # Define the path to save transformation parameters
    header = 'Rotation_X, Rotation_Y, Rotation_Z, Translation_X, Translation_Y, Translation_Z\n'  # Define the CSV header
    with open(save_file, 'w') as f:  # Open the CSV file in write mode
        f.write(header)  # Write the header to the file
        for t in range(0, len(transformations), 2):  # Loop over transformation tensors in steps of 2 (rotation and translation)
            f.write(','.join([str(transformations[t][0].item()), str(transformations[t][1].item()), str(transformations[t][2].item()), 
                            str(transformations[t+1][0].item()), str(transformations[t+1][1].item()), str(transformations[t+1][2].item())]) + '\n')  # Write transformation values to CSV

    # save the losses in a csv file
    save_file = os.path.join(RUNS_FOLDER, 'report', 'losses.csv')  # Define the path to save loss values
    header = 'Silhouette, Intersection, Bounding\n'  # Define the CSV header for losses
    with open(save_file, 'w') as f:  # Open the file in write mode
        f.write(header)  # Write the header line
        for i in range(__iteration__):  # Loop over each iteration
            f.write(','.join([str(losses["silhouette"]["values"][i]), str(losses["intersection"]["values"][i]), str(losses["bounding"]["values"][i]), '\n']))  # Write loss values for the iteration to CSV
    
    visualizer.visualize_3d([i.mesh for i in __deformedMeshFields__], name=RUNS_FOLDER + '/report/structures/final')  # Visualize the final deformed meshes in 3D and save the result
    visualizer.save_obj([i.mesh for i in __deformedMeshFields__], name=RUNS_FOLDER + '/report/structures/final')  # Save the final deformed meshes as a combined object file
    visualizer.save_indivisual_objs([i.mesh for i in __deformedMeshFields__], name=RUNS_FOLDER + '/report/structures/individual')  # Save each deformed mesh object individually
    visualizer.save_gif(RUNS_FOLDER + '/silhouettes/', RUNS_FOLDER + '/progressive.gif')  # Create and save a GIF from the silhouette images in the silhouettes folder
