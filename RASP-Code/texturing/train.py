import warnings  # Import warnings module to control warning messages
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch3d")  # Suppress specific UserWarnings from pytorch3d
import sys  # Import sys module for system-specific parameters and functions
import argparse  # Import argparse module for parsing command-line arguments
import importlib  # Import importlib module for dynamic module import
import os  # Import os module for file and directory operations
import shutil  # Import shutil module for high-level file operations (copy, remove directories)
from dependencies import *  # Import everything from the dependencies module
from meshField import meshField  # Import meshField class from meshField module
from environment import Environment  # Import Environment class from environment module
from visualizer import Visualizer  # Import Visualizer class from visualizer module
from rendering import Rendering  # Import Rendering class from rendering module
from losses import *  # Import everything from the losses module
from PIL import Image  # Import Image class from PIL for image processing
import pdb  # Import Python debugger (pdb) for debugging

parser = argparse.ArgumentParser(description='parse command-line arguments.')  # Create an argument parser with a description
parser.add_argument('--config', type=str, help='config file')  # Add argument for configuration file path
parser.add_argument('--save_mode', type=str, help='save mode [new/override]')  # Add argument for save mode (new or override)
parser.add_argument('--device', type=str, help='device [cpu/id]')  # Add argument for device specification (cpu or GPU id)

ARG_config = parser.parse_args().config  # Parse and retrieve the config file argument
ARG_save_mode = parser.parse_args().save_mode  # Parse and retrieve the save_mode argument
hyperparameters = importlib.import_module(ARG_config)  # Dynamically import the module specified by ARG_config
ARG_device = parser.parse_args().device  # Parse and retrieve the device argument

# validate the arguments
assert ARG_save_mode in ['new', 'override'], "Invalid save mode. Choose from [new, override]"  # Validate save_mode input
assert ARG_device.isdigit() or ARG_device == 'cpu', "Invalid device. Choose from [cpu, id]"  # Validate device input

# folder setup for saving the results
if not os.path.exists('results'): os.makedirs('results')  # Create 'results' folder if it does not exist
_folder = os.listdir('results')  # List contents of 'results' folder
_num = 0  # Initialize variable to track the highest run number
for i in _folder:  # Iterate over each item in the results folder
    _n = i.split('_')[-1]  # Extract the numeric suffix from the folder name
    if _n.isdigit(): _num = max(_num, int(_n))  # Update _num if current numeric suffix is higher

if ARG_save_mode == 'new':  # Check if new run mode is selected
    RUNS_FOLDER = 'results/runs_' + str(_num+1).zfill(4)  # Create new run folder with incremented run number
elif ARG_save_mode == 'override' and _num > 0:  # Check if override mode is selected and run exists
    RUNS_FOLDER = 'results/runs_' + str(_num).zfill(4)  # Use the latest run folder for override

if os.path.exists(RUNS_FOLDER): shutil.rmtree(RUNS_FOLDER), os.makedirs(RUNS_FOLDER)  # If folder exists, remove and recreate it
else: os.makedirs(RUNS_FOLDER)  # Else, create the folder

os.mkdir(os.path.join(RUNS_FOLDER, 'silhouettes'))  # Create a subfolder for silhouettes in the run folder
os.mkdir(os.path.join(RUNS_FOLDER, 'metadata'))  # Create a subfolder for metadata in the run folder
os.mkdir(os.path.join(RUNS_FOLDER, 'report'))  # Create a subfolder for report in the run folder
os.mkdir(os.path.join(RUNS_FOLDER, 'report/structures'))  # Create a subfolder for mesh structures in the report folder
os.mkdir(os.path.join(RUNS_FOLDER, 'report/rendered'))  # Create a subfolder for rendered images in the report folder

# copy the hyperparameters file to the results folder
hyperparam_file_path = os.path.join('config', ARG_config.split('.')[-1] + '.py')  # Build path to the hyperparameters file in the config folder
shutil.copy(hyperparam_file_path, RUNS_FOLDER + '/metadata/params.txt')  # Copy the hyperparameters file to metadata/params.txt in the run folder

# set the device
device = set_device(int(ARG_device))  # Set the computation device based on the provided device argument
# get the visualizer
visualizer = Visualizer()  # Instantiate the Visualizer object

# create the environment
environment = Environment(hyperparameters.ENVIRONMENT_SIZE, device=device)  # Create the Environment object using hyperparameters and device

# create the renderers
renderers = []  # Initialize an empty list for renderers
assert len(hyperparameters.CAMERA_TYPES) == hyperparameters.CAMERA_COUNT, "Number of camera transforms should be equal to the number of cameras"  # Ensure camera types count equals camera count
for i in range(hyperparameters.CAMERA_COUNT):  # Loop over each camera
        camera_transform = hyperparameters.CAMERA_TRANSFORMS[i]  # Get the camera transform for the current camera
        renderer = Rendering(camera_transform, 
                             camera_type=hyperparameters.CAMERA_TYPES[i], 
                             image_size=hyperparameters.IMAGE_SIZE, 
                             sigma=1e-4, 
                             faces_per_pixel=hyperparameters.FACES_PER_PIXEL, 
                             environment_factor=environment.N,
                             device=device)  # Create a Rendering object with specified parameters
        renderers.append(renderer)  # Append the renderer to the renderers list

# create the target silhouettes
target_silhouettes = []  # Initialize an empty list for target silhouettes

for i in range(hyperparameters.CAMERA_COUNT):  # Loop over each camera
    # open target image in grayscale
    trg_image = Image.open(hyperparameters.TARGET_IMAGES[i])  # Open the target image file
    trg_image = trg_image.resize((hyperparameters.IMAGE_SIZE, hyperparameters.IMAGE_SIZE))  # Resize image to specified dimensions
    trg_image = np.array(trg_image)  # Convert the image to a NumPy array
    trg_image = trg_image[:, :, :3]  # Keep only the first 3 channels (RGB)
    trg_image = torch.from_numpy(trg_image).float() / 255.0  # Convert the array to a float tensor and normalize pixel values to [0,1]
    trg_image = trg_image.to(device)  # Move the tensor to the specified device
    target_silhouettes.append(trg_image)  # Append the processed target silhouette to the list

visualizer.visualize_silhouette_multiple(target_silhouettes, name=RUNS_FOLDER + '/metadata/target_silhouettes')  # Visualize and save the target silhouettes

# create the source objects
source_objects = hyperparameters.SOURCE_OBJECTS  # Get the list of source object file paths from hyperparameters
__sourceMeshFields__ = []  # Initialize an empty list for meshField objects representing source objects
for idx, obj_path in enumerate(source_objects):  # Loop over each source object with its index
    vertices, faces, _ = load_obj(obj_path)  # Load the object file and extract vertices, faces, and additional data
    mesh = meshField(device=device)  # Create a meshField object with the specified device

    mesh.populate_Mesh(vertices, faces, scaling_factor=hyperparameters.SOURCE_SCALES[idx], environment_factor=environment.N)  # Populate the mesh with vertices, faces, scaling, and environment factor
    
    __sourceMeshFields__.append(mesh)  # Append the processed mesh to the source mesh fields list

combined_source_mesh = join_meshes_as_scene([i.mesh for i in __sourceMeshFields__])  # Combine all source meshes into a single mesh scene
sourceMeshField = meshField(device=device)  # Create a new meshField object for the combined source mesh
sourceMeshField.mesh = combined_source_mesh  # Set the combined mesh to the new meshField object
__sourceMeshFields__ = [sourceMeshField]  # Replace the list with the single combined source mesh

source_silhouettes = []  # Initialize an empty list for source silhouettes
for i in range(hyperparameters.CAMERA_COUNT):  # Loop over each camera
    renderer = renderers[i]  # Get the renderer corresponding to the current camera
    silhouette = renderer.project_silhouette([mesh.mesh for mesh in __sourceMeshFields__])  # Project the silhouette for the source mesh
    source_silhouettes.append(silhouette)  # Append the computed silhouette to the list
    visualizer.visualize_3d([i.mesh for i in __sourceMeshFields__], name=RUNS_FOLDER + '/metadata/source_objects_' + str(i))  # Visualize and save the source objects in 3D for each camera

visualizer.visualize_silhouette_multiple(source_silhouettes, name=RUNS_FOLDER + '/metadata/source_silhouettes')  # Visualize and save multiple source silhouettes

# setup weights for training
losses = {  
            "silhouette": {"weight": hyperparameters.WEIGHT_SILHOUETTE, "values": []},  # Setup silhouette loss weight and initialize list for loss values
         }

# setup the transformations tensors
transformations = []  # Initialize an empty list for transformation tensors (colors in this case)
num_verts = __sourceMeshFields__[0].mesh.verts_list()[0].shape[0]  # Get the number of vertices in the source mesh
for j in range(num_verts):  # Loop over each vertex
    col = torch.tensor([1, 1, 1], dtype=torch.float32, device=device, requires_grad=True)  # Create a color tensor for the vertex with gradients enabled
    # col = torch.tensor([0, 0, 0], dtype=torch.float32, device=device, requires_grad=True)  (Commented out alternative color)
    transformations.append(col)  # Append the color tensor to the transformations list

# setup the optimizer
optimizer = torch.optim.Adam(transformations, lr=hyperparameters.LEARNING_RATE)  # Initialize the Adam optimizer with the color transformation tensors and learning rate
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)  # Setup a learning rate scheduler to decay the learning rate every 300 steps

# training loop
iterations = tqdm(range(hyperparameters.NUM_EPOCHS))  # Create a progress bar for the number of training epochs

__deformedMeshFields__ = []  # Initialize an empty list to store deformed (updated) mesh fields per iteration
__iteration__ = 0  # Initialize iteration counter

try:
    for __iteration__ in iterations:  # Loop over each epoch
        optimizer.zero_grad()  # Zero out gradients for the optimizer

        __deformedMeshFields__ = []  # Reset the list of deformed mesh fields for this iteration
        src_meshField = __sourceMeshFields__[0].clone()  # Clone the source mesh field to apply transformations

        C = transformations  # Assign the list of color transformation tensors to C
        src_meshField.colorize_mesh(color=C)  # Update the mesh's color using the colorize_mesh method with C

        __deformedMeshFields__.append(src_meshField)  # Append the updated mesh field to the deformed mesh fields list

        loss = {k: torch.tensor(0.0, device=device) for k in losses}  # Initialize a loss dictionary with zero values for each loss type

        for renderer, __targetSilhouette__ in zip(renderers, target_silhouettes):  # Loop over each renderer and corresponding target silhouette
            loss["silhouette"] += Losses.get_silhouette_loss(renderer, __deformedMeshFields__, __targetSilhouette__)  # Accumulate the silhouette loss for the current iteration

        net_loss = torch.tensor(0.0, device=device)  # Initialize the net loss to zero
        for k in losses:  # Loop over each loss type
            net_loss += losses[k]["weight"] * loss[k]  # Compute weighted loss and add to net loss
            losses[k]["values"].append(losses[k]["weight"] * loss[k].item())  # Append the weighted loss value to its tracking list
        
        iterations.set_description("Texture: %.3f" % (losses["silhouette"]["values"][-1]))  # Update the progress bar with the current silhouette loss

        # backpropagate the loss
        net_loss.backward()  # Perform backpropagation to compute gradients
        optimizer.step()  # Update the transformation tensors using the optimizer
        scheduler.step()  # Update the learning rate as per the scheduler

        # visualize the results
        if hyperparameters.PROGRESSIVE_RESULTS:  # Check if progressive visualization is enabled
            _silhouettes = []  # Initialize a temporary list for silhouettes
            for i in range(hyperparameters.CAMERA_COUNT):  # Loop over each camera
                renderer = renderers[i]  # Get the renderer for the current camera
                silhouette = renderer.project_silhouette([mesh.mesh for mesh in __deformedMeshFields__])  # Compute the silhouette for the deformed mesh
                silhouette = silhouette.detach().cpu()  # Detach the tensor and move to CPU
                _silhouettes.append(silhouette)  # Append the silhouette to the temporary list
            visualizer.visualize_silhouette_multiple(_silhouettes, name=RUNS_FOLDER + '/silhouettes')  # Visualize and save the progressive silhouettes
        
        if (__iteration__ + 1) % hyperparameters.PLOT_PERIOD == 0:  # Check if the current iteration is a multiple of the plot period
            _silhouettes = []  # Initialize a new list for silhouettes at this plotting interval
            for i in range(hyperparameters.CAMERA_COUNT):  # Loop over each camera
                renderer = renderers[i]  # Get the renderer for the current camera
                silhouette = renderer.project_silhouette([mesh.mesh for mesh in __deformedMeshFields__])  # Project the silhouette for the deformed mesh
                silhouette = silhouette.detach().cpu()  # Detach the silhouette tensor and move to CPU
                _silhouettes.append(silhouette)  # Append the silhouette to the list
                visualizer.visualize_silhouette_multiple(_silhouettes, name=RUNS_FOLDER + '/silhouettes/image' + '_' + str(__iteration__ + 1).zfill(6))  # Visualize and save the silhouette image with iteration number

            # visualizer.visualize_3d([i.mesh for i in __deformedMeshFields__], name=RUNS_FOLDER+'/report/structures/structure_' + str(__iteration__+1))  (Commented out 3D visualization)

except KeyboardInterrupt:
    print("Training interrupted")  # Print a message if the training loop is interrupted

finally:
    # plot the losses
    visualizer.visualize_curves([
        {"values": losses["silhouette"]["values"], "name": "silhouette"},  # Prepare the silhouette loss curve data
    ], name=RUNS_FOLDER + '/report/loss')  # Visualize and save the loss curves

    # save the transformations in a text file
    save_file = os.path.join(RUNS_FOLDER, 'report', 'structures', 'colors.csv')  # Define the path to save color transformation values
    with open(save_file, 'w') as f:  # Open the file for writing
        for t in range(0, len(transformations)):  # Loop over each transformation tensor
            f.write(','.join([str(transformations[t][0].item()), str(transformations[t][1].item()), str(transformations[t][2].item())]) + '\n')  # Write the RGB values separated by commas

    # save the losses in a csv file
    save_file = os.path.join(RUNS_FOLDER, 'report', 'losses.csv')  # Define the path to save loss values
    header = 'Texture\n'  # Define the CSV header
    with open(save_file, 'w') as f:  # Open the file for writing
        f.write(header)  # Write the header to the file
        for i in range(__iteration__):  # Loop over each iteration
            f.write(','.join([str(losses["silhouette"]["values"][i]), '\n']))  # Write the loss value for each iteration

    # save the parameters in a text file
    save_file = os.path.join(RUNS_FOLDER, 'report', 'parameters.txt')  # Define the path to save parameters
    for t in transformations:  # Loop over each transformation tensor
        with open(save_file, 'a') as f:  # Open the file in append mode
            f.write(','.join([str(t[0].item()), str(t[1].item()), str(t[2].item())]) + '\n')  # Append the RGB values to the file

    visualizer.visualize_3d([i.mesh for i in __deformedMeshFields__], name=RUNS_FOLDER + '/report/structures/final')  # Visualize the final deformed mesh in 3D and save the result
    visualizer.save_obj([i.mesh for i in __deformedMeshFields__], name=RUNS_FOLDER + '/report/structures/final')  # Save the final deformed mesh as an object file
    visualizer.render_from_multiple_directions(renderers[0], __deformedMeshFields__[0].mesh, distance=environment.N * 2, save_dir=RUNS_FOLDER + '/report/rendered')  # Render the final mesh from multiple directions and save images
    visualizer.save_gif(RUNS_FOLDER + '/report/rendered', RUNS_FOLDER + '/structure.gif')  # Create a GIF from the rendered images and save it
    visualizer.save_gif(RUNS_FOLDER + '/silhouettes/', RUNS_FOLDER + '/progressive.gif')  # Create a GIF from the silhouette images and save it
