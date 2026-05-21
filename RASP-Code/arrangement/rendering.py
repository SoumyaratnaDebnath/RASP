from dependencies import *  # Import everything from the dependencies module

class Rendering:  # Define the Rendering class to manage camera setup and rendering of meshes
    def __init__(self, camera_transform:list, camera_type:str='orthographic', image_size:int=128, sigma:float = 1e-4, faces_per_pixel:int=50, environment_factor:int=1.0, device=set_device()):
        self.device = device  # Set the computing device (CPU or GPU) for rendering
        self.camera = self.set_camera(camera_transform, camera_type=camera_type, environment_factor=environment_factor).to(self.device)  # Initialize and set the camera based on the provided parameters, then move it to the device
        self.raster_settings = self.get_raster_settings(image_size=image_size, sigma=sigma, faces_per_pixel=faces_per_pixel)  # Get the rasterization settings using provided parameters
        self.renderer = self.silhoutte_renderer()  # Use the silhouette renderer

    def set_camera(self, camera_transform:list, camera_type, environment_factor:int):
        if camera_type == 'orthographic':  # Check if the camera type is orthographic
            R, T = look_at_view_transform(dist=environment_factor, elev=camera_transform[1], azim=camera_transform[2])  # Compute rotation and translation using view transform parameters for orthographic camera
            camera = FoVOrthographicCameras(R=R, T=T, min_x=-1*environment_factor, max_x=1*environment_factor, min_y=-1*environment_factor, max_y=1*environment_factor)  # Create an orthographic camera with specified bounds
        elif camera_type == 'perspective':  # Check if the camera type is perspective
            R, T = look_at_view_transform(dist=environment_factor*2, elev=camera_transform[1], azim=camera_transform[2])  # Compute rotation and translation with a greater distance for perspective view
            camera = FoVPerspectiveCameras(R=R, T=T, fov=90, degrees=True)  # Create a perspective camera with a 90-degree field of view
        return camera  # Return the configured camera

    def get_raster_settings(self, image_size, sigma, faces_per_pixel):
        raster_settings_soft = RasterizationSettings(  # Create rasterization settings for soft rendering
            image_size=image_size,  # Set the output image size
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma,  # Compute the blur radius based on sigma for smooth silhouettes
            faces_per_pixel=faces_per_pixel,  # Set the number of faces to consider per pixel
        )
        return raster_settings_soft  # Return the rasterization settings

    def silhoutte_renderer(self):
        renderer = MeshRenderer(  # Create a MeshRenderer for silhouette rendering
            rasterizer=MeshRasterizer(  # Set up the rasterizer with the camera and raster settings
                cameras=self.camera,  # Use the configured camera
                raster_settings=self.raster_settings  # Use the defined raster settings
            ),
            shader=SoftSilhouetteShader()  # Use a soft silhouette shader for rendering silhouettes
        )
        return renderer.to(self.device)  # Return the renderer moved to the specified device
    
    def project_silhouette(self, meshes:list):
        combined_meshes = join_meshes_as_scene(meshes)  # Combine multiple meshes into a single scene
        images = self.renderer(combined_meshes)  # Render the scene to produce images
        silhouette = images[0][..., 3]  # Extract the alpha channel (silhouette) from the rendered image
        return silhouette  # Return the computed silhouette
