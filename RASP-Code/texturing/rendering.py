from dependencies import *  # Import everything from the dependencies module

# Define the Rendering class responsible for setting up the camera, lights, rasterization, and rendering pipelines
class Rendering:
    def __init__(self, camera_transform: list, camera_type: str = 'orthographic', image_size: int = 128, sigma: float = 1e-4, faces_per_pixel: int = 50, environment_factor: int = 1.0, device=set_device()):
        self.device = device  # Set the computation device (CPU or GPU)
        # Initialize the camera using the set_camera method and move it to the specified device
        self.camera = self.set_camera(camera_transform, camera_type=camera_type, environment_factor=environment_factor).to(self.device)
        # Initialize a point light at the camera center for rendering illumination
        self.light = PointLights(device=self.device, location=self.camera.get_camera_center())
        # Get the rasterization settings using the provided image_size, sigma, and faces_per_pixel parameters
        self.raster_settings = self.get_raster_settings(image_size=image_size, sigma=sigma, faces_per_pixel=faces_per_pixel)
        # Create the renderer using the texture_renderer method
        self.renderer = self.texture_renderer()

    def set_camera(self, camera_transform: list, camera_type, environment_factor: int):
        if camera_type == 'orthographic':  # Check if the camera type is orthographic
            # Compute rotation and translation matrices for an orthographic camera based on camera_transform parameters
            R, T = look_at_view_transform(dist=environment_factor, elev=camera_transform[1], azim=camera_transform[2])
            # Create an orthographic camera with specified boundaries based on environment_factor
            camera = FoVOrthographicCameras(R=R, T=T, min_x=-1 * environment_factor, max_x=1 * environment_factor, min_y=-1 * environment_factor, max_y=1 * environment_factor)
        elif camera_type == 'perspective':  # Check if the camera type is perspective
            # Compute rotation and translation matrices for a perspective camera; using a larger distance (environment_factor*2)
            R, T = look_at_view_transform(dist=environment_factor * 2, elev=camera_transform[1], azim=camera_transform[2])
            # Create a perspective camera with a field of view of 90 degrees
            camera = FoVPerspectiveCameras(R=R, T=T, fov=90, degrees=True)
        return camera  # Return the configured camera

    def get_raster_settings(self, image_size, sigma, faces_per_pixel):
        # Create a RasterizationSettings object with specified image size, computed blur radius, and faces per pixel
        raster_settings_soft = RasterizationSettings(
            image_size=image_size,  # Set the resolution of the output image
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,  # Compute the blur radius based on sigma
            faces_per_pixel=faces_per_pixel,  # Set the maximum number of faces considered per pixel
        )
        return raster_settings_soft  # Return the configured rasterization settings

    def texture_renderer(self):
        # Create a MeshRenderer with a MeshRasterizer and a SoftPhongShader for textured rendering
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.camera,  # Use the configured camera for rasterization
                raster_settings=self.raster_settings  # Use the specified rasterization settings
            ),
            shader=SoftPhongShader(
                device=self.device,  # Specify the device for shader computation
                cameras=self.camera,  # Pass the camera to the shader for lighting computations
                lights=self.light  # Pass the light source to the shader
            )
        )
        return renderer.to(self.device)  # Return the renderer moved to the specified device

    def project_silhouette(self, meshes: list):
        combined_meshes = join_meshes_as_scene(meshes)  # Combine the list of meshes into a single scene mesh
        images = self.renderer(combined_meshes)  # Render the scene to obtain images
        silhouette = images[0][..., :3]  # Extract the first 3 channels (RGB) from the rendered image as the silhouette
        return silhouette  # Return the computed silhouette
