from gdpyt.utils import generate_grid_input, generate_grid_calibration, generate_synthetic_images, \
    generate_grid_input_from_function, generate_sig_settings
import numpy as np
from os.path import join

n_images = 100
n_calib = 100
grid = (10, 10)
range_z = (-40, 40)
particle_diameter = 2
overlap_scaling = 5
#folder = r'C:\Users\silus\UCSB\master_thesis\python_stuff\gdpyt\tests\test_synthetic\Rossi_DS1_Nc50_Sigma0'.format(n_calib)
folder = r'/Users/mackenzie/Desktop/gdpyt-characterization/datasets/synthetic_overlap_noise-level1/test'

# Properties of the synthetic images
setup_params = dict(
    particle_diameter = particle_diameter,  # diameter of particle
    magnification = 10,             # magnification of the simulated lens
    numerical_aperture = 0.3,       # numerical aperture
    focal_length = 350,             # must be chosen empirically by comparison with real images (typically, 350)
    ri_medium = 1,                  # refractive index of immersion medium of the light path (typically, 1)
    ri_lens = 1.5,                  # refractive index of immersion medium of the lens glass (typically, 1.5)
    pixel_size = 6.5,               # size of the side of a square pixel (in microns)
    pixel_dim_x = 1024,             # number of pixels in x-direction
    pixel_dim_y = 1024,             # number of pixels in y-direction
    background_mean = 500,          # constant value of the image background
    background_noise = 25,          # amplitude of the Gaussian noise added to the image
    points_per_pixel = 40,          # number of point sources in a particle (decrease for larger p's) (typically, 10-20)
    n_rays = 1000,                  # number of rays for point source (typically, 100-500; 500 is better quality)
    gain = 1,                       # additional gain to increase or decrease image intensity
    cyl_focal_length = 0,           # (0 if no cylindrical lens is used) for astigmatic imaging; must be chosen empirically based on real images (typically, 400)
    overlap_scaling=overlap_scaling,              # linearly-scaled overlap factor
)


# Generate settings file
settings_dict, settings_path = generate_sig_settings(setup_params, folder=folder)

# Test images ##########################################################################################################
########################################################################################################################
# Particles in a grid, random Z coordinate in specified range
"""generate_grid_input(settings_path, n_images, grid, range_z=range_z, particle_diameter=particle_diameter,
                    linear_overlap=overlap_scaling)"""

# Particles in a grid, Z coordinate given by function
#def flat_z(xy, i):
#    """Simulates a calibration stack of particles on a plane surface"""
#    return i
#generate_grid_input_from_function(settings_path, n_images, grid, function_z=flat_z, particle_diameter=0.87, folder=None)

#def gauss_z(xy, i):
#    """ Simulates deflection in the shape of a gaussian"""
#    return 0.5 * i * np.exp(-(np.linalg.norm(xy - np.array(shap)/2, axis=1) / (0.2*(shape[0] + shape[1]))))
#generate_grid_input_from_function(settings_path, n_images, grid, function_z=gauss_z, particle_diameter=5, folder=None)

# Generate .tif
"""testtxt_folder = join(folder, 'input')
testimg_folder = join(folder, 'images')
generate_synthetic_images(settings_path, testtxt_folder, testimg_folder)"""

# Calibration images ###################################################################################################
########################################################################################################################

# Particles are always in a grid for those and at the same height for the same image

# Generate .txt
generate_grid_calibration(settings_path, grid, np.linspace(range_z[0], range_z[1], n_calib),
                          particle_diameter=particle_diameter, linear_overlap=overlap_scaling)

# Generate .tif
calibtxt_folder = join(folder, 'calibration_input')
calibimg_folder = join(folder, 'calibration_images')

# Generate images
generate_synthetic_images(settings_path, calibtxt_folder, calibimg_folder)