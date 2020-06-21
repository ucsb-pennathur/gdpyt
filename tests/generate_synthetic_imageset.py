from gdpyt.utils import generate_grid_input, generate_grid_calibration, generate_synthetic_images, \
    generate_grid_input_from_function, generate_sig_settings
import numpy as np
from os.path import join

# Properties of the synthetic images
setup_params = dict(
    magnification = 10,
    numerical_aperture = 0.3,
    focal_length = 350,
    ri_medium = 1,
    ri_lens = 1.5,
    pixel_size = 6.5,
    pixel_dim_x = 512,
    pixel_dim_y = 512,
    background_mean = 200,
    background_noise = 0,
    points_per_pixel = 18,
    n_rays = 500,
    gain = 3.2,
    cyl_focal_length = 0
)

n_images = 50
n_calib = 50
grid = (10, 10)
range_z = (-40, 40)
particle_diameter = 2
folder = r'C:\Users\silus\UCSB\master_thesis\python_stuff\gdpyt\tests\test_synthetic\Rossi_DS1_Nc50_Sigma0'.format(n_calib)

# Generate settings file
settings_dict, settings_path = generate_sig_settings(setup_params, folder=folder)


# Test images ##########################################################################################################
########################################################################################################################
# Particles in a grid, random Z coordinate in specified range
generate_grid_input(settings_path, n_images, grid, range_z=range_z, particle_diameter=particle_diameter)

# Particles in a grid, Z coordinate given by function
#def gauss_z(xy, i):
#    """ Simulates deflection in the shape of a gaussian"""
#    return 0.5 * i * np.exp(-(np.linalg.norm(xy - np.array(shap)/2, axis=1) / (0.2*(shape[0] + shape[1]))))
#generate_grid_input_from_function(settings_path, n_images, grid, function_z=gauss_z, particle_diameter=5, folder=None)

# Generate .tif
testtxt_folder = join(folder, 'input')
testimg_folder = join(folder, 'images')
generate_synthetic_images(settings_path, testtxt_folder, testimg_folder)

# Calibration images ###################################################################################################
########################################################################################################################

# Particles are always in a grid for those and at the same height for the same image

# Generate .txt
generate_grid_calibration(settings_path, grid, np.linspace(range_z[0], range_z[1], n_calib),
                          particle_diameter=particle_diameter)

# Generate .tif
calibtxt_folder = join(folder, 'calibration_input')
calibimg_folder = join(folder, 'calibration_images')

# Generate images
generate_synthetic_images(settings_path, calibtxt_folder, calibimg_folder)