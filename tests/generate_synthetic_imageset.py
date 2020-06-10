from gdpyt.utils import generate_grid_input, generate_grid_calibration, generate_synthetic_images, \
    generate_grid_input_from_function
import numpy as np
from os.path import join

# Properties of the synthetic images
n_images = 50
background_noise = 20
grid = (15, 15)
particle_diameter = 2
range_z = (-40, 40)
shape = (512, 512)
folder = r'C:\Users\silus\UCSB\master_thesis\python_stuff\gdpyt\tests\test_synthetic\DS_Grid_Gaussian_N50_Sigma20'

setting_file = join(folder, 'settings.txt')
testtxt_folder = join(folder, 'input')
calibtxt_folder = join(folder, 'calibration_input')
testimg_folder = join(folder, 'images')
calibimg_folder = join(folder, 'calibration_images')

# Generate input files for synthetic image generator

# Particles in a grid, random Z coordinate in specified range
#generate_grid_input(n_images, grid, background_noise=background_noise,  img_shape=shape,
#                   range_z=range_z, particle_diameter=particle_diameter, folder=folder)

# Particles in a grid, Z coordinate given by function
def gauss_z(xy, i):
    """ Simulates deflection in the shape of a gaussian"""
    return 0.5 * i * np.exp(-(np.linalg.norm(xy - np.array(shape)/2, axis=1) / (0.2*(shape[0] + shape[1]))))
generate_grid_input_from_function(n_images, grid, gauss_z, background_noise=background_noise,  img_shape=shape,
                   particle_diameter=particle_diameter, folder=folder)

# Calibration images. Particles are always in a grid for those and at the same height for the same image
generate_grid_calibration(setting_file, grid, np.linspace(-40, 40, 20))

# Generate .tif
generate_synthetic_images(setting_file, testtxt_folder, testimg_folder)
generate_synthetic_images(setting_file, calibtxt_folder, calibimg_folder)