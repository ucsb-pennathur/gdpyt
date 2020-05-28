from gdpyt.utils import generate_sig_input, generate_sig_calibration, generate_synthetic_images
import numpy as np
from os.path import join

# Properties of the synthetic images
n_images = 50
background_noise = 50
grid = (10, 10)
particle_diameter = 2
range_z = (-40, 40)
shape = (512, 512)
folder = r'C:\Users\silus\UCSB\master_thesis\python_stuff\gdpyt\tests\test_synthetic\DS_Grid_sigma50'

setting_file = join(folder, 'settings.txt')
testtxt_folder = join(folder, 'input')
calibtxt_folder = join(folder, 'calibration_input')
testimg_folder = join(folder, 'images')
calibimg_folder = join(folder, 'calibration_images')

# Generate input files for synthetic image generator
generate_sig_input(n_images=n_images, background_noise=background_noise, particle_density=grid, img_shape=shape,
                   range_z=range_z, particle_diameter=particle_diameter, folder=folder)
generate_sig_calibration(setting_file, np.linspace(-40, 40, 20))

# Generate .tif
generate_synthetic_images(setting_file, testtxt_folder, testimg_folder)
generate_synthetic_images(setting_file, calibtxt_folder, calibimg_folder)