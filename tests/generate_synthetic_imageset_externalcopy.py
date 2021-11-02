from gdpyt.utils import generate_grid_input, generate_grid_calibration, generate_synthetic_images, \
    generate_grid_input_from_function, generate_sig_settings, generate_identical_calibration_and_test, \
    generate_uniform_z_overlap_grid, generate_random_z_overlap_grid, generate_uniform_z_density_distribution, \
    generate_random_z_density_distribution, generate_uniform_z_grid, generate_random_z_grid, \
    generate_paired_random_z_overlap_grid

import numpy as np
from os.path import join

folder = r'/Users/mackenzie/Desktop/gdpyt-characterization/datasets/synthetic_overlap_noise-level2/grid-random-z'

# settings
n_test = 50
n_calib = 81
grid = (10, 10)
particle_diameter = 2
overlap_scaling = 5
particle_density = 1e-3  # [1e-3, 2.5e-3, 5e-3, 7.5e-3, 10e-3]

# z-coordinate
range_z = (-40, 40)  # used for random z-coordinate assignment: TEST
z_levels = np.linspace(range_z[0], range_z[1], n_calib)  # used for uniform z-coordinate assignment: CALIBRATION
z_levels = np.round(z_levels, 5)

# Properties of the synthetic images
setup_params = dict(
    particle_diameter=particle_diameter,  # diameter of particle
    magnification=10,  # magnification of the simulated lens
    numerical_aperture=0.3,  # numerical aperture
    focal_length=350,  # must be chosen empirically by comparison with real images (typically, 350)
    ri_medium=1,  # refractive index of immersion medium of the light path (typically, 1)
    ri_lens=1.5,  # refractive index of immersion medium of the lens glass (typically, 1.5)
    pixel_size=6.5,  # size of the side of a square pixel (in microns)
    pixel_dim_x=1024,  # number of pixels in x-direction
    pixel_dim_y=1024,  # number of pixels in y-direction
    background_mean=500,  # constant value of the image background
    background_noise=50,  # amplitude of the Gaussian noise added to the image
    points_per_pixel=40,  # number of point sources in a particle (decrease for larger p's) (typically, 10-20)
    n_rays=1000,  # number of rays for point source (typically, 100-500; 500 is better quality)
    gain=1,  # additional gain to increase or decrease image intensity
    cyl_focal_length=0,
    # (0 if no cylindrical lens is used) for astigmatic imaging; must be chosen empirically based on real images (typically, 400)
    overlap_scaling=overlap_scaling,  # linearly-scaled overlap factor
    particle_density=particle_density  # number of particles in the field of view / area of the field of view
)

# Generate settings file
settings_dict, settings_path = generate_sig_settings(setup_params, folder=folder)

"""
Generate particle coordinate .txt files:
    1. Grid: uniform z-coordinate
    2. Grid: random z-coordinate
    3. Grid-overlap: uniform z-coordinate
    4. Grid-overlap: random z-coordinate
    5. Random distribution based on density: uniform z-coordinate
    6. Random distribution based on density: random z-coordinate
"""

""" 
1. Grid: uniform z-coordinate 
    * Generate images according to z-levels where all particles are at the same z-coordinate.

generate_uniform_z_grid(settings_path, grid, z_levels, particle_diameter, create_multiple=None)
"""

# ------------------------- ------------------------- ------------------------- ------------------------- -------------

""" 
2. Grid: random z-coordinate 
    * Generate images according to z-levels where all particles are at a random z-coordinate.

generate_random_z_grid(settings_path, n_test, grid, range_z, particle_diameter)
"""

"""z_levels= np.linspace(range_z[0], range_z[1], n_calib)
z_levels = np.round(z_levels, 5)
generate_grid_calibration(settings_path, grid, z_levels=z_levels, particle_diameter=particle_diameter,
                          linear_overlap=None, particle_density=particle_density)"""

# ------------------------- ------------------------- ------------------------- ------------------------- -------------

""" 
3. Grid overlap: uniform z-coordinate
    * Generate images according to z-levels where all particles are at a uniform z-coordinate.
"""
generate_uniform_z_overlap_grid(settings_path, grid, z_levels, particle_diameter, overlap_scaling)


# ------------------------- ------------------------- ------------------------- ------------------------- -------------

""" 
4. Grid overlap: paired random z-coordinate 
    * Generate images according to z-levels with linearly-arrayed overlapped particles at random z-coordinates.
    * Note: all particle pairs are at random z-coordinates but each particle pair is at an identical z-coordinate.
"""
generate_paired_random_z_overlap_grid(settings_path, n_test, grid, range_z, particle_diameter, overlap_scaling)


""" 
4.1 Grid overlap: random z-coordinates across the image but particle pairs have identical z-coordinate
    * Generate images according to z-levels with linearly-arrayed overlapped particles at random z-coordinates.
    * Note: all particles are at random z-coordinates

generate_random_z_overlap_grid(settings_path, n_test, grid, range_z, particle_diameter, overlap_scaling)
"""

# ------------------------- ------------------------- ------------------------- ------------------------- -------------

""" 
5. Random distribution by density: uniform z-coordinate 
    * Generate images according to z-levels with randomly distributed particles at uniform z-coordinates.

generate_uniform_z_density_distribution(settings_path, z_levels, particle_density, particle_diameter,
                                        create_multiple=None)
"""

# ------------------------- ------------------------- ------------------------- ------------------------- -------------

""" 
6. Random distribution by density: random z-coordinate 
    * Generate images according to z-levels with randomly distributed particles at random z-coordinates.

generate_random_z_density_distribution(settings_path, n_test, particle_density, range_z, particle_diameter)
"""

# ------------------------- ------------------------- ------------------------- ------------------------- -------------


# Create calibration images
calibtxt_folder = join(folder, 'calibration_input')
calibimg_folder = join(folder, 'calibration_images')
generate_synthetic_images(settings_path, calibtxt_folder, calibimg_folder)

# Create test images .tif
testtxt_folder = join(folder, 'input')
testimg_folder = join(folder, 'images')
generate_synthetic_images(settings_path, testtxt_folder, testimg_folder)









"""
Generate particle distributions based on particle density
# Generate calibration and test files
generate_identical_calibration_and_test(settings_path, z_levels_calib=np.linspace(range_z[0], range_z[1], n_calib),
                                        z_levels_test=np.linspace(range_z[0], range_z[1], n_test),
                                        particle_diameter=2, particle_density=particle_density)


# Calibration images ###################################################################################################
########################################################################################################################

# Particles are always in a grid for those and at the same height for the same image

# Generate .txt

# grid pattern
z_levels= np.linspace(range_z[0], range_z[1], n_calib)
z_levels = np.round(z_levels, 5)
generate_grid_calibration(settings_path, grid, z_levels=z_levels, particle_diameter=particle_diameter,
                          linear_overlap=overlap_scaling, particle_density=particle_density)

# Generate image inputs
generate_synthetic_images(settings_path, calibtxt_folder, calibimg_folder)

# Create calibration images
calibtxt_folder = join(folder, 'calibration_input')
calibimg_folder = join(folder, 'calibration_images')
generate_synthetic_images(settings_path, calibtxt_folder, calibimg_folder)


# Test images ##########################################################################################################
########################################################################################################################

# Particles in a grid, random Z coordinate in specified range
generate_grid_input(settings_path, n_test, grid, range_z=range_z, particle_diameter=particle_diameter,
                    linear_overlap=overlap_scaling)


# Particles in a grid, Z coordinate given by function
#def flat_z(xy, i):
#    Simulates a calibration stack of particles on a plane surface
#    return i
#generate_grid_input_from_function(settings_path, n_images, grid, function_z=flat_z, particle_diameter=0.87, folder=None)

#def gauss_z(xy, i):
#    Simulates deflection in the shape of a gaussian
#    return 0.5 * i * np.exp(-(np.linalg.norm(xy - np.array(shap)/2, axis=1) / (0.2*(shape[0] + shape[1]))))
#generate_grid_input_from_function(settings_path, n_images, grid, function_z=gauss_z, particle_diameter=5, folder=None)


# Create test images .tif
testtxt_folder = join(folder, 'input')
testimg_folder = join(folder, 'images')
generate_synthetic_images(settings_path, testtxt_folder, testimg_folder)

"""