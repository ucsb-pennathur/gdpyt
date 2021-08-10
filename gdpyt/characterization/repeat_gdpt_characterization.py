from gdpyt import GdpytImageCollection, GdpytCalibrationSet
from os.path import join
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.morphology import disk, square
from sklearn.neighbors import NearestNeighbors



# ----- ----- ----- ----- DEFINE PARAMETERS ----- ----- ----- ----- ----- ----- -----

# define filepaths
CALIB_PATH = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Calibration'
CALIB_IMG_PATH = join(CALIB_PATH, 'Calibration-noise-level0/Calib-0050')


# dataset information
N_CAL = 50.0
MEASUREMENT_VOLUME = 86.0

# synthetic particle generator data
MAGNIFCATION = 10
NA = 0.3
FOCAL_LENGTH = 350
REF_INDEX_MEDIUM = 1
REF_INDEX_LENS = 1.5
PIXEL_SIZE = 6.5
PIXEL_DIM_X = 1024
PIXEL_DIM_Y = 1024
BKG_MEAN = 500
BKG_NOISES = 0
POINTS_PER_PIXEL = 40
N_RAYS = 1000
GAIN = 1
CYL_FOCAL_LENGTH = 0


#optics
WAVELENGTH = 600e-9
N_0 = 1.0003
LATERAL_RESOLLUTION = 16E-6
depth_of_focus = WAVELENGTH * N_0 / NA**2 + N_0 / (MAGNIFCATION * NA) * LATERAL_RESOLLUTION

# image pre-processing
SHAPE_TOL = 0.25        # None == take any shape; 1 == take perfectly circular shape only.
MIN_P_SIZE = 2.75       # minimum particle size (area: units are in pixels) (recommended: 5)
MAX_P_SIZE = 2000       # maximum particle size (area: units are in pixels) (recommended: 200)
SAME_ID_THRESH = 7      # maximum tolerance in x- and y-directions for particle to have the same ID between images
MEDIAN_DISK = 7         # size of median disk filter - [1, 1.5, 2, 2.5, 3, 4,]
MEDIAN_PERCENT = 0.05   # percent additional threshold value from median value

# similarity
INFER_METHODS = 'bccorr'
MIN_CM = 0.8

# display options
SHOW_CALIB_PLOT = True
SAVE_CALIB_PLOT = True


# define filetypes
filetype = '.tif'


# define image processing
processing = {'median': {'args': [disk(MEDIAN_DISK)]}}
threshold = {'median_percent': [0.05]}

# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# ----- ----- ----- ----- CALIBRATION IMAGE COLLECTION ----- ----- ----- ----- ----- ----- -----
# create image collection
"""
calib_col = GdpytImageCollection(CALIB_IMG_PATH,
                                 filetype,
                                 background_subtraction=None,
                                 processing_specs=processing,
                                 thresholding_specs=threshold,
                                 min_particle_size=MIN_P_SIZE,
                                 max_particle_size=MAX_P_SIZE,
                                 shape_tol=SHAPE_TOL,
                                 folder_ground_truth='standard_gdpt')

# uniformize particle id's
calib_col.uniformize_particle_ids(threshold=SAME_ID_THRESH)

# Calibration image filename to z position dictionary
name_to_z = {}
for image in calib_col.images.values():
    name_to_z.update({image.filename: float(image.filename.split('B00')[-1].split('.')[0]) / N_CAL})  # 'calib_X.tif' to z = X
calib_set = calib_col.create_calibration(name_to_z, dilate=True)  # Dilate: dilate images to slide template
"""
# Plot calibration results
"""
# plot calibration images with identified particles
plot_calib_col_imgs = ['B00015.tif','B00020.tif', 'B00030.tif', 'B00040.tif', 'B00045.tif']
fig, ax = plt.subplots(ncols=5, figsize=(12, 4))
for i, img in enumerate(plot_calib_col_imgs):
    num_particles = len(calib_col.images[img].particles)
    ax[i].imshow(calib_col.images[img].draw_particles(raw=False, thickness=1, draw_id=False, draw_bbox=False))
    ax[i].set_title(img + ', {} particles'.format(num_particles))
    ax[i].axis('off')
plt.tight_layout
plt.show()

# plot calibration stack for a single particle
plot_calib_stack_particle_ids = [0]
for id in plot_calib_stack_particle_ids:
    calib_set.calibration_stacks[id].plot(imgs_per_row=8, fig=None, ax=None)
plt.tight_layout
plt.show()
"""
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -------

# ----- ----- ----- ----- SWEEP CALIBRATION GRID NOISE LEVEL ----- ----- ----- ----- ----- ----- --
# define filepaths
CALIB_PATH = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Calibration'
CALIB_RESULTS_PATH = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Results/Calibration'

# define sweep
N_CAL = 50
NOISE_LEVELS = ['0', '1', '2', '3', '4']

# define on/off switches
SHOW_PLOTS = True
SAVE_PLOTS = True

for n in NOISE_LEVELS:
    SAVE_ID = 'Calibration-noise-level' + n
    CALIB_IMG_PATH = join(CALIB_PATH, SAVE_ID, 'Calib-0050')

    # create image collection
    calib_col = GdpytImageCollection(CALIB_IMG_PATH,
                                     filetype,
                                     background_subtraction=None,
                                     processing_specs=processing,
                                     thresholding_specs=threshold,
                                     min_particle_size=MIN_P_SIZE,
                                     max_particle_size=MAX_P_SIZE,
                                     shape_tol=SHAPE_TOL,
                                     folder_ground_truth='standard_gdpt')

    # uniformize particle id's
    calib_col.uniformize_particle_ids(threshold=SAME_ID_THRESH)

    # Calibration image filename to z position dictionary
    name_to_z = {}
    for image in calib_col.images.values():
        name_to_z.update(
            {image.filename: float(image.filename.split('B00')[-1].split('.')[0]) / N_CAL})  # 'calib_X.tif' to z = X
    calib_set = calib_col.create_calibration(name_to_z, dilate=True)  # Dilate: dilate images to slide template

    # get calibration stacks data
    calib_stack_data = calib_set.calibration_stacks[0].calculate_stats(true_num_particles=N_CAL, measurement_volume=MEASUREMENT_VOLUME)

    # plot calibration images with identified particles
    plot_calib_col_imgs = ['B00005.tif', 'B00010.tif', 'B00015.tif', 'B00020.tif', 'B00030.tif', 'B00035.tif', 'B00040.tif', 'B00045.tif']
    fig, ax = plt.subplots(ncols=8, figsize=(16, 4))
    for i, img in enumerate(plot_calib_col_imgs):
        num_particles = len(calib_col.images[img].particles)
        ax[i].imshow(calib_col.images[img].draw_particles(raw=False, thickness=1, draw_id=False, draw_bbox=False))
        ax[i].set_title('z/h = {}'.format(calib_col.images[img]._z))
        ax[i].axis('off')
    plt.suptitle(SAVE_ID)
    plt.tight_layout
    if SAVE_PLOTS is True:
        savepath = join(CALIB_RESULTS_PATH, SAVE_ID + '_calib_col.png')
        plt.savefig(savepath)
    if SHOW_PLOTS is True:
        plt.show()

    # plot calibration stack for a single particle
    plot_calib_stack_particle_ids = [0]
    for id in plot_calib_stack_particle_ids:
        calib_set.calibration_stacks[id].plot(imgs_per_row=9, fig=None, ax=None)
    plt.suptitle(SAVE_ID)
    plt.tight_layout
    if SAVE_PLOTS is True:
        savepath = join(CALIB_RESULTS_PATH, SAVE_ID + '_calib_stack.png')
        plt.savefig(savepath)
    if SHOW_PLOTS is True:
        plt.show()

    # test images on identical calibration set
    test_col = calib_col

    # Infer position using Barnkob's ('bccorr'), znccorr' (zero-normalized cross-corr.) or 'ncorr' (normalized cross-corr.)
    test_col.infer_z(calib_set).bccorr(min_cm=MIN_CM)

    # get test collection stats
    test_col_stats = test_col.calculate_image_stats()

    # get test collection inference uncertainties
    test_col_rmse_uncertainty = test_col.calculate_rmse_uncertainty()

    # plot measurement accuracy against calibration stack
    sort_imgs = lambda x: int(x.split('calib_')[-1].split('.')[0])
    # Pass ids that should be displayed
    fig = test_col.plot_particle_coordinate_calibration(measurement_volume=MEASUREMENT_VOLUME)
    fig.suptitle(SAVE_ID)
    if SAVE_PLOTS is True:
        savefigpath = join(CALIB_RESULTS_PATH, SAVE_ID + '_rmse_uncertainty.png')
        fig.savefig(fname=savefigpath)
    if SHOW_PLOTS:
        fig.show()

    fig = test_col.plot_particle_snr_and(second_plot='area')
    fig.suptitle(SAVE_ID)
    if SAVE_PLOTS is True:
        savefigpath = join(CALIB_RESULTS_PATH, SAVE_ID + '_snr_area.png')
        fig.savefig(fname=savefigpath, bbox_inches='tight')
    if SHOW_PLOTS:
        fig.show()

    fig = test_col.plot_particle_snr_and(second_plot='solidity')
    fig.suptitle(SAVE_ID)
    if SAVE_PLOTS is True:
        savefigpath = join(CALIB_RESULTS_PATH, SAVE_ID + '_snr_solidity.png')
        fig.savefig(fname=savefigpath, bbox_inches='tight')
    if SHOW_PLOTS:
        fig.show()

    fig = test_col.plot_particle_snr_and(second_plot='percent_measured')
    fig.suptitle(SAVE_ID)
    if SAVE_PLOTS is True:
        savefigpath = join(CALIB_RESULTS_PATH, SAVE_ID + '_snr_percent_measured.png')
        fig.savefig(fname=savefigpath, bbox_inches='tight')
    if SHOW_PLOTS:
        fig.show()

    # export data to text file
    export_data = {
        'xy_uncertainty': test_col_rmse_uncertainty['mean_rmse_xy_uncertainty'],
        'z_uncertainty': test_col_rmse_uncertainty['mean_rmse_z_uncertainty'],
        'percent_particles_idd': calib_stack_data['percent_particles_idd'],
        'percent_particles_measured': test_col_rmse_uncertainty['percent_measured_particles'],
        'mean_pixel_density': test_col_stats['mean_pixel_density'],
        'mean_particle_density': test_col_stats['mean_particle_density'],
        'measurement_volume': MEASUREMENT_VOLUME,
        'n_cal': N_CAL,
        'noise_level': n,
        'filter': 'median',
        'filter_disk': MEDIAN_DISK,
        'threshold': 'median_percent',
        'threshold_percent': MEDIAN_PERCENT,
        'infer': INFER_METHODS,
        'mean_snr_filtered': test_col_stats['mean_snr_filtered'],
        'avg_snr': calib_stack_data['avg_snr'],
        'avg_area': calib_stack_data['avg_area'],
        'min_particle_dia': calib_stack_data['min_particle_size'],
        'max_particle_dia': calib_stack_data['max_particle_size'],
        'min_cm': MIN_CM,
        'min_p_size_input': MIN_P_SIZE,
        'max_p_size_input': MAX_P_SIZE,
        'shape_tol': SHAPE_TOL,
        'same_id_thresh': SAME_ID_THRESH,
        'mean_signal': test_col_stats['mean_signal'],
        'mean_background': test_col_stats['mean_background'],
        'std_background': test_col_stats['std_background'],
    }

    j=1


# ----- ----- ----- ----- TEST GDPT ON CALIBRATION IMAGES ----- ----- ----- ----- ----- ----- -----
"""
# test images on identical calibration set
test_col = calib_col

# Infer position using Barnkob's ('bccorr'), znccorr' (zero-normalized cross-corr.) or 'ncorr' (normalized cross-corr.)
test_col.infer_z(calib_set).bccorr(min_cm=MIN_CM)

# plot measurement accuracy against calibration stack
sort_imgs = lambda x: int(x.split('calib_')[-1].split('.')[0])
# Pass ids that should be displayed
fig = test_col.plot_particle_coordinate_calibration()
fig.suptitle('save_id')
fig.show()

fig = test_col.plot_particle_snr_and(second_plot='area')
fig.suptitle('save_id')
fig.show()

fig = test_col.plot_particle_snr_and(second_plot='solidity')
fig.suptitle('save_id')
fig.show()
"""
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# ----- ----- ----- ----- TEST IMAGE COLLECTION ----- ----- ----- ----- ----- ----- -----
"""
# define filepaths
TEST_PATH = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Dataset_I'
TEST_IMG_PATH = join(TEST_PATH, 'Measurement-grid-noise-level0/Images')
TEST_RESULTS_PATH = join(TEST_PATH, 'Measurement-grid-noise-level0/Coordinates')

# define image collection stats
NUM_PARTICLES = 361
NOISE_LEVEL = 0

# create test image collection
test_col = GdpytImageCollection(TEST_IMG_PATH,
                                 filetype,
                                 background_subtraction=None,
                                 processing_specs=processing,
                                 thresholding_specs=threshold,
                                 min_particle_size=MIN_P_SIZE,
                                 max_particle_size=MAX_P_SIZE,
                                 shape_tol=SHAPE_TOL,
                                 folder_ground_truth=TEST_RESULTS_PATH)

# uniformize particle id's
test_col.uniformize_particle_ids(threshold=SAME_ID_THRESH)

# Infer position using Barnkob's ('bccorr'), znccorr' (zero-normalized cross-corr.) or 'ncorr' (normalized cross-corr.)
test_col.infer_z(calib_set).bccorr(min_cm=MIN_CM)
"""
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# ----- ----- ----- ----- SWEEP MEASUREMENT GRID NOISE LEVEL ----- ----- ----- ----- ----- ----- -----
"""
# define filepaths
TEST_PATH = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Dataset_I'

# define image collection stats
NUM_PARTICLES = 361
NOISE_LEVELS = ['0', '1', '2', '3', '4']

# define on/off switches
SHOW_PLOTS = True
SAVE_PLOTS = True

for i in NOISE_LEVELS:

    TEST_IMG_PATH = join(TEST_PATH, 'Measurement-grid-noise-level' + i + '/Images')
    TEST_RESULTS_PATH = join(TEST_PATH, 'Measurement-grid-noise-level' + i + '/Coordinates')

    CALIB_RESULTS_PATH = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Results/Calibration'
    SAVE_ID = 'Measurement-grid-noise-level' + i

    # create test image collection
    test_col = GdpytImageCollection(TEST_IMG_PATH,
                                     filetype,
                                     background_subtraction=None,
                                     processing_specs=processing,
                                     thresholding_specs=threshold,
                                     min_particle_size=MIN_P_SIZE,
                                     max_particle_size=MAX_P_SIZE,
                                     shape_tol=SHAPE_TOL,
                                     folder_ground_truth=TEST_RESULTS_PATH)

    # uniformize particle id's
    test_col.uniformize_particle_ids(threshold=SAME_ID_THRESH)

    # Infer position using Barnkob's ('bccorr'), znccorr' (zero-normalized cross-corr.) or 'ncorr' (normalized cross-corr.)
    test_col.infer_z(calib_set).bccorr(min_cm=MIN_CM)

    if SHOW_PLOTS is True or SAVE_PLOTS is True:
        sort_imgs = lambda x: int(x.split('calib_')[-1].split('.')[0])
        # Pass ids that should be displayed
        fig = test_col.plot_particle_coordinate_calibration()
        fig.suptitle(SAVE_ID)
        if SAVE_PLOTS is True:
            savefigpath = join(CALIB_RESULTS_PATH, SAVE_ID + '_z_truez.png')
            fig.savefig(fname=savefigpath)
        if SHOW_PLOTS:
            fig.show()

        fig = test_col.plot_particle_snr_and(second_plot='area')
        fig.suptitle(SAVE_ID)
        if SAVE_PLOTS is True:
            savefigpath = join(CALIB_RESULTS_PATH, SAVE_ID + '_snr_area.png')
            fig.savefig(fname=savefigpath)
        if SHOW_PLOTS:
            fig.show()

        fig = test_col.plot_particle_snr_and(second_plot='solidity')
        fig.suptitle(SAVE_ID)
        if SAVE_PLOTS is True:
            savefigpath = join(CALIB_RESULTS_PATH, SAVE_ID + '_snr_solidity.png')
            fig.savefig(fname=savefigpath)
        if SHOW_PLOTS:
            fig.show()

"""