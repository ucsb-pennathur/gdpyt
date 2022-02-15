# tests/test_template_sizing
"""
This script:
    1. test script for correctly sizing the templates.
"""

# imports
import os
from os.path import join
from datetime import datetime
import random
import pandas as pd
import numpy as np
from skimage.morphology import disk, square

import matplotlib.pyplot as plt

# GDPyT imports
from gdpyt import GdpytImageCollection


# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----

# general image details
dataset = '02.07.22_membrane_characterization'
bkg_mean = 108
bkg_noise = 3
base_dir = join('/Users/mackenzie/Desktop/gdpyt-characterization/experiments', dataset)
filetype = '.tif'

# calibration image details
calib_img_path = join(base_dir, 'images/calibration')
calib_base_string = 'calib_'
calibration_z_step_size = 1.0
calib_subset = [0, 150, 50]
calib_baseline_image = 'calib_58.tif'

# test image details
test_img_path = join(base_dir, 'images/tests/dynamic/negative_to_positive')
test_base_string = 'test_X'
test_subset = [0, 500, 90]
test_baseline_image = 'test_X1.tif'

# MOST IMPORTANT USER INPUTS
calib_template_padding = 18
test_template_padding = calib_template_padding - 2
cropping_specs = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 30}
min_particle_area = 25  # minimum particle size (area: units are in pixels) (recommended: 20)
max_particle_area = 750  # (750) maximum particle size (area: units are in pixels) (recommended: 200)
same_id_threshold = 7  # maximum tolerance in x- and y-directions for particle to have the same ID between images

# image filtering/noise reduction
processing_method = 'median'
processing_filter_type = 'square'
processing_filter_size = 3
processing_params = {'none': None}

# image segmentation
threshold_method = 'manual'
threshold_modifier = bkg_mean + bkg_noise * 10
threshold_params = {threshold_method: [threshold_modifier]}

# END SETUP
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----

# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----
# START CALIBRATION COLLECTION

# calibration collection
calib_col = GdpytImageCollection(folder=calib_img_path,
                                 filetype=filetype,
                                 image_collection_type='calibration',
                                 file_basestring=calib_base_string,
                                 subset=calib_subset,
                                 calibration_stack_z_step=calibration_z_step_size,
                                 true_num_particles=1,
                                 folder_ground_truth=None,
                                 stacks_use_raw=True,
                                 crop_specs=cropping_specs,
                                 background_subtraction=None,
                                 processing_specs=processing_params,
                                 thresholding_specs=threshold_params,
                                 min_particle_size=min_particle_area,
                                 max_particle_size=max_particle_area,
                                 shape_tol=0.25,
                                 overlap_threshold=0.5,
                                 same_id_threshold=same_id_threshold,
                                 template_padding=calib_template_padding,
                                 if_img_stack_take='first',
                                 take_subset_mean=None,
                                 inspect_contours_for_every_image=False,
                                 baseline=calib_baseline_image,
                                 hard_baseline=True,
                                 static_templates=True,
                                 overlapping_particles=True,
                                 )

# calibration set
# method for converting filenames to z-coordinates
name_to_z = {}
for image in calib_col.images.values():
    name_to_z.update({image.filename: float(image.filename.split(calib_base_string)[-1].split('.tif')[0])})

# create the calibration set consisting of calibration stacks for each particle
calib_set = calib_col.create_calibration(name_to_z=name_to_z,
                                         template_padding=calib_template_padding,
                                         min_num_layers=0.9 * len(calib_col.images),
                                         self_similarity_method='sknccorr',
                                         dilate=False)

# plot the baseline image with particle ID's
if calib_col.baseline is not None:
    fig = calib_col.plot_baseline_image_and_particle_ids()
    plt.suptitle('Calibration baseline image: {}'.format(calib_baseline_image))
    plt.tight_layout()
    plt.show()

# choose particle ID's at random from calibration set
if len(calib_set.particle_ids) < 5:
    plot_calib_stack_particle_ids = calib_set.particle_ids
else:
    plot_calib_stack_particle_ids = [pid for pid in random.sample(set(calib_set.particle_ids), 5)]

for id in plot_calib_stack_particle_ids:
    # plot calibration stack and contour outline for a single particle: '_filtered_calib_stack.png'
    calib_set.calibration_stacks[id].plot_calib_stack(imgs_per_row=3, fig=None, ax=None, format_string=False)
    plt.suptitle('Calibration particle {} stack'.format(id))
    plt.tight_layout()
    plt.show()

# END CALIBRATION COLLECTION
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----

# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----
# START TEST COLLECTION
"""
# test collection
test_col = GdpytImageCollection(folder=test_img_path,
                                filetype=filetype,
                                image_collection_type='test',
                                file_basestring=test_base_string,
                                calibration_stack_z_step=calibration_z_step_size,
                                subset=test_subset,
                                true_num_particles=1,
                                folder_ground_truth=None,
                                stacks_use_raw=True,
                                crop_specs=cropping_specs,
                                background_subtraction=None,
                                processing_specs=processing_params,
                                thresholding_specs=threshold_params,
                                min_particle_size=min_particle_area,
                                max_particle_size=max_particle_area,
                                shape_tol=0.25,
                                overlap_threshold=0.5,
                                same_id_threshold=same_id_threshold,
                                measurement_depth=100,
                                template_padding=test_template_padding,
                                if_img_stack_take='first',
                                take_subset_mean=None,
                                inspect_contours_for_every_image=False,
                                baseline=calib_set,
                                hard_baseline=True,
                                static_templates=True,
                                particle_id_image=test_baseline_image,
                                overlapping_particles=True,
                                xydisplacement=[[0, 0]],
                                )

# plot the baseline image with particle ID's
if test_col.baseline is not None:
    fig = test_col.plot_baseline_image_and_particle_ids()
    plt.suptitle('Test baseline image: {}'.format(test_baseline_image))
    plt.tight_layout()
    plt.show()

# plot for a random selection of particles in the test collection
if len(test_col.particle_ids) >= 15:
    plot_test_particle_ids = [int(p) for p in random.sample(set(test_col.particle_ids), 15)]
else:
    plot_test_particle_ids = test_col.particle_ids

for plot_test_particle_ids in plot_test_particle_ids:
    plot_test_particle_ids = int(plot_test_particle_ids)

    # plot particle stack with z and true_z as subplot titles
    fig = test_col.plot_single_particle_stack(particle_id=plot_test_particle_ids)
    if fig is not None:
        fig.suptitle('Test particle id {}'.format(plot_test_particle_ids))
        plt.tight_layout()
        plt.show()
"""

# END TEST COLLECTION
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----

# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----
print("Analysis completed without errors.")