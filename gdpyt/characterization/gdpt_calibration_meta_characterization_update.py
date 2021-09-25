"""
This program tests the GDPyT measurement accuracy on... DataSet I
"""

from gdpyt import GdpytImageCollection, GdpytSetup, GdpytCharacterize
from os.path import join
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.morphology import disk, square

# ----- ----- ----- ----- DEFINE PARAMETERS ----- ----- ----- ----- ----- ----- -----

# define file paths
CALIB_IMG_PATH = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Calibration/Calibration-noise-level0/Calib-0050'
CALIB_BASE_STRING = 'B000'
CALIB_GROUND_TRUTH_PATH = 'standard_gdpt'
CALIB_ID = 'Calibration-noise-level0'
CALIB_RESULTS_PATH = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Results/Calibration/updated_meta_characterization'

TEST_IMG_PATH = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Calibration/Calibration-noise-level2/Calib-0250'
TEST_BASE_STRING = 'B00'
TEST_GROUND_TRUTH_PATH = 'standard_gdpt'
TEST_ID = 'Calibration-noise-level2'
TEST_RESULTS_PATH = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Results/Calibration/updated_meta_characterization'

filetype = '.tif'
filetype_ground_truth = '.txt'

# display options
INSPECT_CALIB_CONTOURS = False
SHOW_CALIB_PLOTS = False
SAVE_CALIB_PLOTS = False
INSPECT_TEST_CONTOURS = False
SHOW_PLOTS = False
SAVE_PLOTS = True

# calib dataset information
CALIBRATION_Z_STEP_SIZE = 0.344
TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
IF_CALIB_IMAGE_STACK = 'first'
TAKE_CALIB_IMAGE_SUBSET_MEAN = []

# test dataset information
TEST_SUBSET = None
TRUE_NUM_PARTICLES_PER_IMAGE = 1
IF_TEST_IMAGE_STACK = 'first'
TAKE_TEST_IMAGE_SUBSET_MEAN = []

# optics
PARTICLE_DIAMETER = 2e-6
MAGNIFICATION = 10
NA = 0.3
FOCAL_LENGTH = 350
REF_INDEX_MEDIUM = 1
REF_INDEX_LENS = 1.5
PIXEL_SIZE = 6.5e-6
PIXEL_DIM_X = 1024
PIXEL_DIM_Y = 1024
BKG_MEAN = 500
BKG_NOISES = 0
POINTS_PER_PIXEL = 40
N_RAYS = 1000
GAIN = 1
CYL_FOCAL_LENGTH = 0
WAVELENGTH = 600e-9

# image pre-processing
DILATE = None # None or True
SHAPE_TOL = 0.5  # None == take any shape; 1 == take perfectly circular shape only.
MIN_P_AREA = 25  # minimum particle size (area: units are in pixels) (recommended: 5)
MAX_P_AREA = 2000  # maximum particle size (area: units are in pixels) (recommended: 200)
SAME_ID_THRESH = 8  # maximum tolerance in x- and y-directions for particle to have the same ID between images
BACKGROUND_SUBTRACTION = None

# calibration processing parameters
CALIB_TEMPLATE_PADDING = 3
CALIB_PROCESSING_METHOD = 'median'
CALIB_PROCESSING_FILTER_TYPE = 'square'
CALIB_PROCESSING_FILTER_SIZE = 2
CALIB_PROCESSING = {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}} # {'median': {'args': [square(MEDIAN_DISK), None, 'wrap']}} # {'none': {}}
CALIB_THRESHOLD_METHOD = 'manual'
CALIB_THRESHOLD_MODIFIER = 550 # MEDIAN_PERCENT = 0.05  # percent additional threshold value from median value
CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}  # {'median_percent': [MEDIAN_PERCENT]} # {'otsu': []}  #

# test processing parameters
TEST_TEMPLATE_PADDING = 2
TEST_PROCESSING_METHOD = 'median'
TEST_PROCESSING_FILTER_TYPE = 'square'
TEST_PROCESSING_FILTER_SIZE = 2
TEST_PROCESSING = {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
TEST_THRESHOLD_METHOD = 'manual'
TEST_THRESHOLD_MODIFIER = 575
TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

# similarity
STACKS_USE_RAW = True
MIN_STACKS = 0.5 # percent of calibration stack
ZERO_CALIB_STACKS = False
ZERO_STACKS_OFFSET = 0.0 # TODO: should be derived or defined by a plane of interest
INFER_METHODS = 'sknccorr'
MIN_CM = 0.5
SUB_IMAGE_INTERPOLATION = True

# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

calib_inputs = GdpytSetup.inputs(image_collection_type='calib',
                                 image_path=CALIB_IMG_PATH,
                                 image_file_type=filetype,
                                 image_base_string=CALIB_BASE_STRING,
                                 calibration_z_step_size=CALIBRATION_Z_STEP_SIZE,
                                 if_image_stack=IF_CALIB_IMAGE_STACK,
                                 take_image_stack_subset_mean_of=TAKE_CALIB_IMAGE_SUBSET_MEAN,
                                 ground_truth_file_path=CALIB_GROUND_TRUTH_PATH,
                                 ground_truth_file_type=filetype_ground_truth,
                                 true_number_of_particles=TRUE_NUM_PARTICLES_PER_CALIB_IMAGE)

calib_outputs = GdpytSetup.outputs(CALIB_RESULTS_PATH,
                                    CALIB_ID,
                                    show_plots=SHOW_CALIB_PLOTS,
                                    save_plots=SAVE_CALIB_PLOTS,
                                    inspect_contours=INSPECT_CALIB_CONTOURS)

calib_processing = GdpytSetup.processing(min_layers_per_stack=MIN_STACKS,
                                   filter_params=CALIB_PROCESSING,
                                   threshold_params=CALIB_THRESHOLD_PARAMS,
                                   background_subtraction=BACKGROUND_SUBTRACTION,
                                   processing_method=CALIB_PROCESSING_METHOD,
                                   processing_filter_type=CALIB_PROCESSING_FILTER_TYPE,
                                   processing_filter_size=CALIB_PROCESSING_FILTER_SIZE,
                                   threshold_method=CALIB_THRESHOLD_METHOD,
                                   threshold_modifier=CALIB_THRESHOLD_MODIFIER,
                                   shape_tolerance=SHAPE_TOL,
                                   min_particle_area=MIN_P_AREA,
                                   max_particle_area=MAX_P_AREA,
                                   template_padding=CALIB_TEMPLATE_PADDING,
                                   dilate=DILATE,
                                   same_id_threshold_distance=SAME_ID_THRESH,
                                   stacks_use_raw=STACKS_USE_RAW,
                                   zero_calib_stacks=ZERO_CALIB_STACKS,
                                   zero_stacks_offset=ZERO_STACKS_OFFSET
                                   )

optics = GdpytSetup.optics(particle_diameter=PARTICLE_DIAMETER,
                           magnification=MAGNIFICATION,
                           numerical_aperture=NA,
                           focal_length=FOCAL_LENGTH,
                           ref_index_medium=REF_INDEX_MEDIUM,
                           ref_index_lens=REF_INDEX_LENS,
                           pixel_size=PIXEL_SIZE,
                           pixel_dim_x=PIXEL_DIM_X,
                           pixel_dim_y=PIXEL_DIM_Y,
                           bkg_mean=BKG_MEAN,
                           bkg_noise=BKG_NOISES,
                           points_per_pixel=POINTS_PER_PIXEL,
                           n_rays=N_RAYS,
                           gain=GAIN,
                           cyl_focal_length=CYL_FOCAL_LENGTH,
                           wavelength=WAVELENGTH)


calib_settings = GdpytSetup.GdpytSetup(calib_inputs, calib_outputs, calib_processing, z_assessment=None, optics=optics)

test_inputs = GdpytSetup.inputs(image_collection_type='test',
                                image_path=TEST_IMG_PATH,
                                image_file_type=filetype,
                                image_base_string=TEST_BASE_STRING,
                                image_subset=TEST_SUBSET,
                                if_image_stack=IF_TEST_IMAGE_STACK,
                                take_image_stack_subset_mean_of=TAKE_TEST_IMAGE_SUBSET_MEAN,
                                ground_truth_file_path=TEST_GROUND_TRUTH_PATH,
                                ground_truth_file_type=filetype_ground_truth,
                                true_number_of_particles=TRUE_NUM_PARTICLES_PER_IMAGE)

test_outputs = GdpytSetup.outputs(TEST_RESULTS_PATH,
                                  TEST_ID,
                                  show_plots=SHOW_PLOTS,
                                  save_plots=SAVE_PLOTS,
                                  inspect_contours=INSPECT_TEST_CONTOURS)

test_processing = GdpytSetup.processing(min_layers_per_stack=MIN_STACKS,
                                   filter_params=TEST_PROCESSING,
                                   threshold_params=TEST_THRESHOLD_PARAMS,
                                   background_subtraction=BACKGROUND_SUBTRACTION,
                                   processing_method=TEST_PROCESSING_METHOD,
                                   processing_filter_type=TEST_PROCESSING_FILTER_TYPE,
                                   processing_filter_size=TEST_PROCESSING_FILTER_SIZE,
                                   threshold_method=TEST_THRESHOLD_METHOD,
                                   threshold_modifier=TEST_THRESHOLD_MODIFIER,
                                   shape_tolerance=SHAPE_TOL,
                                   min_particle_area=MIN_P_AREA,
                                   max_particle_area=MAX_P_AREA,
                                   template_padding=TEST_TEMPLATE_PADDING,
                                   dilate=DILATE,
                                   same_id_threshold_distance=SAME_ID_THRESH,
                                   stacks_use_raw=STACKS_USE_RAW,
                                   zero_calib_stacks=ZERO_CALIB_STACKS,
                                   zero_stacks_offset=ZERO_STACKS_OFFSET
                                   )

test_z_assessment = GdpytSetup.z_assessment(infer_method=INFER_METHODS,
                                            min_cm=MIN_CM,
                                            sub_image_interpolation=SUB_IMAGE_INTERPOLATION)

test_settings = GdpytSetup.GdpytSetup(test_inputs, test_outputs, test_processing, z_assessment=test_z_assessment, optics=optics)

calib_col, calib_set, calib_col_image_stats, calib_stack_data, test_col, test_col_stats, \
test_col_local_meas_quality, test_col_global_meas_quality = GdpytCharacterize.test(calib_settings, test_settings)

j =1