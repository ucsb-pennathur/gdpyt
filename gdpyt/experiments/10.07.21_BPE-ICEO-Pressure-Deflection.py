# Particle Tracking Datasets
"""
This script:
    1. contains the per-experiment GDPyT settings for every particle tracking dataset.
"""

# imports
from os.path import join
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.morphology import disk, square
from gdpyt import GdpytSetup, GdpytImageCollection
from gdpyt.GdpytCharacterize import *

DATASET = '10.07.21-BPE_Pressure_Deflection'
TESTSET = 'z4000um'
STATIC_TEMPLATES = True
HARD_BASELINE = True
SINGLE_PARTICLE_CALIBRATION = False

# shared variables

BASE_DIR = join('/Users/mackenzie/Desktop/gdpyt-characterization/experiments', DATASET)

# file types
FILETYPE = '.tif'
FILETYPE_GROUND_TRUTH = '.txt'

# optics
PARTICLE_DIAMETER = 5.61e-6
MAGNIFICATION = 20
DEMAG = 1
NA = 0.45
FOCAL_LENGTH = 350
REF_INDEX_MEDIUM = 1
REF_INDEX_LENS = 1.5
PIXEL_SIZE = 16e-6
PIXEL_DIM_X = 512
PIXEL_DIM_Y = 512
BKG_MEAN = 500
BKG_NOISES = 50
POINTS_PER_PIXEL = None
N_RAYS = None
GAIN = 4
CYL_FOCAL_LENGTH = 0
WAVELENGTH = 600e-9
OVERLAP_SCALING = None

optics = GdpytSetup.optics(particle_diameter=PARTICLE_DIAMETER,
                           magnification=MAGNIFICATION,
                           demag=DEMAG,
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
                           wavelength=WAVELENGTH,
                           overlap_scaling=OVERLAP_SCALING)

# image pre-processing
DILATE = None  # None or True
SHAPE_TOL = None  # None == take any shape; 1 == take perfectly circular shape only.
MIN_P_AREA = 20  # minimum particle size (area: units are in pixels) (recommended: 5)
MAX_P_AREA = 1200  # maximum particle size (area: units are in pixels) (recommended: 200)
SAME_ID_THRESH = 5  # maximum tolerance in x- and y-directions for particle to have the same ID between images
OVERLAP_THRESHOLD = 0.1
BACKGROUND_SUBTRACTION = None

CALIB_IMG_PATH = join(BASE_DIR, 'images/calibration')
CALIB_BASE_STRING = 'calib_'
CALIB_GROUND_TRUTH_PATH = None
CALIB_ID = DATASET + '-calib'
CALIB_RESULTS_PATH = join(BASE_DIR, 'results/calibration')

# calib dataset information
CALIB_SUBSET = None
CALIBRATION_Z_STEP_SIZE = 1.0
TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 600
IF_CALIB_IMAGE_STACK = 'first'
TAKE_CALIB_IMAGE_SUBSET_MEAN = None
BASELINE_IMAGE = 'calib_58.tif'

# calibration processing parameters
CALIB_TEMPLATE_PADDING = 7
CALIB_CROPPING_SPECS = {'xmin': 150, 'xmax': 350, 'ymin': 50, 'ymax': 450, 'pad': 30} # {'xmin': 350, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 30}
CALIB_PROCESSING_METHOD = 'median'
CALIB_PROCESSING_FILTER_TYPE = 'square'
CALIB_PROCESSING_FILTER_SIZE = 2
CALIB_PROCESSING_METHOD2 = 'gaussian'
CALIB_PROCESSING_FILTER_TYPE2 = None
CALIB_PROCESSING_FILTER_SIZE2 = 1
CALIB_PROCESSING = None  # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
# CALIB_PROCESSING_METHOD2: {'args': [], 'kwargs': dict(sigma=CALIB_PROCESSING_FILTER_SIZE2, preserve_range=True)}}
CALIB_THRESHOLD_METHOD = 'manual'
CALIB_THRESHOLD_MODIFIER = 175
CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

# similarity
STACKS_USE_RAW = True
MIN_STACKS = 0.5  # percent of calibration stack
ZERO_CALIB_STACKS = False
ZERO_STACKS_OFFSET = 0.0
INFER_METHODS = 'sknccorr'
MIN_CM = 0.5
SUB_IMAGE_INTERPOLATION = True

# display options
INSPECT_CALIB_CONTOURS = False
SHOW_CALIB_PLOTS = False
SAVE_CALIB_PLOTS = True

# ----- ----- ----- ----- ----- END CALIBRATION ----- ----- ----- ----- ----- ----- ----- --------

# ----- ----- ----- ----- ----- - START TEST ----- ----- ----- ----- ----- ----- ----- ----- -----

TEST_IMG_PATH = join(BASE_DIR, 'images/test', TESTSET)
TEST_BASE_STRING = 'test_'
TEST_GROUND_TRUTH_PATH = None
TEST_ID = DATASET + '-test'
TEST_RESULTS_PATH = join(BASE_DIR, 'results/test')

# test dataset information
TEST_SUBSET = [1, 30]
TRUE_NUM_PARTICLES_PER_IMAGE = TRUE_NUM_PARTICLES_PER_CALIB_IMAGE
IF_TEST_IMAGE_STACK = 'first'
TAKE_TEST_IMAGE_SUBSET_MEAN = []
TEST_PARTICLE_ID_IMAGE = 'test_001.tif'

# test processing parameters
TEST_TEMPLATE_PADDING = CALIB_TEMPLATE_PADDING - 3
TEST_CROPPING_SPECS = CALIB_CROPPING_SPECS
TEST_PROCESSING = CALIB_PROCESSING
TEST_THRESHOLD_METHOD = 'manual'
TEST_THRESHOLD_MODIFIER = 165
TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

# similarity
ASSESS_SIMILARITY_FOR_ALL_STACKS = False

# display options
INSPECT_TEST_CONTOURS = False
SHOW_PLOTS = False
SAVE_PLOTS = False

# ----- ----- ----- ----- ----- - END TEST ----- ----- ----- ----- ----- ----- ----- ----- -----

# ----- ----- ----- ----- ----- SETUP DATALOADER ----- ----- ----- ----- ----- ----- ----- ----- --------

calib_inputs = GdpytSetup.inputs(dataset=DATASET,
                                 image_collection_type='calibration',
                                 image_path=CALIB_IMG_PATH,
                                 image_file_type=FILETYPE,
                                 image_base_string=CALIB_BASE_STRING,
                                 calibration_z_step_size=CALIBRATION_Z_STEP_SIZE,
                                 single_particle_calibration=SINGLE_PARTICLE_CALIBRATION,
                                 image_subset=CALIB_SUBSET,
                                 baseline_image=BASELINE_IMAGE,
                                 hard_baseline=HARD_BASELINE,
                                 static_templates=STATIC_TEMPLATES,
                                 if_image_stack=IF_CALIB_IMAGE_STACK,
                                 take_image_stack_subset_mean_of=TAKE_CALIB_IMAGE_SUBSET_MEAN,
                                 ground_truth_file_path=CALIB_GROUND_TRUTH_PATH,
                                 ground_truth_file_type=FILETYPE_GROUND_TRUTH,
                                 true_number_of_particles=TRUE_NUM_PARTICLES_PER_CALIB_IMAGE)

calib_outputs = GdpytSetup.outputs(CALIB_RESULTS_PATH,
                                   CALIB_ID,
                                   show_plots=SHOW_CALIB_PLOTS,
                                   save_plots=SAVE_CALIB_PLOTS,
                                   inspect_contours=INSPECT_CALIB_CONTOURS)

calib_processing = GdpytSetup.processing(min_layers_per_stack=MIN_STACKS,
                                         cropping_params=CALIB_CROPPING_SPECS,
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
                                         overlap_threshold=OVERLAP_THRESHOLD,
                                         same_id_threshold_distance=SAME_ID_THRESH,
                                         stacks_use_raw=STACKS_USE_RAW,
                                         zero_calib_stacks=ZERO_CALIB_STACKS,
                                         zero_stacks_offset=ZERO_STACKS_OFFSET
                                         )

test_inputs = GdpytSetup.inputs(dataset=DATASET,
                                image_collection_type='test',
                                image_path=TEST_IMG_PATH,
                                image_file_type=FILETYPE,
                                image_base_string=TEST_BASE_STRING,
                                baseline_image=TEST_PARTICLE_ID_IMAGE,
                                hard_baseline=HARD_BASELINE,
                                static_templates=STATIC_TEMPLATES,
                                image_subset=TEST_SUBSET,
                                if_image_stack=IF_TEST_IMAGE_STACK,
                                take_image_stack_subset_mean_of=TAKE_TEST_IMAGE_SUBSET_MEAN,
                                ground_truth_file_path=TEST_GROUND_TRUTH_PATH,
                                ground_truth_file_type=FILETYPE_GROUND_TRUTH,
                                true_number_of_particles=TRUE_NUM_PARTICLES_PER_IMAGE)

test_outputs = GdpytSetup.outputs(TEST_RESULTS_PATH,
                                  TEST_ID,
                                  show_plots=SHOW_PLOTS,
                                  save_plots=SAVE_PLOTS,
                                  inspect_contours=INSPECT_TEST_CONTOURS,
                                  assess_similarity_for_all_stacks=ASSESS_SIMILARITY_FOR_ALL_STACKS)

test_processing = GdpytSetup.processing(min_layers_per_stack=MIN_STACKS,
                                        cropping_params=TEST_CROPPING_SPECS,
                                        filter_params=TEST_PROCESSING,
                                        threshold_params=TEST_THRESHOLD_PARAMS,
                                        background_subtraction=BACKGROUND_SUBTRACTION,
                                        processing_method=CALIB_PROCESSING_METHOD,
                                        processing_filter_type=CALIB_PROCESSING_FILTER_TYPE,
                                        processing_filter_size=CALIB_PROCESSING_FILTER_SIZE,
                                        threshold_method=CALIB_THRESHOLD_METHOD,
                                        threshold_modifier=CALIB_THRESHOLD_MODIFIER,
                                        shape_tolerance=SHAPE_TOL,
                                        min_particle_area=MIN_P_AREA,
                                        max_particle_area=MAX_P_AREA,
                                        template_padding=TEST_TEMPLATE_PADDING,
                                        dilate=DILATE,
                                        overlap_threshold=OVERLAP_THRESHOLD,
                                        same_id_threshold_distance=SAME_ID_THRESH,
                                        stacks_use_raw=STACKS_USE_RAW,
                                        zero_calib_stacks=ZERO_CALIB_STACKS,
                                        zero_stacks_offset=ZERO_STACKS_OFFSET
                                        )

test_z_assessment = GdpytSetup.z_assessment(infer_method=INFER_METHODS,
                                            min_cm=MIN_CM,
                                            sub_image_interpolation=SUB_IMAGE_INTERPOLATION)

calib_settings = GdpytSetup.GdpytSetup(calib_inputs, calib_outputs, calib_processing, z_assessment=None, optics=optics)
test_settings = GdpytSetup.GdpytSetup(test_inputs, test_outputs, test_processing, z_assessment=test_z_assessment, optics=optics)

# calibration collection
calib_col = GdpytImageCollection(folder=calib_settings.inputs.image_path,
                                 filetype=calib_settings.inputs.image_file_type,
                                 image_collection_type=calib_settings.inputs.image_collection_type,
                                 file_basestring=calib_settings.inputs.image_base_string,
                                 subset=calib_settings.inputs.image_subset,
                                 calibration_stack_z_step=calib_settings.inputs.calibration_z_step_size,
                                 true_num_particles=calib_settings.inputs.true_number_of_particles,
                                 folder_ground_truth=calib_settings.inputs.ground_truth_file_path,
                                 stacks_use_raw=calib_settings.processing.stacks_use_raw,
                                 crop_specs=calib_settings.processing.cropping_params,
                                 background_subtraction=calib_settings.processing.background_subtraction,
                                 processing_specs=calib_settings.processing.processing_params,
                                 thresholding_specs=calib_settings.processing.threshold_params,
                                 min_particle_size=calib_settings.processing.min_particle_area,
                                 max_particle_size=calib_settings.processing.max_particle_area,
                                 shape_tol=calib_settings.processing.shape_tolerance,
                                 overlap_threshold=calib_settings.processing.overlap_threshold,
                                 same_id_threshold=calib_settings.processing.same_id_threshold_distance,
                                 template_padding=calib_settings.processing.template_padding,
                                 if_img_stack_take=calib_settings.inputs.if_image_stack,
                                 take_subset_mean=calib_settings.inputs.take_image_stack_subset_mean_of,
                                 inspect_contours_for_every_image=calib_settings.outputs.inspect_contours,
                                 baseline=calib_settings.inputs.baseline_image,
                                 hard_baseline=HARD_BASELINE,
                                 static_templates=calib_settings.inputs.static_templates,
                                 )

# method for converting filenames to z-coordinates
name_to_z = {}
for image in calib_col.images.values():
    name_to_z.update({image.filename: float(image.filename.split(calib_settings.inputs.image_base_string)[-1].split('.tif')[0])})

# create the calibration set consisting of calibration stacks for each particle
calib_set = calib_col.create_calibration(name_to_z=name_to_z,
                                         template_padding=calib_settings.processing.template_padding,
                                         min_num_layers=calib_settings.processing.min_layers_per_stack * len(calib_col.images),
                                         self_similarity_method=test_settings.z_assessment.infer_method,
                                         dilate=calib_settings.processing.dilate)

# calculate particle similarity per image
df_sim = calib_col.calculate_image_particle_similarity()
savedata = join('/Users/mackenzie/Desktop', 'similarity_' + calib_settings.inputs.dataset + '.xlsx')
df_sim.to_excel(savedata, index=True)

# plot the baseline image with particle ID's
if calib_col.baseline is not None:
    fig = calib_col.plot_baseline_image_and_particle_ids()
    plt.show()

# export the particle coordinates
calib_coords = export_particle_coords(calib_col, calib_settings, test_settings)

# get calibration collection image stats
calib_col_image_stats = calib_col.calculate_calibration_image_stats()

# get calibration collection mean stats
calib_col_stats = calib_col.calculate_image_stats()

# get calibration stacks data
calib_stack_data = calib_set.calculate_stacks_stats()

# plot
plot_calibration(calib_settings, test_settings, calib_col, calib_set, calib_col_image_stats, calib_col_stats, calib_stack_data)

# for analyses of the standard GDPT dataset, there must not be a baseline.
if calib_settings.inputs.single_particle_calibration is False:
    """ any analysis where the particle distribution in the test collection matches the calibration collection """
    test_collection_baseline = calib_set
    test_particle_id_image = test_settings.inputs.baseline_image
elif calib_settings.inputs.single_particle_calibration is True:
    """ random synthetic particle dataset with a single calibration image """
    test_collection_baseline = test_settings.inputs.baseline_image
    test_particle_id_image = None
else:
    raise ValueError("Need to set SINGLE_PARTICLE_CALIBRATION")

# test collection
test_col = GdpytImageCollection(folder=test_settings.inputs.image_path,
                                filetype=test_settings.inputs.image_file_type,
                                image_collection_type=test_settings.inputs.image_collection_type,
                                file_basestring=test_settings.inputs.image_base_string,
                                calibration_stack_z_step=calib_settings.inputs.calibration_z_step_size,
                                subset=test_settings.inputs.image_subset,
                                true_num_particles=test_settings.inputs.true_number_of_particles,
                                folder_ground_truth=test_settings.inputs.ground_truth_file_path,
                                stacks_use_raw=test_settings.processing.stacks_use_raw,
                                crop_specs=test_settings.processing.cropping_params,
                                background_subtraction=test_settings.processing.background_subtraction,
                                processing_specs=test_settings.processing.processing_params,
                                thresholding_specs=test_settings.processing.threshold_params,
                                min_particle_size=test_settings.processing.min_particle_area,
                                max_particle_size=test_settings.processing.max_particle_area,
                                shape_tol=test_settings.processing.shape_tolerance,
                                overlap_threshold=test_settings.processing.overlap_threshold,
                                same_id_threshold=test_settings.processing.same_id_threshold_distance,
                                measurement_depth=100,  # calib_col.measurement_range,
                                template_padding=test_settings.processing.template_padding,
                                if_img_stack_take=test_settings.inputs.if_image_stack,
                                take_subset_mean=test_settings.inputs.take_image_stack_subset_mean_of,
                                inspect_contours_for_every_image=test_settings.outputs.inspect_contours,
                                baseline=test_collection_baseline,
                                hard_baseline=HARD_BASELINE,
                                static_templates=test_settings.inputs.static_templates,
                                particle_id_image=test_particle_id_image,
                                )

# method for converting filenames to z-coordinates
name_to_z_test = {}
for image in test_col.images.values():
    name_to_z_test.update({image.filename: float(image.filename.split(test_settings.inputs.image_base_string)[-1].split('.tif')[0])})
test_col.set_true_z(image_to_z=name_to_z_test)

# Infer the z-height of each particle
test_col.infer_z(calib_set, infer_sub_image=test_settings.z_assessment.sub_image_interpolation).sknccorr(
    min_cm=test_settings.z_assessment.min_cm, use_stack=None)

# export the particle coordinates
test_coords = export_particle_coords(test_col, calib_settings, test_settings)

# get test collection stats
test_col_stats = test_col.calculate_image_stats()

# get test collection inference local uncertainties
test_col_local_meas_quality = test_col.calculate_measurement_quality_local(num_bins=20, min_cm=0.5,
                                                                           true_xy=test_settings.inputs.ground_truth_file_path)

# export local measurement quality
export_local_meas_quality(calib_settings, test_settings, test_col_local_meas_quality)

# get test collection inference global uncertainties
test_col_global_meas_quality = test_col.calculate_measurement_quality_global(local=test_col_local_meas_quality)

# export data to excel
export_results_and_settings('test', calib_settings, calib_col, calib_stack_data, calib_col_stats,
                            test_settings, test_col, test_col_global_meas_quality, test_col_stats)

# export key results to excel (redundant data as full excel export but useful for quickly ascertaining results)
export_key_results(calib_settings, calib_col, calib_stack_data, calib_col_stats, test_settings, test_col,
                   test_col_global_meas_quality, test_col_stats)

# plot
plot_test(test_settings, test_col, test_col_stats, test_col_local_meas_quality, test_col_global_meas_quality)

# assess every calib stack and particle ID
assess_every_stack = False
if assess_every_stack:
    assess_every_particle_and_stack_id(test_settings, calib_set, test_col)