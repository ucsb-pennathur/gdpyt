# Particle Tracking Datasets
"""
This script:
    1. contains the per-experiment GDPyT settings for every particle tracking dataset.
"""

# imports
import os
from os.path import join
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.morphology import disk, square
from gdpyt import GdpytSetup

class dataset_unpacker(object):
    def __init__(self, dataset, collection_type, noise_level=None, number_of_images=None, particle_distribution=None,
                 particle_density=None, static_templates=False, single_particle_calibration=False,
                 hard_baseline=False, particles_overlapping=False, sweep_method=None, sweep_param=None):
        self.dataset = dataset
        self.collection_type = collection_type
        self.noise_level = noise_level
        self.number_of_images = number_of_images
        self.particle_distribution = particle_distribution
        self.particle_density = particle_density
        self.static_templates = static_templates
        self.single_particle_calibration = single_particle_calibration
        self.hard_baseline = hard_baseline
        self.particles_overlapping = particles_overlapping
        self.sweep_method = sweep_method
        self.sweep_param = sweep_param

    def unpack(self):

        if self.dataset == 'JP-EXF01-20':
            """
            Notes:
                * This dataset spans the z-coordinates: -67 to +18 (i.e. z-range = 85)
            """

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20'

            assert self.number_of_images is not None
            STATIC_TEMPLATES = self.static_templates
            SINGLE_PARTICLE_CALIBRATION = self.single_particle_calibration
            if self.particle_distribution == 'Dataset_I':
                assert self.noise_level is not None
                self.particle_density = 361
            elif self.particle_distribution == 'Dataset_II':
                assert self.particle_density is not None
                self.noise_level = 0

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 2e-6
            MAGNIFICATION = 10
            DEMAG = 1
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
                                       wavelength=WAVELENGTH)

            # image pre-processing
            DILATE = None  # None or True
            SHAPE_TOL = None  # None == take any shape; 1 == take perfectly circular shape only.
            MIN_P_AREA = 50  # minimum particle size (area: units are in pixels) (recommended: 5)
            MAX_P_AREA = 5000  # maximum particle size (area: units are in pixels) (recommended: 200)
            SAME_ID_THRESH = 10  # 8  # maximum tolerance in x- and y-directions for particle to have the same ID between images
            OVERLAP_THRESHOLD = 0.1
            BACKGROUND_SUBTRACTION = None

            if self.collection_type == 'calibration':

                if len(str(self.number_of_images)) == 3:
                    leading_zeros = '00'
                else:
                    leading_zeros = '000'

                CALIB_IMG_PATH = join(base_dir, 'Calibration/Calibration-noise-level{}/Calib-{}{}'.format(self.noise_level, leading_zeros[:-1], self.number_of_images))
                CALIB_BASE_STRING = 'B{}'.format(leading_zeros)
                CALIB_GROUND_TRUTH_PATH = 'standard_gdpt'
                CALIB_ID = 'JP-EXF01-Calib-nl{}_Num-images-{}'.format(self.noise_level, self.number_of_images)
                CALIB_RESULTS_PATH = join(base_dir, 'Results/{}'.format(self.particle_distribution))

                # calib dataset information
                CALIBRATION_Z_STEP_SIZE = 86 / self.number_of_images
                CALIB_SUBSET = None
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                IF_CALIB_IMAGE_STACK = 'first'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = []
                BASELINE_IMAGE = None

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 4
                CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 59, 'ymin': 0, 'ymax': 59, 'pad': 5}
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 7
                CALIB_PROCESSING_METHOD2 = 'gaussian'
                CALIB_PROCESSING_FILTER_TYPE2 = None
                CALIB_PROCESSING_FILTER_SIZE2 = 1
                if self.particle_distribution == 'Dataset_II':
                    CALIB_PROCESSING = None
                    calib_threshold_modifier = 515
                elif self.particle_distribution == 'Dataset_I':
                    CALIB_PROCESSING = {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                    calib_threshold_modifier = 630  # noise levels: 0=515, 1=575, 2=665 (w/o filtering)
                    """ Barnkob and Rossi use a threshold of max[1, noise] * background--which equals 750 for sigma = 50 """

                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = calib_threshold_modifier  # 575
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                if self.particle_distribution == 'Dataset_II':
                    STACKS_USE_RAW = True
                elif self.particle_distribution == 'Dataset_I':
                    STACKS_USE_RAW = False
                MIN_STACKS = 0.5  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = -67.5
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = True

            if self.collection_type == 'test':

                TEST_BASE_STRING = 'B00'
                TEST_ID = 'JP-EXF01-{}-pnum{}-nl{}'.format(self.particle_distribution, self.particle_density, self.noise_level)
                TEST_RESULTS_PATH = join(base_dir, 'Results/{}'.format(self.particle_distribution))

                if self.particle_distribution == 'Dataset_I':
                    TEST_IMG_PATH = join(base_dir, '{}/Measurement-grid-noise-level{}/Images'.format(self.particle_distribution, self.noise_level))
                    TEST_GROUND_TRUTH_PATH =  join(base_dir, '{}/Measurement-grid-noise-level{}/Coordinates'.format(self.particle_distribution, self.noise_level))

                elif self.particle_distribution == 'Dataset_II':
                    TEST_IMG_PATH = join(base_dir, '{}/Images/Part-per-image-{}'.format(self.particle_distribution, self.particle_density))
                    TEST_GROUND_TRUTH_PATH = join(base_dir, '{}/Coordinates/Part-per-image-{}'.format(self.particle_distribution, self.particle_density))
                    TEST_RESULTS_PATH = join(base_dir, 'Results/{}'.format(self.particle_distribution))

                # test dataset information
                TEST_SUBSET = [1, self.number_of_images]
                TRUE_NUM_PARTICLES_PER_IMAGE = float(self.particle_density)
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = []
                TEST_PARTICLE_ID_IMAGE = 'B00001.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 3
                TEST_CROPPING_SPECS = None  # {'xmin': 150, 'xmax': 850, 'ymin': 150, 'ymax': 850, 'pad': 20}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 7
                TEST_PROCESSING_METHOD2 = 'gaussian'
                TEST_PROCESSING_FILTER_TYPE2 = None
                TEST_PROCESSING_FILTER_SIZE2 = 1

                if self.particle_distribution == 'Dataset_II':
                    TEST_PROCESSING = None
                    test_threshold_modifier = 575
                elif self.particle_distribution == 'Dataset_I':
                    TEST_PROCESSING = {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                    test_threshold_modifier = 630  # noise levels: 0=515, 1=575, 2=665 (w/o filtering)

                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = test_threshold_modifier
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                if self.particle_distribution == 'Dataset_II':
                    STACKS_USE_RAW = True
                elif self.particle_distribution == 'Dataset_I':
                    STACKS_USE_RAW = False
                MIN_STACKS = 0.5  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = True
                ASSESS_SIMILARITY_FOR_ALL_STACKS = False

                # display options
                INSPECT_TEST_CONTOURS = False
                SHOW_PLOTS = False
                SAVE_PLOTS = True

        if self.dataset == 'synthetic_overlap_noise-level':

            assert self.noise_level is not None

            SINGLE_PARTICLE_CALIBRATION = self.single_particle_calibration

            # organize noise-dependent variables
            nl = self.noise_level
            if nl == 0:
                bkg_noise = 0
                calibration_z_step_size = 1.0
                baseline_image = 'calib_-40.0.tif'
                threshold_modifier = 615
            elif nl == 1:
                bkg_noise = 25
                calibration_z_step_size = 1.0
                threshold_modifier = 575  # 585-600 w/o filter
                baseline_image = 'calib_-15.0.tif'
                test_particle_id_image = 'B0001.tif'
            elif nl == 2:
                bkg_noise = 50
                calibration_z_step_size = 1.0  # overlap grid: 1.01266

                if self.sweep_method == 'filter' and self.sweep_param == 0:
                    threshold_modifier = 665
                else:
                    threshold_modifier = 575

                if self.sweep_method == 'baseline_image':
                    baseline_image = self.sweep_param
                else:
                    baseline_image = 'calib_-25.0.tif'  # overlap grid: 'calib_-36.962025316455694.tif'
                test_particle_id_image = 'B0001.tif'  # overlap grid: 'calib_-36.76767676767677.tif'
            else:
                raise ValueError("Noise level {} not in dataset (yet)".format(nl))

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/synthetic_overlap_noise-level{}'.format(nl)

            if self.particle_distribution == 'grid-random-z' and SINGLE_PARTICLE_CALIBRATION is False:
                calib_id = 'Grid-random-z_calib_nl{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid-random-z')
                base_dir = join(base_dir, 'grid-random-z')
                calib_cropping_specs = {'xmin': 0, 'xmax': 960, 'ymin': 0, 'ymax': 960, 'pad': 1}
                calib_baseline_image = baseline_image
                calib_true_num_particles = 170
                overlap_scaling = 5
                test_base_string = 'B00'
                true_num_particles = calib_true_num_particles
                cropping_specs = calib_cropping_specs
            elif self.particle_distribution == 'grid-random-z' and SINGLE_PARTICLE_CALIBRATION is True:
                calib_id = 'Grid-random-z_calib_nl{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid-random-z')
                base_dir = join(base_dir, 'grid-random-z')
                calib_cropping_specs = {'xmin': 50, 'xmax': 960, 'ymin': 50, 'ymax': 960, 'pad': 1}
                calib_baseline_image = None
                calib_true_num_particles = 170
                overlap_scaling = 5
                test_base_string = 'B00'
                cropping_specs = calib_cropping_specs  # {'xmin': 50, 'xmax': 690, 'ymin': 50, 'ymax': 690, 'pad': 1}
                true_num_particles = 170  # 49
            elif self.particle_distribution == 'grid-no-overlap' and SINGLE_PARTICLE_CALIBRATION is False:
                calib_id = 'Grid_calib_nl{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid-no-overlap')
                base_dir = join(base_dir, 'grid-no-overlap')
                calib_cropping_specs = {'xmin': 50, 'xmax': 690, 'ymin': 50, 'ymax': 690, 'pad': 1}
                calib_baseline_image = baseline_image
                calib_true_num_particles = 49
                overlap_scaling = None
                true_num_particles = calib_true_num_particles
                cropping_specs = calib_cropping_specs
            elif self.particle_distribution == 'grid-no-overlap' and SINGLE_PARTICLE_CALIBRATION is True:
                calib_id = 'Grid_calib_nl{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid-no-overlap')
                base_dir = join(base_dir, 'grid-no-overlap')
                calib_cropping_specs = {'xmin': 50, 'xmax': 240, 'ymin': 50, 'ymax': 240, 'pad': 1}
                calib_baseline_image = None
                calib_true_num_particles = 4
                overlap_scaling = None
                cropping_specs = {'xmin': 50, 'xmax': 690, 'ymin': 50, 'ymax': 690, 'pad': 1}
                true_num_particles = 49
            elif self.particle_distribution == 'grid' and SINGLE_PARTICLE_CALIBRATION is True:
                calib_id = 'GridOverlapSPC_calib_nll{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid')
                base_dir = join(base_dir, 'grid')
                calib_cropping_specs = {'xmin': 50, 'xmax': 145, 'ymin': 50, 'ymax': 145, 'pad': 20}
                calib_baseline_image = None
                calib_true_num_particles = 1
                overlap_scaling = 5
                true_num_particles = 170
                cropping_specs = None
            elif self.particle_distribution == 'grid' and SINGLE_PARTICLE_CALIBRATION is False:
                calib_id = 'GridOverlapSPC_calib_nll{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid')
                base_dir = join(base_dir, 'grid')
                calib_cropping_specs = None
                calib_baseline_image = baseline_image
                calib_true_num_particles = 1
                overlap_scaling = 5
                true_num_particles = 170
                cropping_specs = None
            elif self.particle_distribution == 'random' and SINGLE_PARTICLE_CALIBRATION is False:
                calib_id = 'RandOverlapGDPyT_calib_pd{}_nl{}_'.format(self.particle_density, nl)
                calib_base_dir = join(base_dir, 'random', 'particle_density_' + self.particle_density)
                base_dir = join(base_dir, 'random', 'particle_density_' + self.particle_density)
                overlap_scaling = None
                calib_true_num_particles = 180
                calib_cropping_specs = {'xmin': 0, 'xmax': 1023, 'ymin': 0, 'ymax': 1023, 'pad': 30}
                calib_baseline_image = 'calib_-27.0.tif'
                true_num_particles = 180  # 1048
                baseline_image = 'calib_-27.0.tif'  # 'calib_-35.0.tif'
                test_particle_id_image = 'calib_-27.1356783919598.tif'  # 'calib_-35.175879396984925.tif'
                cropping_specs = {'xmin': 250, 'xmax': 750, 'ymin': 250, 'ymax': 750, 'pad': 30}

            elif self.particle_distribution == 'random' and SINGLE_PARTICLE_CALIBRATION is True:
                calib_id = 'RandOverlapSPC_calib_pd{}_nl{}_'.format(self.particle_density, nl)
                calib_base_dir = join(base_dir, 'random', 'particle_density_1e-3')
                base_dir = join(base_dir, 'random', 'particle_density_' + self.particle_density)
                calib_cropping_specs = {'xmin': 623, 'xmax': 698, 'ymin': 186, 'ymax': 261, 'pad': 30}
                threshold_modifier = 600
                calib_baseline_image = None
                calib_true_num_particles = 1
                overlap_scaling = None

                if self.particle_density == '1e-3':
                    true_num_particles = 104
                    baseline_image = 'calib_-27.0.tif' # 'calib_-35.0.tif'
                    test_particle_id_image = 'calib_-27.1356783919598.tif' # 'calib_-35.175879396984925.tif'
                    cropping_specs = {'xmin': 0, 'xmax': 1023, 'ymin': 0, 'ymax': 1023, 'pad': 30}
                    # cropping_specs = {'xmin': 0, 'xmax': 470, 'ymin': 0, 'ymax': 1024, 'pad': 30}
                elif self.particle_density == '2.5e-3':
                    true_num_particles = 175
                    baseline_image = 'calib_-27.0.tif' # 'calib_-35.0.tif'
                    test_particle_id_image = 'calib_-27.1356783919598.tif' # 'calib_-35.175879396984925.tif'
                    cropping_specs = {'xmin': 100, 'xmax': 900, 'ymin': 100, 'ymax': 900, 'pad': 30}
                elif self.particle_density == '5e-3':
                    true_num_particles = 175 # 524
                    baseline_image = 'calib_-27.0.tif' # 'calib_-35.0.tif'
                    test_particle_id_image = 'calib_-27.1356783919598.tif' # 'calib_-35.175879396984925.tif'
                    cropping_specs = {'xmin': 225, 'xmax': 775, 'ymin': 225, 'ymax': 775, 'pad': 30}
                elif self.particle_density == '7.5e-3':
                    true_num_particles = 300 # 786
                    baseline_image = 'calib_-26.0.tif' # 'calib_-35.0.tif'
                    test_particle_id_image = 'calib_-25.527638190954775.tif' # 'calib_-35.175879396984925.tif'
                    cropping_specs = {'xmin': 215, 'xmax': 825, 'ymin': 215, 'ymax': 825, 'pad': 30}
                elif self.particle_density == '10e-3':
                    true_num_particles = 180 # 1048
                    baseline_image = 'calib_-27.0.tif' # 'calib_-35.0.tif'
                    test_particle_id_image = 'calib_-27.1356783919598.tif' # 'calib_-35.175879396984925.tif'
                    cropping_specs = {'xmin': 250, 'xmax': 750, 'ymin': 250, 'ymax': 750, 'pad': 30}
                else:
                    raise ValueError("Unknown particle density")

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 2e-6
            MAGNIFICATION = 10
            DEMAG = 1
            NA = 0.3
            FOCAL_LENGTH = 350
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 6.5e-6
            PIXEL_DIM_X = 1024
            PIXEL_DIM_Y = 1024
            BKG_MEAN = 500
            BKG_NOISES = bkg_noise
            POINTS_PER_PIXEL = 40
            N_RAYS = 1000
            GAIN = 1
            CYL_FOCAL_LENGTH = 0
            WAVELENGTH = 600e-9
            OVERLAP_SCALING = overlap_scaling

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
            MIN_P_AREA = 50  # minimum particle size (area: units are in pixels) (recommended: 5)
            MAX_P_AREA = 22000  # maximum particle size (area: units are in pixels) (recommended: 200)
            if self.sweep_method == 'same_id_thresh':
                SAME_ID_THRESH = self.sweep_param
            else:
                SAME_ID_THRESH = 5  # maximum tolerance in x- and y-directions for particle to have the same ID between images
            OVERLAP_THRESHOLD = 0.1
            BACKGROUND_SUBTRACTION = None

            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(calib_base_dir, 'calibration_images')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = join(calib_base_dir, 'calibration_input')

                if self.sweep_param is not None:
                    if self.single_particle_calibration is True:
                        CALIB_ID = calib_id + 'SPC_{}'.format(self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-SPC-{}'.format(self.sweep_param))
                    else:
                        CALIB_ID = calib_id + 'Static_{}'.format(self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-static-{}'.format(self.sweep_param))
                else:
                    CALIB_ID = calib_id
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = None
                CALIBRATION_Z_STEP_SIZE = calibration_z_step_size
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = calib_true_num_particles
                IF_CALIB_IMAGE_STACK = 'first'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = []
                BASELINE_IMAGE = calib_baseline_image

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 6
                CALIB_CROPPING_SPECS = calib_cropping_specs  # cropping_specs
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                if self.sweep_method == 'filter':
                    CALIB_PROCESSING_FILTER_SIZE = self.sweep_param
                else:
                    CALIB_PROCESSING_FILTER_SIZE = 5
                CALIB_PROCESSING_METHOD2 = 'gaussian'
                CALIB_PROCESSING_FILTER_TYPE2 = None
                CALIB_PROCESSING_FILTER_SIZE2 = 1

                if self.sweep_method == 'filter' and self.sweep_param == 0:
                    CALIB_PROCESSING = {'none': None}
                else:
                    CALIB_PROCESSING = {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}

                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = threshold_modifier
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                if self.sweep_method == 'filter' and self.sweep_param == 0:
                    STACKS_USE_RAW = True
                else:
                    STACKS_USE_RAW = False

                MIN_STACKS = 0.5  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = False

            if self.collection_type == 'test':
                TEST_IMG_PATH = join(base_dir, 'test_images')
                TEST_BASE_STRING = test_base_string  # 'B00' or 'calib_'
                TEST_GROUND_TRUTH_PATH = join(base_dir, 'test_input')

                if self.sweep_param is not None:
                    if self.single_particle_calibration is True:
                        TEST_ID = '{}_test_SPC_nl{}_sweep{}_'.format(self.particle_distribution, nl, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir, 'results/test-SPC-{}'.format(self.sweep_param))
                    else:
                        TEST_ID = '{}_test_static_nl{}_sweep{}_'.format(self.particle_distribution, nl, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir, 'results/test-static-{}'.format(self.sweep_param))
                else:
                    TEST_ID = '{}_test_nl{}_'.format(self.particle_distribution, nl)
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = [0, self.number_of_images]
                TRUE_NUM_PARTICLES_PER_IMAGE = true_num_particles
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = []
                TEST_PARTICLE_ID_IMAGE = test_particle_id_image

                # test processing parameters
                TEST_TEMPLATE_PADDING = 2
                TEST_CROPPING_SPECS = cropping_specs
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                if self.sweep_method == 'filter':
                    TEST_PROCESSING_FILTER_SIZE = self.sweep_param
                else:
                    TEST_PROCESSING_FILTER_SIZE = 5
                TEST_PROCESSING_METHOD2 = 'gaussian'
                TEST_PROCESSING_FILTER_TYPE2 = None
                TEST_PROCESSING_FILTER_SIZE2 = 1

                if self.sweep_method == 'filter' and self.sweep_param == 0:
                    TEST_PROCESSING = {'none': None}
                else:
                    TEST_PROCESSING = {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}

                # TEST_PROCESSING_METHOD2: {'args': [], 'kwargs': dict(sigma=TEST_PROCESSING_FILTER_SIZE2, preserve_range=True)}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = threshold_modifier
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                if self.sweep_method == 'filter' and self.sweep_param == 0:
                    STACKS_USE_RAW = True
                else:
                    STACKS_USE_RAW = False

                MIN_STACKS = 0.5  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = True
                ASSESS_SIMILARITY_FOR_ALL_STACKS = False

                # display options
                INSPECT_TEST_CONTOURS = False
                SHOW_PLOTS = False
                SAVE_PLOTS = True

        if self.dataset == '20X_1Xmag_5.61umPink_HighInt':

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/20X_1Xmag_5.61umPink_HighInt'

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 5.61e-6
            DEMAG = 1
            MAGNIFICATION = 20
            NA = 0.45
            FOCAL_LENGTH = 350
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 512
            PIXEL_DIM_Y = 512
            BKG_MEAN = 130
            BKG_NOISES = 8
            POINTS_PER_PIXEL = None
            N_RAYS = None
            GAIN = 1
            CYL_FOCAL_LENGTH = 0
            WAVELENGTH = 600e-9

            optics = GdpytSetup.optics(particle_diameter=PARTICLE_DIAMETER,
                                       demag=DEMAG,
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

            # image pre-processing
            DILATE = None  # None or True
            SHAPE_TOL = 0.75  # None == take any shape; 1 == take perfectly circular shape only.
            MIN_P_AREA = np.pi * optics.pixels_per_particle_in_focus ** 2 * 0.4  # minimum particle size (area: units are in pixels) (recommended: 5)
            MAX_P_AREA = np.pi * np.max(optics.particle_diameter_z1 / 11) ** 2  # maximum particle size (area: units are in pixels) (recommended: 200)
            SAME_ID_THRESH = int(optics.pixels_per_particle_in_focus)  # maximum tolerance in x- and y-directions for particle to have the same ID between images
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 8

            if self.collection_type == 'calibration':

                    CALIB_IMG_PATH = join(base_dir, 'images/calibration')
                    CALIB_BASE_STRING = 'calib_'
                    CALIB_GROUND_TRUTH_PATH = None
                    CALIB_ID = self.dataset + '_Calib'
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')

                    # calib dataset information
                    CALIB_SUBSET = [5, 90]
                    CALIBRATION_Z_STEP_SIZE = 1.0
                    TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 13
                    IF_CALIB_IMAGE_STACK = 'first'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = []
                    BASELINE_IMAGE = 'calib_40.tif'

                    # calibration processing parameters
                    CALIB_TEMPLATE_PADDING = 5
                    CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
                    CALIB_PROCESSING_METHOD = 'median'
                    CALIB_PROCESSING_FILTER_TYPE = 'square'
                    CALIB_PROCESSING_FILTER_SIZE = int(np.round(optics.pixels_per_particle_in_focus / 3))
                    CALIB_PROCESSING = {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None,
                                                                           'wrap']}}
                    CALIB_THRESHOLD_METHOD = 'manual'
                    CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 5
                    CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                    # similarity
                    STACKS_USE_RAW = False
                    MIN_STACKS = 0.5  # percent of calibration stack
                    ZERO_CALIB_STACKS = False
                    ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                    INFER_METHODS = 'sknccorr'
                    MIN_CM = 0.5
                    SUB_IMAGE_INTERPOLATION = True

                    # display options
                    INSPECT_CALIB_CONTOURS = False
                    SHOW_CALIB_PLOTS = False
                    SAVE_CALIB_PLOTS = False

            elif self.collection_type == 'test':
                TEST_IMG_PATH = join(base_dir, 'images/calibration')
                TEST_BASE_STRING = 'calib_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = self.dataset + '_Meta-Test'
                TEST_RESULTS_PATH = join(base_dir, 'results/test')

                # test dataset information
                TEST_SUBSET = None
                TRUE_NUM_PARTICLES_PER_IMAGE = 13
                IF_TEST_IMAGE_STACK = 'subset'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [1, 2]
                TEST_PARTICLE_ID_IMAGE = 'calib_40.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 2
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = int(np.round(optics.pixels_per_particle_in_focus / 3))
                TEST_PROCESSING = {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 5
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = False
                MIN_STACKS = 0.5  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = True
                ASSESS_SIMILARITY_FOR_ALL_STACKS = False

                # display options
                INSPECT_TEST_CONTOURS = False
                SHOW_PLOTS = False
                SAVE_PLOTS = True

        if self.dataset == '10X_0.5Xmag_2.15umNR_HighInt_0.03XHg':

            base_dir = join('/Users/mackenzie/Desktop/gdpyt-characterization/calibration', self.dataset)

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 2.15e-6
            DEMAG = 0.5
            MAGNIFICATION = 10
            NA = 0.3
            FOCAL_LENGTH = 350
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 512
            PIXEL_DIM_Y = 512
            BKG_MEAN = 105
            BKG_NOISES = 11
            POINTS_PER_PIXEL = None
            N_RAYS = None
            GAIN = 1
            CYL_FOCAL_LENGTH = 0
            WAVELENGTH = 600e-9
            Z_RANGE = 40

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
                                       z_range=Z_RANGE)

            # image pre-processing
            DILATE = None  # None or True
            SHAPE_TOL = 0.25  # None == take any shape; 1 == take perfectly circular shape only.
            MIN_P_AREA = 4  # minimum particle size (area: units are in pixels) (recommended: 5)
            MAX_P_AREA = 2000     # maximum particle size (area: units are in pixels) (recommended: 200)
            SAME_ID_THRESH = 10  # maximum tolerance in x- and y-directions for particle to have the same ID between images
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 8

            if self.collection_type == 'calibration':
                CALIB_IMG_PATH = join(base_dir, 'images/calibration')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = self.dataset + '_Calib'
                CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')

                # calib dataset information
                CALIB_SUBSET = None
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 150
                IF_CALIB_IMAGE_STACK = 'first'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = []
                BASELINE_IMAGE = 'calib_90.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 2
                CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 1
                CALIB_PROCESSING = {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None,
                                                                       'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 2
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = False
                MIN_STACKS = 0.15  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = True

            if self.collection_type == 'test':
                TEST_IMG_PATH = join(base_dir, 'images/calibration')
                TEST_BASE_STRING = 'calib_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = self.dataset + '_Meta-Test'
                TEST_RESULTS_PATH = join(base_dir, 'results/test')

                # test dataset information
                TEST_SUBSET = None
                TRUE_NUM_PARTICLES_PER_IMAGE = 150
                IF_TEST_IMAGE_STACK = 'subset'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [1, 2]
                TEST_PARTICLE_ID_IMAGE = 'calib_90.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 1
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 2
                TEST_PROCESSING = {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 2
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = False
                MIN_STACKS = 0.15  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = True
                ASSESS_SIMILARITY_FOR_ALL_STACKS = False

                # display options
                INSPECT_TEST_CONTOURS = False
                SHOW_PLOTS = False
                SAVE_PLOTS = True

        if self.dataset == '10X_1Xmag_2.15umNR_HighInt_0.12XHg':

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/10X_1Xmag_2.15umNR_HighInt_0.12XHg'

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 2.15e-6
            DEMAG = 1
            MAGNIFICATION = 10
            NA = 0.3
            Z_RANGE = 30
            FOCAL_LENGTH = 350
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 512
            PIXEL_DIM_Y = 512
            BKG_MEAN = 112
            BKG_NOISES = 6
            POINTS_PER_PIXEL = None
            N_RAYS = None
            GAIN = 1
            CYL_FOCAL_LENGTH = 0
            WAVELENGTH = 600e-9

            optics = GdpytSetup.optics(particle_diameter=PARTICLE_DIAMETER,
                                       demag=DEMAG,
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
                                       wavelength=WAVELENGTH,
                                       z_range=Z_RANGE)

            # image pre-processing
            DILATE = None  # None or True
            SHAPE_TOL = 0.5  # None == take any shape; 1 == take perfectly circular shape only.
            MIN_P_AREA = np.pi * optics.pixels_per_particle_in_focus ** 2  # ~ 5
            MAX_P_AREA = np.pi * np.max(optics.particle_diameter_z1 / 5) ** 2 * 0.6  # ~ 1600
            SAME_ID_THRESH = int(optics.pixels_per_particle_in_focus) * 6  # ~ 6
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 8


            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(base_dir, 'images/calibration')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = self.dataset + '_Calib'
                CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')

                # calib dataset information
                CALIB_SUBSET = [1, 78]
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 70
                IF_CALIB_IMAGE_STACK = 'subset'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, 1]
                BASELINE_IMAGE = 'calib_75.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 3
                CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                CALIB_PROCESSING = {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None,
                                                                       'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 2.5
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.1  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = True

            elif self.collection_type == 'test':
                TEST_IMG_PATH = join(base_dir, 'images/calibration')
                TEST_BASE_STRING = 'calib_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = self.dataset + '_Meta-Test'
                TEST_RESULTS_PATH = join(base_dir, 'results/test')

                # test dataset information
                TEST_SUBSET = [5, 75]
                TRUE_NUM_PARTICLES_PER_IMAGE = 70
                IF_TEST_IMAGE_STACK = 'subset'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [1, 2]
                TEST_PARTICLE_ID_IMAGE = 'calib_80.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 2
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                TEST_PROCESSING = {
                    TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 2.5
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.1  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = True
                ASSESS_SIMILARITY_FOR_ALL_STACKS = False

                # display options
                INSPECT_TEST_CONTOURS = False
                SHOW_PLOTS = False
                SAVE_PLOTS = True

        if self.dataset == '10X_1Xmag_5.1umNR_HighInt_0.06XHg':

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/10X_1Xmag_5.1umNR_HighInt_0.06XHg'

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 5.1e-6
            MAGNIFICATION = 10
            DEMAG = 1
            NA = 0.3
            Z_RANGE = 30
            FOCAL_LENGTH = 350
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 512
            PIXEL_DIM_Y = 512
            BKG_MEAN = 100
            BKG_NOISES = 5
            POINTS_PER_PIXEL = None
            N_RAYS = None
            GAIN = 1
            CYL_FOCAL_LENGTH = 0
            WAVELENGTH = 600e-9

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
                                       z_range=Z_RANGE)

            # image pre-processing
            DILATE = None  # None or True
            SHAPE_TOL = 0.5  # None == take any shape; 1 == take perfectly circular shape only.
            MIN_P_AREA = np.pi * optics.pixels_per_particle_in_focus ** 2 * 0.4 # ~ 12
            MAX_P_AREA = np.pi * np.max(optics.particle_diameter_z1 / 5) ** 2 * 0.5 # = 1750
            SAME_ID_THRESH = int(optics.pixels_per_particle_in_focus) * 2  # = 10
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 8

            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(base_dir, 'images/calibration')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = self.dataset + '_Calib'
                CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')

                # calib dataset information
                CALIB_SUBSET = [30, 90]
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 40
                IF_CALIB_IMAGE_STACK = 'subset'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, 3]
                BASELINE_IMAGE = 'calib_80.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 3
                CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                CALIB_PROCESSING = {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None,
                                                                       'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 2.25
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = False
                MIN_STACKS = 0.9  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = False

            elif self.collection_type == 'test':
                TEST_IMG_PATH = join(base_dir, 'images/calibration')
                TEST_BASE_STRING = 'calib_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = self.dataset + '_Meta-Test'
                TEST_RESULTS_PATH = join(base_dir, 'results/test')

                # test dataset information
                TEST_SUBSET = [50, 80]
                TRUE_NUM_PARTICLES_PER_IMAGE = 40
                IF_TEST_IMAGE_STACK = 'subset'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [5, 8]
                TEST_PARTICLE_ID_IMAGE = 'calib_80.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 2
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                TEST_PROCESSING = {
                    TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 2.25
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = False
                MIN_STACKS = 0.5  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = True
                ASSESS_SIMILARITY_FOR_ALL_STACKS = False

                # display options
                INSPECT_TEST_CONTOURS = False
                SHOW_PLOTS = False
                SAVE_PLOTS = True

        if self.dataset == '20X_0.5Xmag_2.15umRed_HighInt_0.03Hg':

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/20X_0.5Xmag_2.15umRed_HighInt_0.03Hg'

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 2.15e-6
            DEMAG = 0.5
            MAGNIFICATION = 20
            NA = 0.45
            Z_RANGE = 30
            FOCAL_LENGTH = 350
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 512
            PIXEL_DIM_Y = 512
            BKG_MEAN = 105
            BKG_NOISES = 6
            POINTS_PER_PIXEL = None
            N_RAYS = None
            GAIN = 1
            CYL_FOCAL_LENGTH = 0
            WAVELENGTH = 600e-9

            optics = GdpytSetup.optics(particle_diameter=PARTICLE_DIAMETER,
                                       demag=DEMAG,
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
                                       wavelength=WAVELENGTH,
                                       z_range=Z_RANGE)

            # image pre-processing
            DILATE = None  # None or True
            SHAPE_TOL = 0.5  # None == take any shape; 1 == take perfectly circular shape only.
            MIN_P_AREA = np.pi * optics.pixels_per_particle_in_focus ** 2 * 0.4 # ~ 12
            MAX_P_AREA = np.pi * np.max(optics.particle_diameter_z1 / 5) ** 2 * 0.5 # = 1750
            SAME_ID_THRESH = int(optics.pixels_per_particle_in_focus) * 2  # = 10
            BACKGROUND_SUBTRACTION = None

            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(base_dir, 'images/calibration')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = self.dataset + '_Calib'
                CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')

                # calib dataset information
                CALIB_SUBSET = [0, 65]
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 40
                IF_CALIB_IMAGE_STACK = 'subset'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, 1]
                BASELINE_IMAGE = 'calib_60.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 3
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                CALIB_PROCESSING = {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None,
                                                                       'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 2.5
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = False
                MIN_STACKS = 0.25  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = True

            elif self.collection_type == 'test':
                TEST_IMG_PATH = join(base_dir, 'images/calibration')
                TEST_BASE_STRING = 'calib_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = self.dataset + '_Meta-Test'
                TEST_RESULTS_PATH = join(base_dir, 'results/test')

                # test dataset information
                TEST_SUBSET = [30, 50]
                TRUE_NUM_PARTICLES_PER_IMAGE = 60
                IF_TEST_IMAGE_STACK = 'subset'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [1, 2]

                # test processing parameters
                TEST_TEMPLATE_PADDING = 2
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                TEST_PROCESSING = {
                    TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 2.25
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = False
                MIN_STACKS = 0.25  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = True
                ASSESS_SIMILARITY_FOR_ALL_STACKS = False

                # display options
                INSPECT_TEST_CONTOURS = False
                SHOW_PLOTS = False
                SAVE_PLOTS = True

        if self.dataset == '20X_1Xmag_0.87umNR':

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/20X_1Xmag_0.87umNR'

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 0.87e-6
            MAGNIFICATION = 20
            DEMAG = 1
            NA = 0.3
            Z_RANGE = 35
            FOCAL_LENGTH = 350
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 512
            PIXEL_DIM_Y = 512
            BKG_MEAN = 110
            BKG_NOISES = 7
            POINTS_PER_PIXEL = None
            N_RAYS = None
            GAIN = 1
            CYL_FOCAL_LENGTH = 0
            WAVELENGTH = 600e-9

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
                                       z_range=Z_RANGE)

            # image pre-processing
            DILATE = None  # None or True
            SHAPE_TOL = 0.5  # None == take any shape; 1 == take perfectly circular shape only.
            MIN_P_AREA = np.pi * optics.pixels_per_particle_in_focus ** 2 * 0.3 # ~ 12
            MAX_P_AREA = np.pi * np.max(optics.particle_diameter_z1 / 5) ** 2 * 0.65 # = 1750
            SAME_ID_THRESH = int(optics.pixels_per_particle_in_focus) * 2  # = 10
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 8

            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(base_dir, 'images/calibration')
                CALIB_BASE_STRING = 'nilered870nm_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = self.dataset + '_Calib'
                CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')

                # calib dataset information
                CALIB_SUBSET = None
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 40
                IF_CALIB_IMAGE_STACK = 'first'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = None
                BASELINE_IMAGE = 'nilered870nm_17.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 3
                CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                CALIB_PROCESSING = {'none': None} #  {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 7
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.9  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = True

            elif self.collection_type == 'test':
                TEST_IMG_PATH = join(base_dir, 'images/calibration')
                TEST_BASE_STRING = 'nilered870nm_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = self.dataset + '_Meta-Test'
                TEST_RESULTS_PATH = join(base_dir, 'results/test')

                # test dataset information
                TEST_SUBSET = None
                TRUE_NUM_PARTICLES_PER_IMAGE = 40
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = None
                TEST_PARTICLE_ID_IMAGE = 'nilered870nm_17.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 2
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                TEST_PROCESSING = {'none': None} #  {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 7
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = True
                ASSESS_SIMILARITY_FOR_ALL_STACKS = False

                # display options
                INSPECT_TEST_CONTOURS = False
                SHOW_PLOTS = False
                SAVE_PLOTS = True

        if self.dataset == '20X_1Xmag_0.87umNR_on_Silpuran':

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/20X_1Xmag_0.87umNR_on_Silpuran'

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 0.87e-6
            MAGNIFICATION = 20
            DEMAG = 1
            NA = 0.4
            Z_RANGE = 30
            FOCAL_LENGTH = 350
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 512
            PIXEL_DIM_Y = 512
            BKG_MEAN = 105
            BKG_NOISES = 8
            POINTS_PER_PIXEL = None
            N_RAYS = None
            GAIN = 1
            CYL_FOCAL_LENGTH = 0
            WAVELENGTH = 600e-9

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
                                       z_range=Z_RANGE)

            # image pre-processing
            DILATE = None  # None or True
            SHAPE_TOL = 0.5  # None == take any shape; 1 == take perfectly circular shape only.
            MIN_P_AREA = np.pi * optics.pixels_per_particle_in_focus ** 2 * 0.4 # ~ 12
            MAX_P_AREA = np.pi * np.max(optics.particle_diameter_z1 / 5) ** 2 * 0.5 # = 1750
            SAME_ID_THRESH = int(optics.pixels_per_particle_in_focus) * 2  # = 10
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 5

            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(base_dir, 'images/calibration')
                CALIB_BASE_STRING = 'nr870silpuran_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = self.dataset + '_Calib'
                CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')

                # calib dataset information
                CALIB_SUBSET = None
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 40
                IF_CALIB_IMAGE_STACK = 'first'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = None
                BASELINE_IMAGE = 'nr870silpuran_12.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 3
                CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 2
                CALIB_PROCESSING = {'none': None}  # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 2.25
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = False
                MIN_STACKS = 0.9  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = False

            elif self.collection_type == 'test':
                TEST_IMG_PATH = join(base_dir, 'images/calibration')
                TEST_BASE_STRING = 'nr870silpuran_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = self.dataset + '_Meta-Test'
                TEST_RESULTS_PATH = join(base_dir, 'results/test')

                # test dataset information
                TEST_SUBSET = None
                TRUE_NUM_PARTICLES_PER_IMAGE = 40
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = None
                TEST_PARTICLE_ID_IMAGE = 'nr870silpuran_12.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 2
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                TEST_PROCESSING = {'none': None} #  {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 2.25
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = False
                MIN_STACKS = 0.5  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = True
                ASSESS_SIMILARITY_FOR_ALL_STACKS = False

                # display options
                INSPECT_TEST_CONTOURS = False
                SHOW_PLOTS = False
                SAVE_PLOTS = True

        if self.dataset == '20X_1Xmag_2.15umNR_HighInt_0.03XHg':


            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/10X_1Xmag_5.1umNR_HighInt_0.06XHg'

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 5.1e-6
            MAGNIFICATION = 10
            NA = 0.3
            Z_RANGE = 30
            FOCAL_LENGTH = 350
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 512
            PIXEL_DIM_Y = 512
            BKG_MEAN = 100
            BKG_NOISES = 5
            POINTS_PER_PIXEL = None
            N_RAYS = None
            GAIN = 1
            CYL_FOCAL_LENGTH = 0
            WAVELENGTH = 600e-9

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
                                       wavelength=WAVELENGTH,
                                       z_range=Z_RANGE)

            # image pre-processing
            DILATE = None  # None or True
            SHAPE_TOL = 0.5  # None == take any shape; 1 == take perfectly circular shape only.
            MIN_P_AREA = np.pi * optics.pixels_per_particle_in_focus ** 2 * 0.4 # ~ 12
            MAX_P_AREA = np.pi * np.max(optics.particle_diameter_z1 / 5) ** 2 * 0.5 # = 1750
            SAME_ID_THRESH = int(optics.pixels_per_particle_in_focus) * 2  # = 10
            BACKGROUND_SUBTRACTION = None

            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(base_dir, 'images/calibration')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = self.dataset + '_Calib'
                CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')

                # calib dataset information
                CALIB_SUBSET = [45, 85]
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 40
                IF_CALIB_IMAGE_STACK = 'subset'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, 3]
                BASELINE_IMAGE = 'calib_80.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 3
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                CALIB_PROCESSING = {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None,
                                                                       'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 2.25
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.9  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = False

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = False

            elif self.collection_type == 'test':
                TEST_IMG_PATH = join(base_dir, 'images/calibration')
                TEST_BASE_STRING = 'calib_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = self.dataset + '_Meta-Test'
                TEST_RESULTS_PATH = join(base_dir, 'results/test')

                # test dataset information
                TEST_SUBSET = [50, 80]
                TRUE_NUM_PARTICLES_PER_IMAGE = 40
                IF_TEST_IMAGE_STACK = 'subset'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [5, 8]

                # test processing parameters
                TEST_TEMPLATE_PADDING = 2
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                TEST_PROCESSING = {
                    TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 2.25
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = False
                ASSESS_SIMILARITY_FOR_ALL_STACKS = False

                # display options
                INSPECT_TEST_CONTOURS = False
                SHOW_PLOTS = False
                SAVE_PLOTS = True

        if self.dataset == '20X_1Xmag_5.1umNR_HighInt_0.03XHg':

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/10X_1Xmag_5.1umNR_HighInt_0.06XHg'

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 5.1e-6
            MAGNIFICATION = 10
            NA = 0.3
            Z_RANGE = 30
            FOCAL_LENGTH = 350
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 512
            PIXEL_DIM_Y = 512
            BKG_MEAN = 100
            BKG_NOISES = 5
            POINTS_PER_PIXEL = None
            N_RAYS = None
            GAIN = 1
            CYL_FOCAL_LENGTH = 0
            WAVELENGTH = 600e-9

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
                                       wavelength=WAVELENGTH,
                                       z_range=Z_RANGE)

            # image pre-processing
            DILATE = None  # None or True
            SHAPE_TOL = 0.5  # None == take any shape; 1 == take perfectly circular shape only.
            MIN_P_AREA = np.pi * optics.pixels_per_particle_in_focus ** 2 * 0.4 # ~ 12
            MAX_P_AREA = np.pi * np.max(optics.particle_diameter_z1 / 5) ** 2 * 0.5 # = 1750
            SAME_ID_THRESH = int(optics.pixels_per_particle_in_focus) * 2  # = 10
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 8

            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(base_dir, 'images/calibration')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = self.dataset + '_Calib'
                CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')

                # calib dataset information
                CALIB_SUBSET = [45, 85]
                CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 40
                IF_CALIB_IMAGE_STACK = 'subset'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, 3]
                BASELINE_IMAGE = 'calib_80.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 3
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                CALIB_PROCESSING = {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None,
                                                                       'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 2.25
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = False
                MIN_STACKS = 0.9  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = False

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = False

            elif self.collection_type == 'test':
                TEST_IMG_PATH = join(base_dir, 'images/calibration')
                TEST_BASE_STRING = 'calib_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = self.dataset + '_Meta-Test'
                TEST_RESULTS_PATH = join(base_dir, 'results/test')

                # test dataset information
                TEST_SUBSET = [50, 80]
                TRUE_NUM_PARTICLES_PER_IMAGE = 40
                IF_TEST_IMAGE_STACK = 'subset'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [5, 8]
                TEST_PARTICLE_ID_IMAGE = 'calib_80.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 2
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                TEST_PROCESSING = {
                    TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 2.25
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = False
                MIN_STACKS = 0.5  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = False
                ASSESS_SIMILARITY_FOR_ALL_STACKS = False

                # display options
                INSPECT_TEST_CONTOURS = False
                SHOW_PLOTS = False
                SAVE_PLOTS = True

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == '10.07.21-BPE_Pressure_Deflection':

            # shared variables

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/tests/10.07.21-BPE_Pressure_Deflection'

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

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
            MIN_P_AREA = 25  # minimum particle size (area: units are in pixels) (recommended: 5)
            MAX_P_AREA = 22000  # maximum particle size (area: units are in pixels) (recommended: 200)
            SAME_ID_THRESH = 5  # maximum tolerance in x- and y-directions for particle to have the same ID between images
            OVERLAP_THRESHOLD = 0.1
            BACKGROUND_SUBTRACTION = None

            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(calib_base_dir, 'images/calibration')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = '10.07.21-BPE_Pressure_Deflection'
                CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')

                # calib dataset information
                CALIB_SUBSET = None
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 250
                IF_CALIB_IMAGE_STACK = 'first'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = None
                BASELINE_IMAGE = 'calib_40.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 3
                CALIB_CROPPING_SPECS = calib_cropping_specs # cropping_specs
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 2
                CALIB_PROCESSING_METHOD2 = 'gaussian'
                CALIB_PROCESSING_FILTER_TYPE2 = None
                CALIB_PROCESSING_FILTER_SIZE2 = 1
                CALIB_PROCESSING = None  # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                    # CALIB_PROCESSING_METHOD2: {'args': [], 'kwargs': dict(sigma=CALIB_PROCESSING_FILTER_SIZE2, preserve_range=True)}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = 140
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

            if self.collection_type == 'test':
                TEST_IMG_PATH = join(base_dir, 'images/test')
                TEST_BASE_STRING = 'test'
                TEST_GROUND_TRUTH_PATH =  join(base_dir, 'test_input')
                TEST_ID = '10.07.21_BPE_'
                TEST_RESULTS_PATH = join(base_dir, 'results/test')

                # test dataset information
                TEST_SUBSET = None
                TRUE_NUM_PARTICLES_PER_IMAGE = true_num_particles
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = []
                TEST_PARTICLE_ID_IMAGE = test_particle_id_image

                # test processing parameters
                TEST_TEMPLATE_PADDING = 1
                TEST_CROPPING_SPECS = cropping_specs
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 3
                TEST_PROCESSING_METHOD2 = 'gaussian'
                TEST_PROCESSING_FILTER_TYPE2 = None
                TEST_PROCESSING_FILTER_SIZE2 = 1
                TEST_PROCESSING = {'none': None}  # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                                  # TEST_PROCESSING_METHOD2: {'args': [], 'kwargs': dict(sigma=TEST_PROCESSING_FILTER_SIZE2, preserve_range=True)}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = threshold_modifier
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = False
                MIN_STACKS = 0.5  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5 # 0.75
                SUB_IMAGE_INTERPOLATION = True
                ASSESS_SIMILARITY_FOR_ALL_STACKS = False

                # display options
                INSPECT_TEST_CONTOURS = False
                SHOW_PLOTS = False
                SAVE_PLOTS = True

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.collection_type == 'calibration':
            calib_inputs = GdpytSetup.inputs(dataset=self.dataset,
                                             image_collection_type='calibration',
                                             image_path=CALIB_IMG_PATH,
                                             image_file_type=filetype,
                                             image_base_string=CALIB_BASE_STRING,
                                             calibration_z_step_size=CALIBRATION_Z_STEP_SIZE,
                                             single_particle_calibration=self.single_particle_calibration,
                                             overlapping_particles=self.particles_overlapping,
                                             image_subset=CALIB_SUBSET,
                                             baseline_image=BASELINE_IMAGE,
                                             hard_baseline=self.hard_baseline,
                                             static_templates=self.static_templates,
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

            return GdpytSetup.GdpytSetup(calib_inputs, calib_outputs, calib_processing, z_assessment=None,
                                                   optics=optics)

        elif self.collection_type == 'test':

            test_inputs = GdpytSetup.inputs(dataset=self.dataset,
                                            image_collection_type='test',
                                            image_path=TEST_IMG_PATH,
                                            image_file_type=filetype,
                                            image_base_string=TEST_BASE_STRING,
                                            baseline_image=TEST_PARTICLE_ID_IMAGE,
                                            hard_baseline=self.hard_baseline,
                                            static_templates=self.static_templates,
                                            overlapping_particles=self.particles_overlapping,
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
                                              inspect_contours=INSPECT_TEST_CONTOURS,
                                              assess_similarity_for_all_stacks=ASSESS_SIMILARITY_FOR_ALL_STACKS)

            test_processing = GdpytSetup.processing(min_layers_per_stack=MIN_STACKS,
                                                    cropping_params=TEST_CROPPING_SPECS,
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
                                                    overlap_threshold=OVERLAP_THRESHOLD,
                                                    same_id_threshold_distance=SAME_ID_THRESH,
                                                    stacks_use_raw=STACKS_USE_RAW,
                                                    zero_calib_stacks=ZERO_CALIB_STACKS,
                                                    zero_stacks_offset=ZERO_STACKS_OFFSET
                                                    )

            test_z_assessment = GdpytSetup.z_assessment(infer_method=INFER_METHODS,
                                                        min_cm=MIN_CM,
                                                        sub_image_interpolation=SUB_IMAGE_INTERPOLATION)

            return GdpytSetup.GdpytSetup(test_inputs, test_outputs, test_processing,
                                         z_assessment=test_z_assessment, optics=optics)

# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------