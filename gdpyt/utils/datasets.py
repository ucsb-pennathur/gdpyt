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
                 hard_baseline=False, particles_overlapping=False, sweep_method=None, sweep_param=None, known_z=None,
                 use_stack_id=None):

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
        self.known_z = known_z
        self.use_stack_id = use_stack_id  # use_stack_id

        if sweep_method == 'use_stack_id':
            self.use_stack_id = self.sweep_param

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
            nl = self.noise_level
            SINGLE_PARTICLE_CALIBRATION = self.single_particle_calibration
            XY_DISPLACEMENT = [[0, 0]]

            # organize noise-dependent variables
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
                test_particle_id_image = 'B0000.tif'
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
                    baseline_image = 'calib_-15.0.tif'  # overlap grid: 'calib_-36.962025316455694.tif'

                test_particle_id_image = 'B0000.tif'  # overlap grid: 'calib_-36.76767676767677.tif'
            else:
                raise ValueError("Noise level {} not in dataset (yet)".format(nl))

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/synthetic_overlap_noise-level{}'.format(nl)

            if self.particle_distribution == 'grid-random-z' and SINGLE_PARTICLE_CALIBRATION is False:
                calib_id = 'Grid-random-z_calib_nl{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid-random-z')
                base_dir = join(base_dir, 'grid-random-z')
                calib_cropping_specs = {'xmin': 0, 'xmax': 1024, 'ymin': 0, 'ymax': 1024, 'pad': 1}
                calib_baseline_image = baseline_image
                calib_true_num_particles = 170
                overlap_scaling = 5
                test_base_string = 'B0'
                true_num_particles = calib_true_num_particles
                cropping_specs = calib_cropping_specs
            elif self.particle_distribution == 'grid-random-z' and SINGLE_PARTICLE_CALIBRATION is True:
                calib_id = 'Grid-random-z_calib_nl{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid-random-z')
                base_dir = join(base_dir, 'grid-random-z')
                calib_cropping_specs = {'xmin': 0, 'xmax': 150, 'ymin': 0, 'ymax': 1024, 'pad': 1}
                calib_baseline_image = None
                calib_true_num_particles = 10
                overlap_scaling = 5
                test_base_string = 'B0'
                cropping_specs = {'xmin': 0, 'xmax': 1024, 'ymin': 0, 'ymax': 1024, 'pad': 1}
                true_num_particles = 170
            elif self.particle_distribution == 'grid-no-overlap-random-z' and SINGLE_PARTICLE_CALIBRATION is False:
                calib_id = 'Grid_calib_nl{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid-no-overlap-random-z')
                base_dir = join(base_dir, 'grid-no-overlap-random-z')
                calib_cropping_specs = {'xmin': 0, 'xmax': 1024, 'ymin': 0, 'ymax': 1024, 'pad': 1}
                baseline_image = 'calib_-15.0.tif'
                test_particle_id_image = 'B0000.tif'
                calib_baseline_image = baseline_image
                calib_true_num_particles = 100
                overlap_scaling = None
                test_base_string = 'B00'
                true_num_particles = calib_true_num_particles
                cropping_specs = calib_cropping_specs
            elif self.particle_distribution == 'grid-no-overlap-random-z' and SINGLE_PARTICLE_CALIBRATION is True:
                calib_id = 'Grid_calib_nl{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid-no-overlap-random-z')
                base_dir = join(base_dir, 'grid-no-overlap-random-z')
                calib_cropping_specs = {'xmin': 0, 'xmax': 150, 'ymin': 0, 'ymax': 1024, 'pad': 1}
                calib_baseline_image = None
                calib_true_num_particles = 10
                overlap_scaling = None
                test_base_string = 'B00'
                cropping_specs = {'xmin': 0, 'xmax': 1024, 'ymin': 0, 'ymax': 1024, 'pad': 1}
                true_num_particles = 100
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
                calib_true_num_particles = 1084
                calib_cropping_specs = {'xmin': 0, 'xmax': 1024, 'ymin': 0, 'ymax': 1024, 'pad': 1}
                calib_baseline_image = 'calib_-15.0.tif'
                baseline_image = 'calib_-15.0.tif'  # 'calib_-35.0.tif'
                test_particle_id_image = 'calib_-14.673366834170853.tif'  # 'calib_-35.175879396984925.tif'
                cropping_specs = calib_cropping_specs
                test_base_string = 'calib_'

            elif self.particle_distribution == 'random' and SINGLE_PARTICLE_CALIBRATION is True:
                calib_id = 'RandOverlapSPC_calib_pd{}_nl{}_'.format(self.particle_density, nl)
                calib_base_dir = join(base_dir, 'random', 'particle_density_1e-3')
                base_dir = join(base_dir, 'random', 'particle_density_' + self.particle_density)
                calib_cropping_specs = {'xmin': 623, 'xmax': 698, 'ymin': 186, 'ymax': 261, 'pad': 30}
                threshold_modifier = 575
                calib_baseline_image = None
                calib_true_num_particles = 1
                overlap_scaling = None
                test_particle_id_image = 'calib_-14.673366834170853.tif'
                cropping_specs = {'xmin': 0, 'xmax': 1024, 'ymin': 0, 'ymax': 1024, 'pad': 1}
                test_base_string = 'calib_'

            if self.particle_distribution == 'random':
                if self.particle_density == '1e-3':
                    true_num_particles = 104
                    test_particle_id_image = 'calib_-15.07537688442211.tif'
                elif self.particle_density == '2.5e-3':
                    true_num_particles = 366
                    test_particle_id_image = 'calib_-15.07537688442211.tif'
                elif self.particle_density == '5e-3':
                    true_num_particles = 524
                    test_particle_id_image = 'calib_-15.07537688442211.tif'
                elif self.particle_density == '7.5e-3':
                    true_num_particles = 786
                    test_particle_id_image = 'calib_-14.673366834170853.tif'
                elif self.particle_density == '10e-3':
                    true_num_particles = 1048
                    test_particle_id_image = 'calib_-15.07537688442211.tif'
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
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-SPC-{}-{}'.format(self.sweep_method, self.sweep_param))
                    else:
                        CALIB_ID = calib_id + 'Static_{}'.format(self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-static-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    CALIB_ID = calib_id
                    if self.single_particle_calibration is True:
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-SPC')
                    else:
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-static')
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
                CALIB_TEMPLATE_PADDING = 24
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
                        TEST_RESULTS_PATH = join(base_dir, 'results/test-SPC-{}-{}'.format(self.sweep_method, self.sweep_param))
                    else:
                        TEST_ID = '{}_test_static_nl{}_sweep{}_'.format(self.particle_distribution, nl, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir, 'results/test-static-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    TEST_ID = '{}_test_nl{}_'.format(self.particle_distribution, nl)
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = None
                TRUE_NUM_PARTICLES_PER_IMAGE = true_num_particles
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = []
                TEST_PARTICLE_ID_IMAGE = test_particle_id_image

                # test processing parameters
                if self.sweep_method == 'testpad':
                    TEST_TEMPLATE_PADDING = self.sweep_param
                else:
                    TEST_TEMPLATE_PADDING = 22
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
                SAVE_PLOTS = False

        if self.dataset == 'synthetic_experiment':

            XY_DISPLACEMENT = [[0, 0]]

            # organize noise-dependent variables
            bkg_mean = 120
            bkg_noise = 4
            calibration_z_step_size = 1.0
            if self.sweep_method == 'threshold':
                threshold_modifier = self.sweep_param
            else:
                threshold_modifier = bkg_mean + bkg_noise * 7

            baseline_image = 'calib_-15.0.tif'

            if self.particle_distribution == 'grid-uniform-z':
                test_particle_id_image = 'calib_-8.12346.tif'
            elif self.particle_distribution == 'grid-random-z':
                test_particle_id_image = 'B0000.tif'  # corresponds to calib_-15.tif from calibration images
            elif self.particle_distribution == 'grid-uniform-z-disp-x':
                test_particle_id_image = 'B0201.tif'  # corresponds to calib_-15.tif from calibration images


            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/synthetic_experiment/{}'.format(self.particle_distribution)

            if self.particle_distribution == 'grid-uniform-z':
                calib_id = 'grid-uniform-z_calib_'
                calib_base_dir = base_dir
                calib_cropping_specs = None
                calib_baseline_image = baseline_image
                calib_true_num_particles = 36
                overlap_scaling = None
                test_base_string = 'calib_'
                true_num_particles = calib_true_num_particles
                cropping_specs = calib_cropping_specs
            elif self.particle_distribution == 'grid-random-z':
                calib_id = 'grid-random-z_calib_'
                calib_base_dir = base_dir
                calib_cropping_specs = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 1}
                calib_baseline_image = baseline_image
                calib_true_num_particles = 100
                overlap_scaling = None
                test_base_string = 'B0'
                true_num_particles = calib_true_num_particles
                cropping_specs = calib_cropping_specs
            elif self.particle_distribution == 'grid-uniform-z-disp-x':
                calib_id = 'grid-uniform-z-disp-x_calib_'
                calib_base_dir = base_dir
                calib_cropping_specs = {'xmin': 0, 'xmax': 128, 'ymin': 0, 'ymax': 128, 'pad': 1}
                calib_baseline_image = baseline_image
                calib_true_num_particles = 1
                overlap_scaling = None
                test_base_string = 'B0'
                true_num_particles = calib_true_num_particles
                cropping_specs = calib_cropping_specs

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 2.15e-6  # 2.15e-6
            MAGNIFICATION = 20
            DEMAG = 0.5
            NA = 0.2  # Note: NA = 0.45 was used to generate the synthetic particle images
            FOCAL_LENGTH = 150
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 128  # 512
            PIXEL_DIM_Y = 128  # 512
            BKG_MEAN = bkg_mean
            BKG_NOISES = bkg_noise
            POINTS_PER_PIXEL = 20
            N_RAYS = 1500
            GAIN = 2
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
            MIN_P_AREA = 15  # minimum particle size (area: units are in pixels)
            MAX_P_AREA = 2000  # maximum particle size (area: units are in pixels) (recommended: 200)

            if self.sweep_method == 'same_id_thresh':
                SAME_ID_THRESH = self.sweep_param
            else:
                SAME_ID_THRESH = 8  # maximum tolerance in x- and y-directions for particle to have the same ID between images

            OVERLAP_THRESHOLD = 0.1
            BACKGROUND_SUBTRACTION = None

            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(calib_base_dir, 'calibration_images')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = join(calib_base_dir, 'calibration_input')

                if self.sweep_param is not None:
                    if self.single_particle_calibration is True:
                        CALIB_ID = calib_id + 'SPC_{}'.format(self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-SPC-{}-{}'.format(self.sweep_method, self.sweep_param))
                    else:
                        CALIB_ID = calib_id + 'Static_{}'.format(self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-static-{}-{}'.format(self.sweep_method, self.sweep_param))
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
                if self.single_particle_calibration is True:
                    CALIB_TEMPLATE_PADDING = 3
                else:
                    CALIB_TEMPLATE_PADDING = 13
                CALIB_CROPPING_SPECS = calib_cropping_specs  # cropping_specs
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                if self.sweep_method == 'filter':
                    CALIB_PROCESSING_FILTER_SIZE = self.sweep_param
                else:
                    CALIB_PROCESSING_FILTER_SIZE = 3
                CALIB_PROCESSING_METHOD2 = 'gaussian'
                CALIB_PROCESSING_FILTER_TYPE2 = None
                CALIB_PROCESSING_FILTER_SIZE2 = 1

                if self.sweep_method == 'filter' and self.sweep_param == 0:
                    CALIB_PROCESSING = {'none': None}
                elif self.sweep_method == 'filter':
                    CALIB_PROCESSING = {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                else:
                    CALIB_PROCESSING = {'none': None}

                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = threshold_modifier
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                if self.sweep_method == 'filter' and self.sweep_param == 0:
                    STACKS_USE_RAW = True
                elif self.sweep_method == 'filter':
                    STACKS_USE_RAW = False
                else:
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
                TEST_IMG_PATH = join(base_dir, 'xdisp{}'.format(self.sweep_param), 'test_images')
                TEST_BASE_STRING = test_base_string  # 'B00' or 'calib_'
                TEST_GROUND_TRUTH_PATH = join(base_dir, 'xdisp{}'.format(self.sweep_param), 'test_input')

                if self.sweep_param is not None:
                    if self.single_particle_calibration is True:
                        TEST_ID = '{}_test_SPC_sweep{}_'.format(self.particle_distribution, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir, 'results/test-SPC-{}-{}'.format(self.sweep_method, self.sweep_param))
                    else:
                        TEST_ID = '{}_test_static_sweep{}_'.format(self.particle_distribution, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir, 'results/test-static-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    TEST_ID = '{}_test_'.format(self.particle_distribution)
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = None
                TRUE_NUM_PARTICLES_PER_IMAGE = true_num_particles
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = []
                TEST_PARTICLE_ID_IMAGE = test_particle_id_image

                # test processing parameters
                if self.single_particle_calibration is True:
                    TEST_TEMPLATE_PADDING = 1
                else:
                    TEST_TEMPLATE_PADDING = 11
                TEST_CROPPING_SPECS = cropping_specs
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                if self.sweep_method == 'filter':
                    TEST_PROCESSING_FILTER_SIZE = self.sweep_param
                else:
                    TEST_PROCESSING_FILTER_SIZE = 3
                TEST_PROCESSING_METHOD2 = 'gaussian'
                TEST_PROCESSING_FILTER_TYPE2 = None
                TEST_PROCESSING_FILTER_SIZE2 = 1

                if self.sweep_method == 'filter' and self.sweep_param == 0:
                    TEST_PROCESSING = {'none': None}
                elif self.sweep_method == 'filter':
                    TEST_PROCESSING = {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                else:
                    TEST_PROCESSING = {'none': None}

                # TEST_PROCESSING_METHOD2: {'args': [], 'kwargs': dict(sigma=TEST_PROCESSING_FILTER_SIZE2, preserve_range=True)}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = threshold_modifier
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                if self.sweep_method == 'filter' and self.sweep_param == 0:
                    STACKS_USE_RAW = True
                elif self.sweep_method == 'filter':
                    STACKS_USE_RAW = False
                else:
                    STACKS_USE_RAW = True

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
            XY_DISPLACEMENT = [[0, 0]]

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 5.61e-6
            DEMAG = 1
            MAGNIFICATION = 20
            NA = 0.45
            FOCAL_LENGTH = 100
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
            SAME_ID_THRESH = 5  # maximum tolerance in x- and y-directions for particle to have the same ID between images
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 8

            if self.collection_type == 'calibration':

                    CALIB_IMG_PATH = join(base_dir, 'images/calibration')
                    CALIB_BASE_STRING = 'calib_'
                    CALIB_GROUND_TRUTH_PATH = None
                    CALIB_ID = self.dataset + '_Calib'

                    # ---
                    if self.sweep_param is not None:
                        if self.single_particle_calibration:
                            CALIB_ID = CALIB_ID + '_SPC_{}-{}'.format(self.sweep_method, self.sweep_param)
                            CALIB_RESULTS_PATH = join(base_dir,
                                                      'results/SPC-calibration-{}-{}'.format(self.sweep_method,
                                                                                             self.sweep_param))
                        else:
                            CALIB_ID = CALIB_ID + '_static_{}-{}'.format(self.sweep_method, self.sweep_param)
                            CALIB_RESULTS_PATH = join(base_dir,
                                                      'results/static-calibration-{}-{}'.format(self.sweep_method,
                                                                                                self.sweep_param))
                    else:
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                    if not os.path.exists(CALIB_RESULTS_PATH):
                        os.makedirs(CALIB_RESULTS_PATH)

                    # calib dataset information
                    if len(self.sweep_param) == 2:
                        CALIB_SUBSET = self.sweep_param[1]
                    else:
                        CALIB_SUBSET = None

                    CALIBRATION_Z_STEP_SIZE = 1.0
                    TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 13

                    if self.sweep_method == 'subset_mean':
                        if self.sweep_param[0] == 1:
                            IF_CALIB_IMAGE_STACK = 'first'
                            TAKE_CALIB_IMAGE_SUBSET_MEAN = []
                        else:
                            IF_CALIB_IMAGE_STACK = 'subset'
                            TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, self.sweep_param[0]]
                    else:
                        IF_CALIB_IMAGE_STACK = 'first'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = []
                    # ---

                    BASELINE_IMAGE = 'calib_42.tif'

                    # calibration processing parameters
                    CALIB_TEMPLATE_PADDING = 19
                    CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 200, 'pad': 2}
                    CALIB_PROCESSING_METHOD = 'median'
                    CALIB_PROCESSING_FILTER_TYPE = 'square'
                    CALIB_PROCESSING_FILTER_SIZE = int(np.round(optics.pixels_per_particle_in_focus / 3))
                    CALIB_PROCESSING = {'none': None}
                    # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                    CALIB_THRESHOLD_METHOD = 'manual'
                    CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 4
                    CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.5  # percent of calibration stack
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

                # ---
                if self.sweep_param is not None:
                    if self.single_particle_calibration:
                        TEST_ID = TEST_ID + '_SPC_{}-{}'.format(self.sweep_method, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir, 'results/SPC_test-{}-{}'.format(self.sweep_method, self.sweep_param))
                    else:
                        TEST_ID = TEST_ID + '_static_{}-{}'.format(self.sweep_method, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir, 'results/static_test-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                if len(self.sweep_param) == 2:
                    TEST_SUBSET = self.sweep_param[1]
                else:
                    TEST_SUBSET = None

                TRUE_NUM_PARTICLES_PER_IMAGE = 13

                if self.sweep_method == 'subset_mean':
                    if self.sweep_param == 1:
                        IF_TEST_IMAGE_STACK = 'subset'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = [8, 9]
                    else:
                        IF_TEST_IMAGE_STACK = 'last'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = [20, 21]
                else:
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = []
                # ---

                TEST_PARTICLE_ID_IMAGE = 'calib_42.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 16
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 200, 'pad': 2}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = int(np.round(optics.pixels_per_particle_in_focus / 3))
                TEST_PROCESSING = {'none': None}
                # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 4
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

        if self.dataset == '10X_0.5Xmag_2.15umNR_HighInt_0.03XHg':

            base_dir = join('/Users/mackenzie/Desktop/gdpyt-characterization/calibration', self.dataset)

            # shared variables
            XY_DISPLACEMENT = [[0, 0]]

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 2.15e-6
            DEMAG = 0.5
            MAGNIFICATION = 10
            NA = 0.3
            FOCAL_LENGTH = 2000
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
            Z_RANGE = 100

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
            SAME_ID_THRESH = 5  # maximum tolerance in x- and y-directions for particle to have the same ID between images
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 10  # 8

            if self.collection_type == 'calibration':
                CALIB_IMG_PATH = join(base_dir, 'images/calibration')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = self.dataset + '_calib'

                if self.sweep_param is not None:
                    if self.single_particle_calibration:
                        CALIB_ID = CALIB_ID + '_SPC_{}-{}'.format(self.sweep_method, self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/SPC-calibration-{}-{}'.format(self.sweep_method, self.sweep_param))
                    else:
                        CALIB_ID = CALIB_ID + '_static_{}-{}'.format(self.sweep_method, self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/static-calibration-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = [74, 180]
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 200

                if self.sweep_method == 'subset_mean':
                    if self.sweep_param == 1:
                        IF_CALIB_IMAGE_STACK = 'first'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = []
                    else:
                        IF_CALIB_IMAGE_STACK = 'subset'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, self.sweep_param]
                else:
                    IF_CALIB_IMAGE_STACK = 'first'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = []

                BASELINE_IMAGE = 'calib_75.tif'  # 'calib_90.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 10
                if self.single_particle_calibration:
                    CALIB_CROPPING_SPECS = {'xmin': 235, 'xmax': 310, 'ymin': 310, 'ymax': 430, 'pad': 15}
                else:
                    CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 15}
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 1
                CALIB_PROCESSING = {'none': None}
                # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 3
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
                TEST_IMG_PATH = join(base_dir, 'images/calibration')
                TEST_BASE_STRING = 'calib_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = self.dataset + '_Meta-Test'

                if self.sweep_param is not None:
                    if self.single_particle_calibration:
                        TEST_ID = TEST_ID + '_SPC_{}-{}'.format(self.sweep_method, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir, 'results/SPC_test-{}-{}'.format(self.sweep_method, self.sweep_param))
                    else:
                        TEST_ID = TEST_ID + '_static_{}-{}'.format(self.sweep_method, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir, 'results/static_test-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = [74, 180]
                TRUE_NUM_PARTICLES_PER_IMAGE = 200

                if self.sweep_method == 'subset_mean':
                    if self.sweep_param == 1:
                        IF_TEST_IMAGE_STACK = 'subset'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = [8, 9]
                    else:
                        IF_TEST_IMAGE_STACK = 'last'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = [20, 21]
                else:
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = []

                TEST_PARTICLE_ID_IMAGE = 'calib_75.tif'  # 'calib_90.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 7
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 15}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 2
                TEST_PROCESSING = {'none': None}  # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 4
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
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

        if self.dataset == '10X_1Xmag_2.15umNR_HighInt_0.12XHg':

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/10X_1Xmag_2.15umNR_HighInt_0.12XHg'

            # shared variables
            XY_DISPLACEMENT = [[0, 0]]

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
            OVERLAP_THRESHOLD = 10


            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(base_dir, 'images/calibration')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = self.dataset + '_calib'

                if self.sweep_param is not None:
                    if self.single_particle_calibration:
                        CALIB_ID = CALIB_ID + '_SPC_{}-{}'.format(self.sweep_method, self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/SPC-calibration-{}-{}'.format(self.sweep_method,
                                                                                                   self.sweep_param))
                    else:
                        CALIB_ID = CALIB_ID + '_static_{}-{}'.format(self.sweep_method, self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/static-calibration-{}-{}'.format(self.sweep_method,
                                                                                                      self.sweep_param))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                if len(self.sweep_param) == 2:
                    CALIB_SUBSET = self.sweep_param[1]
                else:
                    CALIB_SUBSET = None

                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 70

                if self.sweep_method == 'subset_mean':
                    if self.sweep_param[0] == 1:
                        IF_CALIB_IMAGE_STACK = 'first'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = []
                    else:
                        IF_CALIB_IMAGE_STACK = 'subset'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, self.sweep_param[0]]
                else:
                    IF_CALIB_IMAGE_STACK = 'first'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = []

                BASELINE_IMAGE = 'calib_75.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 14
                CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 20}
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                CALIB_PROCESSING = {'none': None}
                # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 3
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5  # percent of calibration stack
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
                TEST_ID = self.dataset + '_test'

                if self.sweep_param is not None:
                    if self.single_particle_calibration:
                        TEST_ID = TEST_ID + '_SPC_{}-{}'.format(self.sweep_method, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir,
                                                 'results/SPC_test-{}-{}'.format(self.sweep_method, self.sweep_param))
                    else:
                        TEST_ID = TEST_ID + '_static_{}-{}'.format(self.sweep_method, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir, 'results/static_test-{}-{}'.format(self.sweep_method,
                                                                                              self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                if len(self.sweep_param) == 2:
                    TEST_SUBSET = self.sweep_param[1]
                else:
                    TEST_SUBSET = None

                TRUE_NUM_PARTICLES_PER_IMAGE = 70

                if self.sweep_method == 'subset_mean':
                    if self.sweep_param[0] == 1:
                        IF_TEST_IMAGE_STACK = 'first'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = []
                    else:
                        IF_TEST_IMAGE_STACK = 'last'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = [20, 21]
                else:
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = []

                TEST_PARTICLE_ID_IMAGE = 'calib_80.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 12
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 20}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                TEST_PROCESSING = {'none': None}
                # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 3
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

        if self.dataset == '10X_1Xmag_5.1umNR_HighInt_0.06XHg':

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/10X_1Xmag_5.1umNR_HighInt_0.06XHg'

            # shared variables
            XY_DISPLACEMENT = [[0, 0]]

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 5.1e-6
            MAGNIFICATION = 10
            DEMAG = 1
            NA = 0.3
            Z_RANGE = 30
            FOCAL_LENGTH = 50
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
            SAME_ID_THRESH = int(optics.pixels_per_particle_in_focus)  # = 5
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 8

            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(base_dir, 'images/calibration')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = self.dataset + '_Calib'

                if self.sweep_param is not None:
                    if self.single_particle_calibration:
                        CALIB_ID = CALIB_ID + '_SPC_{}-{}'.format(self.sweep_method, self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/SPC-calibration-{}-{}'.format(self.sweep_method,
                                                                                                   self.sweep_param))
                    else:
                        CALIB_ID = CALIB_ID + '_static_{}-{}'.format(self.sweep_method, self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/static-calibration-{}-{}'.format(self.sweep_method,
                                                                                                      self.sweep_param))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                if len(self.sweep_param) == 2:
                    CALIB_SUBSET = self.sweep_param[1]
                else:
                    CALIB_SUBSET = None

                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 45

                if self.sweep_method == 'subset_mean':
                    if self.sweep_param[0] == 1:
                        IF_CALIB_IMAGE_STACK = 'first'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = []
                    else:
                        IF_CALIB_IMAGE_STACK = 'subset'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, self.sweep_param[0]]
                else:
                    IF_CALIB_IMAGE_STACK = 'first'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = []

                BASELINE_IMAGE = 'calib_92.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 10
                CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 20}
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 3  # int(np.ceil(optics.pixels_per_particle_in_focus))
                CALIB_PROCESSING = {'none': None}
                # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 3
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5  # percent of calibration stack
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

                if self.sweep_param is not None:
                    if self.single_particle_calibration:
                        TEST_ID = TEST_ID + '_SPC_{}-{}'.format(self.sweep_method, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir,
                                                 'results/SPC_test-{}-{}'.format(self.sweep_method, self.sweep_param))
                    else:
                        TEST_ID = TEST_ID + '_static_{}-{}'.format(self.sweep_method, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir, 'results/static_test-{}-{}'.format(self.sweep_method,
                                                                                              self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                if len(self.sweep_param) == 2:
                    TEST_SUBSET = self.sweep_param[1]
                else:
                    TEST_SUBSET = None

                TRUE_NUM_PARTICLES_PER_IMAGE = 40

                if self.sweep_method == 'subset_mean':
                    if self.sweep_param[0] == 1:
                        IF_TEST_IMAGE_STACK = 'first'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = []
                    else:
                        IF_TEST_IMAGE_STACK = 'last'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = [20, 21]
                else:
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = []

                TEST_PARTICLE_ID_IMAGE = 'calib_92.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 7
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 20}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 3  # int(np.ceil(optics.pixels_per_particle_in_focus))
                TEST_PROCESSING = {'none': None}
                # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 3
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

        if self.dataset == '20X_0.5Xmag_2.15umNR_HighInt_0.03XHg':

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/20X_0.5Xmag_2.15umNR_HighInt_0.03XHg'

            # shared variables
            XY_DISPLACEMENT = [[0, 0]]

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
            MIN_P_AREA = np.pi * optics.pixels_per_particle_in_focus ** 2 * 0.4  # ~ 12
            MAX_P_AREA = np.pi * np.max(optics.particle_diameter_z1 / 5) ** 2 * 0.5  # = 1750
            SAME_ID_THRESH = int(optics.pixels_per_particle_in_focus) * 5  # = 10
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 5

            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(base_dir, 'images/calibration')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = self.dataset + '_calib'

                if self.sweep_param is not None:
                    if self.single_particle_calibration:
                        CALIB_ID = CALIB_ID + '_SPC_{}-{}'.format(self.sweep_method, self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/SPC-calibration-{}-{}'.format(self.sweep_method,
                                                                                                   self.sweep_param))
                    else:
                        CALIB_ID = CALIB_ID + '_static_{}-{}'.format(self.sweep_method, self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/static-calibration-{}-{}'.format(self.sweep_method,
                                                                                                      self.sweep_param))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                if len(self.sweep_param) == 2:
                    CALIB_SUBSET = self.sweep_param[1]
                else:
                    CALIB_SUBSET = None

                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 40

                if self.sweep_method == 'subset_mean':
                    if self.sweep_param[0] == 1:
                        IF_CALIB_IMAGE_STACK = 'first'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = []
                    else:
                        IF_CALIB_IMAGE_STACK = 'subset'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, self.sweep_param[0]]
                else:
                    IF_CALIB_IMAGE_STACK = 'first'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = []

                BASELINE_IMAGE = 'calib_60.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 16
                CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 20}
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                CALIB_PROCESSING = {'none': None}
                # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 3
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5  # percent of calibration stack
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

                if self.sweep_param is not None:
                    if self.single_particle_calibration:
                        TEST_ID = TEST_ID + '_SPC_{}-{}'.format(self.sweep_method, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir,
                                                 'results/SPC_test-{}-{}'.format(self.sweep_method, self.sweep_param))
                    else:
                        TEST_ID = TEST_ID + '_static_{}-{}'.format(self.sweep_method, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir, 'results/static_test-{}-{}'.format(self.sweep_method,
                                                                                              self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                if len(self.sweep_param) == 2:
                    TEST_SUBSET = self.sweep_param[1]
                else:
                    TEST_SUBSET = None

                TRUE_NUM_PARTICLES_PER_IMAGE = 60

                if self.sweep_method == 'subset_mean':
                    if self.sweep_param[0] == 1:
                        IF_TEST_IMAGE_STACK = 'first'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = []
                    else:
                        IF_TEST_IMAGE_STACK = 'last'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = [20, 21]
                else:
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = []

                TEST_PARTICLE_ID_IMAGE = 'calib_60.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 14
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 20}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                TEST_PROCESSING = {'none': None}
                # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 3
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

        if self.dataset == '20X_1Xmag_0.87umNR':

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/20X_1Xmag_0.87umNR'

            # shared variables
            XY_DISPLACEMENT = [[0, 0]]

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
                CALIB_ID = self.dataset + '_calib'

                if self.sweep_param is not None:
                    if self.single_particle_calibration:
                        CALIB_ID = CALIB_ID + '_SPC_{}-{}'.format(self.sweep_method, self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/SPC-calibration-{}-{}'.format(self.sweep_method,
                                                                                                   self.sweep_param))
                    else:
                        CALIB_ID = CALIB_ID + '_static_{}-{}'.format(self.sweep_method, self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/static-calibration-{}-{}'.format(self.sweep_method,
                                                                                                      self.sweep_param))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                if len(self.sweep_param) == 2:
                    CALIB_SUBSET = self.sweep_param[1]
                else:
                    CALIB_SUBSET = None

                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 40

                if self.sweep_method == 'subset_mean':
                    if self.sweep_param[0] == 1:
                        IF_CALIB_IMAGE_STACK = 'first'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = []
                    else:
                        IF_CALIB_IMAGE_STACK = 'subset'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, self.sweep_param[0]]
                else:
                    IF_CALIB_IMAGE_STACK = 'first'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = []

                BASELINE_IMAGE = 'nilered870nm_17.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 6
                CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 20}
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                CALIB_PROCESSING = {'none': None}
                #  {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 5
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5  # percent of calibration stack
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

                if self.sweep_param is not None:
                    if self.single_particle_calibration:
                        TEST_ID = TEST_ID + '_SPC_{}-{}'.format(self.sweep_method, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir,
                                                 'results/SPC_test-{}-{}'.format(self.sweep_method, self.sweep_param))
                    else:
                        TEST_ID = TEST_ID + '_static_{}-{}'.format(self.sweep_method, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir, 'results/static_test-{}-{}'.format(self.sweep_method,
                                                                                              self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                if len(self.sweep_param) == 2:
                    TEST_SUBSET = self.sweep_param[1]
                else:
                    TEST_SUBSET = None

                TRUE_NUM_PARTICLES_PER_IMAGE = 40

                if self.sweep_method == 'subset_mean':
                    if self.sweep_param[0] == 1:
                        IF_TEST_IMAGE_STACK = 'first'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = []
                    else:
                        IF_TEST_IMAGE_STACK = 'last'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = [20, 21]
                else:
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = []

                TEST_PARTICLE_ID_IMAGE = 'nilered870nm_17.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 4
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 20}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                TEST_PROCESSING = {'none': None}
                #  {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 5
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

        if self.dataset == '11.06.21_z-micrometer-v2':

            base_dir = join('/Users/mackenzie/Desktop/gdpyt-characterization/experiments', self.dataset)

            if self.particle_distribution == 'SILPURAN':
                bkg_mean = 120
                bkg_noise = 4
                calib_img_path = join(base_dir, 'images/calibration/calib_1umSteps_microscope')
                calib_id = self.dataset + '_cSILPURAN'
                subset_i, subset_f = 0, 2

                if self.collection_type == 'meta-test':
                    test_img_path = calib_img_path
                    meta_test_id = self.dataset + '_calibration-meta-assessment_'
                    test_id = meta_test_id
                    metaset_i, metaset_f = subset_f, subset_f + 1
                    XY_DISPLACEMENT = [[0, 0]]

                if self.sweep_method == 'testset':
                    test_img_path = join(base_dir, 'images/tests/{}'.format(self.sweep_param[0]))
                    test_id = self.dataset + '_tcSILPURAN_'
                    XY_DISPLACEMENT = self.sweep_param[1]

            elif self.particle_distribution == 'Glass':
                bkg_mean = 100
                bkg_noise = 4
                calib_img_path = join(base_dir, 'images/calibration_microscope_1umSteps')
                calib_id = self.dataset + '_calibGlass'
                subset_i, subset_f = 0, 2
                test_img_path = join(base_dir, 'images/calibration_micrometer_5umSteps')
                test_id = self.dataset + '_testGlass'
                meta_test_id = self.dataset + '_calibration-meta-assessment_'
                metaset_i, metaset_f = 2, 3
            else:
                raise ValueError("Unknown particle distribution.")

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 2.15e-6
            DEMAG = 0.5
            MAGNIFICATION = 20
            NA = 0.3
            FOCAL_LENGTH = 500
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 512
            PIXEL_DIM_Y = 512
            BKG_MEAN = bkg_mean
            BKG_NOISES = bkg_noise
            POINTS_PER_PIXEL = None
            N_RAYS = None
            GAIN = 7
            CYL_FOCAL_LENGTH = 0
            WAVELENGTH = 600e-9
            Z_RANGE = 100

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
            MIN_P_AREA = 8  # minimum particle size (area: units are in pixels) (recommended: 5)
            MAX_P_AREA = 750     # (750) maximum particle size (area: units are in pixels) (recommended: 200)
            SAME_ID_THRESH = 5  # maximum tolerance in x- and y-directions for particle to have the same ID between images
            TEST_SAME_ID_THRESH = 12
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 5  # 8

            if self.collection_type == 'calibration':
                CALIB_IMG_PATH = calib_img_path
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = calib_id

                if self.sweep_param is not None:
                    CALIB_ID = CALIB_ID + '_{}'.format(self.sweep_param[0])  # '_{}-{}'.format(self.sweep_method, self.sweep_param)
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = None
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                BASELINE_IMAGE = 'calib_30.tif'  # 'calib_30.tif'

                # image stack averaging
                IF_CALIB_IMAGE_STACK = 'subset'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # averages the first two images (i.e. up to 2 but not including)
                """if self.sweep_method == 'subset_mean':
                    if self.sweep_param == 1:
                        IF_CALIB_IMAGE_STACK = 'first'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = []
                    else:
                        IF_CALIB_IMAGE_STACK = 'subset'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, self.sweep_param]
                else:
                    IF_CALIB_IMAGE_STACK = 'first'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, 1]"""

                # calibration processing parameters
                if self.static_templates:
                    CALIB_TEMPLATE_PADDING = 12  # if calib_30.tif, then 12; if calib_50.tif, then 15
                else:
                    CALIB_TEMPLATE_PADDING = 3

                if self.single_particle_calibration:
                    CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 5}
                    # {'xmin': 210, 'xmax': 275, 'ymin': 205, 'ymax': 260, 'pad': 0}
                else:
                    CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 5}

                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 3
                CALIB_PROCESSING_METHOD2 = 'gaussian'
                CALIB_PROCESSING_FILTER_TYPE2 = None
                CALIB_PROCESSING_FILTER_SIZE2 = 1
                CALIB_PROCESSING = {'none': None}
                # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}
                # CALIB_PROCESSING_METHOD2: {'args': [], 'kwargs': dict(sigma=CALIB_PROCESSING_FILTER_SIZE2, preserve_range=True)}
                # {'none': None}
                CALIB_THRESHOLD_METHOD = 'manual'
                if 'none' in list(CALIB_PROCESSING.keys()):
                    CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 3  # 3
                else:
                    CALIB_THRESHOLD_MODIFIER = optics.bkg_mean - optics.bkg_noise * 3
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.9  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = True

            if self.collection_type == 'test' or self.collection_type == 'meta-test':
                TEST_IMG_PATH = test_img_path
                TEST_BASE_STRING = 'test_X'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = test_id

                if self.sweep_param is not None:
                    TEST_ID = TEST_ID + '{}'.format(self.sweep_param)  # '{}-{}'.format(self.sweep_method, self.sweep_param)
                    TEST_RESULTS_PATH = join(base_dir, 'results/test-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = None
                TRUE_NUM_PARTICLES_PER_IMAGE = 90

                if self.sweep_method == 'subset_mean':
                    if self.sweep_param == 1:
                        IF_TEST_IMAGE_STACK = 'first'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = None
                    else:
                        IF_TEST_IMAGE_STACK = 'subset'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = [0, self.sweep_param]
                else:
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [1, 2]

                TEST_PARTICLE_ID_IMAGE = 'test_X001.tif'  # This is only used for 'testsets'; not meta-tests (below).

                # test processing parameters
                if self.static_templates:
                    TEST_TEMPLATE_PADDING = 10
                else:
                    TEST_TEMPLATE_PADDING = 1

                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 5}

                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 3
                TEST_PROCESSING_METHOD2 = 'gaussian'
                TEST_PROCESSING_FILTER_TYPE2 = None
                TEST_PROCESSING_FILTER_SIZE2 = 1
                TEST_PROCESSING = {'none': None}
                # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}} # ,
                # TEST_PROCESSING_METHOD2: {'args': [], 'kwargs': dict(sigma=TEST_PROCESSING_FILTER_SIZE2, preserve_range=True)}}

                TEST_THRESHOLD_METHOD = 'manual'
                if 'none' in list(TEST_PROCESSING.keys()):
                    TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 3
                else:
                    TEST_THRESHOLD_MODIFIER = optics.bkg_mean - optics.bkg_noise * 3
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.9  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
                SUB_IMAGE_INTERPOLATION = True
                ASSESS_SIMILARITY_FOR_ALL_STACKS = False

                # display options
                INSPECT_TEST_CONTOURS = False
                SHOW_PLOTS = False
                SAVE_PLOTS = True

            if self.collection_type == 'meta-test':
                TEST_BASE_STRING = 'calib_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 5}

                TEST_SUBSET = None
                IF_TEST_IMAGE_STACK = 'subset'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [metaset_i, metaset_f]
                TEST_PARTICLE_ID_IMAGE = 'calib_30.tif'  # was calib_30.tif

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == '11.09.21_z-micrometer-v3':

            base_dir = join('/Users/mackenzie/Desktop/gdpyt-characterization/experiments', self.dataset)

            if self.particle_distribution == 'SILPURAN':
                bkg_mean = 125
                bkg_noise = 4
                calib_img_path = join(base_dir, 'images/calibration/calib_5umSteps_microscope')
                calib_id = self.dataset + '_cSILPURAN'

                if self.sweep_method == 'testset':
                    test_img_path = join(base_dir, 'images/tests/{}'.format(self.sweep_param))
                    test_id = self.dataset + '_{}_'.format(self.sweep_param)
                    XY_DISPLACEMENT = [[0, 0]]
            else:
                raise ValueError("Unknown particle distribution.")

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 5.1e-6
            DEMAG = 1
            MAGNIFICATION = 10
            NA = 0.3
            FOCAL_LENGTH = 100
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 512
            PIXEL_DIM_Y = 512
            BKG_MEAN = bkg_mean
            BKG_NOISES = bkg_noise
            POINTS_PER_PIXEL = None
            N_RAYS = None
            GAIN = 1
            CYL_FOCAL_LENGTH = 0
            WAVELENGTH = 600e-9
            Z_RANGE = 100

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
            MIN_P_AREA = 10  # minimum particle size (area: units are in pixels) (recommended: 5)
            MAX_P_AREA = 1700     # maximum particle size (area: units are in pixels) (recommended: 200)
            SAME_ID_THRESH = 10  # maximum tolerance in x- and y-directions for particle to have the same ID between images
            TEST_SAME_ID_THRESH = 12
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 5  # 8

            if self.collection_type == 'calibration':
                CALIB_IMG_PATH = calib_img_path
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = calib_id

                if self.sweep_param is not None:
                    CALIB_ID = CALIB_ID + '_{}-{}'.format(self.sweep_method, self.sweep_param)
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = [0, 32]
                CALIBRATION_Z_STEP_SIZE = 5.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 330

                if self.sweep_method == 'subset_mean':
                    if self.sweep_param == 1:
                        IF_CALIB_IMAGE_STACK = 'first'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = []
                    else:
                        IF_CALIB_IMAGE_STACK = 'subset'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, self.sweep_param]
                else:
                    IF_CALIB_IMAGE_STACK = 'mean'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, 3]
                BASELINE_IMAGE = 'calib_30.tif'  # 'calib_90.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 15
                CALIB_CROPPING_SPECS = None # {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 1
                CALIB_PROCESSING = {'none': None}
                # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 4
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
                SAVE_CALIB_PLOTS = False

            if self.collection_type == 'test':
                TEST_IMG_PATH = test_img_path
                TEST_BASE_STRING = 'test_X'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = test_id

                if self.sweep_param is not None:
                    TEST_ID = TEST_ID + '{}-{}'.format(self.sweep_method, self.sweep_param)
                    TEST_RESULTS_PATH = join(base_dir, 'results/test-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = None
                TRUE_NUM_PARTICLES_PER_IMAGE = 330

                if self.sweep_method == 'subset_mean':
                    if self.sweep_param == 1:
                        IF_TEST_IMAGE_STACK = 'first'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = None
                    else:
                        IF_TEST_IMAGE_STACK = 'subset'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = [0, self.sweep_param]
                else:
                    IF_TEST_IMAGE_STACK = 'subset'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [1, 2]

                TEST_PARTICLE_ID_IMAGE = 'test_X1.tif'  # 'calib_90.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 3
                TEST_CROPPING_SPECS = None  # {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 2
                TEST_PROCESSING = {'none': None}  # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 6
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
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
                SAVE_PLOTS = False

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == '11.02.21-BPE_Pressure_Deflection_20X':

            base_dir = join('/Users/mackenzie/Desktop/gdpyt-characterization/experiments', self.dataset)

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 2.15e-6
            DEMAG = 1
            MAGNIFICATION = 20
            NA = 0.45
            FOCAL_LENGTH = 100
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 512
            PIXEL_DIM_Y = 512
            BKG_MEAN = 240
            BKG_NOISES = 20
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
            MIN_P_AREA = 8  # minimum particle size (area: units are in pixels) (recommended: 5)
            MAX_P_AREA = 1500     # maximum particle size (area: units are in pixels) (recommended: 200)
            SAME_ID_THRESH = 5  # maximum tolerance in x- and y-directions for particle to have the same ID between images
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 5  # 8

            if self.collection_type == 'calibration':
                CALIB_IMG_PATH = join(base_dir, 'images/calibration/calib1')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = self.dataset + '_calib1'

                if self.sweep_param is not None:
                    CALIB_ID = CALIB_ID + '_{}-{}'.format(self.sweep_method, self.sweep_param)
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration1-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = None
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 200

                if self.sweep_method == 'subset_mean':
                    if self.sweep_param == 1:
                        IF_CALIB_IMAGE_STACK = 'first'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = []
                    else:
                        IF_CALIB_IMAGE_STACK = 'subset'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, self.sweep_param]
                else:
                    IF_CALIB_IMAGE_STACK = 'first'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = []
                BASELINE_IMAGE = 'calib_35.tif'  # 'calib_90.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 12
                CALIB_CROPPING_SPECS = None # {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 1
                CALIB_PROCESSING = {'none': None}
                # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 1
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
                SAVE_CALIB_PLOTS = False

            if self.collection_type == 'test':
                TEST_IMG_PATH = join(base_dir, 'images/calibration/calib3')
                TEST_BASE_STRING = 'calib_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = self.dataset + 'calib1_assess_calib3'

                if self.sweep_param is not None:
                    TEST_ID = TEST_ID + 'calib3_{}-{}'.format(self.sweep_method, self.sweep_param)
                    TEST_RESULTS_PATH = join(base_dir, 'results/test-calib3-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = None
                TRUE_NUM_PARTICLES_PER_IMAGE = 200

                if self.sweep_method == 'subset_mean':
                    if self.sweep_param == 1:
                        IF_TEST_IMAGE_STACK = 'subset'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = [1, 2]
                    else:
                        IF_TEST_IMAGE_STACK = 'subset'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = [1, 1+self.sweep_param]
                else:
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = []

                TEST_PARTICLE_ID_IMAGE = 'calib_35.tif'  # 'calib_90.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 10
                TEST_CROPPING_SPECS = None  # {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 2
                TEST_PROCESSING = {'none': None}  # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 1
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
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


        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == '08.02.21 - bpe.g2 deflection':

            base_dir = join('/Users/mackenzie/Desktop/gdpyt-characterization/experiments', self.dataset)

            if self.particle_distribution == 'SILPURAN':
                bkg_mean = 120
                bkg_noise = 17
                calib_bkg_mean = 135
                calib_bkg_noise = 7
                calib_img_path = join(base_dir, 'images/calibration/08.03.21-20X-5.61umPink')
                calib_id = self.dataset + '_c08.03.21-20X-5.61umPink'
                subset_i, subset_f = 0, 20

                if self.collection_type == 'meta-test':
                    test_img_path = calib_img_path
                    meta_test_id = self.dataset + '_calibration-meta-assessment_'
                    test_id = meta_test_id
                    metaset_i, metaset_f = subset_f, subset_f + 1
                    XY_DISPLACEMENT = [[0, 0]]

                if self.sweep_method == 'testset':
                    test_img_path = join(base_dir, 'images/tests/ON_10s_OFF_15s/{}'.format(self.sweep_param))
                    test_id = self.dataset + '_tcSILPURAN_'
                    XY_DISPLACEMENT = [[0, 0]]

            else:
                raise ValueError("Unknown particle distribution.")

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 5.6e-6
            DEMAG = 1
            MAGNIFICATION = 20
            NA = 0.4
            FOCAL_LENGTH = 500
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 512
            PIXEL_DIM_Y = 512
            BKG_MEAN = bkg_mean
            BKG_NOISES = bkg_noise
            POINTS_PER_PIXEL = None
            N_RAYS = None
            GAIN = 7
            CYL_FOCAL_LENGTH = 0
            WAVELENGTH = 600e-9
            Z_RANGE = 100

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
            MIN_P_AREA = 8  # minimum particle size (area: units are in pixels) (recommended: 5)
            MAX_P_AREA = 1500     # (750) maximum particle size (area: units are in pixels) (recommended: 200)
            SAME_ID_THRESH = 10  # maximum tolerance in x- and y-directions for particle to have the same ID between images
            TEST_SAME_ID_THRESH = 12
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 5  # 8

            if self.collection_type == 'calibration':
                CALIB_IMG_PATH = calib_img_path
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = calib_id

                if self.sweep_param is not None:
                    CALIB_ID = CALIB_ID + '_{}'.format(self.sweep_param[0])  # '_{}-{}'.format(self.sweep_method, self.sweep_param)
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = [7, 75]
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                BASELINE_IMAGE = 'calib_42.tif'  # 'calib_30.tif'

                # image stack averaging
                IF_CALIB_IMAGE_STACK = 'subset'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # averages the first two images (i.e. up to 2 but not including)
                """if self.sweep_method == 'subset_mean':
                    if self.sweep_param == 1:
                        IF_CALIB_IMAGE_STACK = 'first'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = []
                    else:
                        IF_CALIB_IMAGE_STACK = 'subset'
                        TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, self.sweep_param]
                else:
                    IF_CALIB_IMAGE_STACK = 'first'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, 1]"""

                # calibration processing parameters
                if self.static_templates:
                    CALIB_TEMPLATE_PADDING = 14  # if calib_30.tif, then 12; if calib_50.tif, then 15
                else:
                    CALIB_TEMPLATE_PADDING = 3

                if self.single_particle_calibration:
                    CALIB_CROPPING_SPECS = {'xmin': 75, 'xmax': 450, 'ymin': 30, 'ymax': 480, 'pad': 1}
                    # {'xmin': 210, 'xmax': 275, 'ymin': 205, 'ymax': 260, 'pad': 0}
                else:
                    CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 5}

                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 3
                CALIB_PROCESSING_METHOD2 = 'gaussian'
                CALIB_PROCESSING_FILTER_TYPE2 = None
                CALIB_PROCESSING_FILTER_SIZE2 = 1
                CALIB_PROCESSING = {'none': None}
                # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}
                # CALIB_PROCESSING_METHOD2: {'args': [], 'kwargs': dict(sigma=CALIB_PROCESSING_FILTER_SIZE2, preserve_range=True)}
                # {'none': None}
                CALIB_THRESHOLD_METHOD = 'manual'
                if 'none' in list(CALIB_PROCESSING.keys()):
                    CALIB_THRESHOLD_MODIFIER = calib_bkg_mean + calib_bkg_noise * 3  # 3
                else:
                    CALIB_THRESHOLD_MODIFIER = optics.bkg_mean - optics.bkg_noise * 3
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.9  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = False

            if self.collection_type == 'test' or self.collection_type == 'meta-test':
                TEST_IMG_PATH = test_img_path
                TEST_BASE_STRING = 'test_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = test_id

                if self.sweep_param is not None:
                    TEST_ID = TEST_ID + '{}'.format(self.sweep_param)  # '{}-{}'.format(self.sweep_method, self.sweep_param)
                    TEST_RESULTS_PATH = join(base_dir, 'results/test-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = None
                TRUE_NUM_PARTICLES_PER_IMAGE = 15

                if self.sweep_method == 'subset_mean':
                    if self.sweep_param == 1:
                        IF_TEST_IMAGE_STACK = 'first'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = None
                    else:
                        IF_TEST_IMAGE_STACK = 'subset'
                        TAKE_TEST_IMAGE_SUBSET_MEAN = [0, self.sweep_param]
                else:
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [1, 2]

                TEST_PARTICLE_ID_IMAGE = 'test_000.tif'  # This is only used for 'testsets'; not meta-tests (below).

                # test processing parameters
                if self.static_templates:
                    TEST_TEMPLATE_PADDING = 10
                else:
                    TEST_TEMPLATE_PADDING = 1

                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 70, 'ymax': 512, 'pad': 5}

                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 3
                TEST_PROCESSING_METHOD2 = 'gaussian'
                TEST_PROCESSING_FILTER_TYPE2 = None
                TEST_PROCESSING_FILTER_SIZE2 = 1
                TEST_PROCESSING = {'none': None}
                # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}} # ,
                # TEST_PROCESSING_METHOD2: {'args': [], 'kwargs': dict(sigma=TEST_PROCESSING_FILTER_SIZE2, preserve_range=True)}}

                TEST_THRESHOLD_METHOD = 'manual'
                if 'none' in list(TEST_PROCESSING.keys()):
                    TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 3
                else:
                    TEST_THRESHOLD_MODIFIER = optics.bkg_mean - optics.bkg_noise * 3
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.9  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.5
                SUB_IMAGE_INTERPOLATION = True
                ASSESS_SIMILARITY_FOR_ALL_STACKS = False

                # display options
                INSPECT_TEST_CONTOURS = False
                SHOW_PLOTS = False
                SAVE_PLOTS = False

            if self.collection_type == 'meta-test':
                TEST_BASE_STRING = 'calib_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 5}

                TEST_SUBSET = None
                IF_TEST_IMAGE_STACK = 'subset'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [metaset_i, metaset_f]
                TEST_PARTICLE_ID_IMAGE = 'calib_30.tif'  # was calib_30.tif

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == '02.07.22_membrane_characterization':

            base_dir = join('/Users/mackenzie/Desktop/gdpyt-characterization/experiments', self.dataset)

            bkg_mean = 108
            bkg_noise = 3
            calib_img_path = join(base_dir, 'images/calibration')
            calib_id = self.dataset + '_calib'
            # subset_i, subset_f = 0, 10  # averages all 10 calibration images per z-step
            subset_i, subset_f = 0, self.sweep_param[0]

            if self.collection_type == 'meta-test':
                test_img_path = calib_img_path
                meta_test_id = self.dataset + '_calibration-meta-assessment_'
                test_id = meta_test_id
                subset_i, subset_f = 0, self.sweep_param[0]  # 9; averages the first 9 calibration images per z-step
                metaset_i, metaset_f = subset_f + self.sweep_param[1], subset_f + self.sweep_param[1] + 1
                XY_DISPLACEMENT = [[0, 0]]
            else:
                if self.sweep_method == 'testset':
                    if self.particle_distribution == 'pos':
                        test_img_path = join(base_dir, 'images/tests/pos/test_{}mm_pos'.format(self.sweep_param))
                    elif self.particle_distribution == 'dynamic':
                        test_img_path = join(base_dir, 'images/tests/dynamic/{}'.format(self.sweep_param[0]))
                    test_id = self.dataset + '_test_'
                    XY_DISPLACEMENT = [[0, 0]]

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # image pre-processing
            cropping_specs = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 30}
            DILATE = None  # None or True
            SHAPE_TOL = 0.25  # None == take any shape; 1 == take perfectly circular shape only.
            MIN_P_AREA = 25  # minimum particle size (area: units are in pixels) (recommended: 20)
            MAX_P_AREA = 750     # (750) maximum particle size (area: units are in pixels) (recommended: 200)
            SAME_ID_THRESH = 7  # maximum tolerance in x- and y-directions for particle to have the same ID between images
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 5

            # optics
            PARTICLE_DIAMETER = 2.15e-6
            DEMAG = 0.5
            MAGNIFICATION = 20
            NA = 0.3
            FOCAL_LENGTH = 500
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 512
            PIXEL_DIM_Y = 512
            BKG_MEAN = bkg_mean
            BKG_NOISES = bkg_noise
            POINTS_PER_PIXEL = None
            N_RAYS = None
            GAIN = 7
            CYL_FOCAL_LENGTH = 0
            WAVELENGTH = 600e-9
            Z_RANGE = 150

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

            if self.collection_type == 'calibration':
                CALIB_IMG_PATH = calib_img_path
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = calib_id

                if self.sweep_param is not None:
                    CALIB_ID = CALIB_ID + '_{}'.format(self.sweep_param)
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = [50, 60, 10]  # used [30, 59] for membrane curvature measurements
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                BASELINE_IMAGE = 'calib_58.tif'  # NOTES: ~'calib_57-60.tif' is the peak intensity image.

                # image stack averaging
                IF_CALIB_IMAGE_STACK = 'subset'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # averages up to subset_f but not including

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 18  # used 9 for membrane curvature measurements
                CALIB_CROPPING_SPECS = cropping_specs
                CALIB_PROCESSING_METHOD = 'flipud'  # 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 3
                CALIB_PROCESSING = {'none': None}  # {CALIB_PROCESSING_METHOD: {'args': []}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 10
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.9  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = True

            if self.collection_type == 'test' or self.collection_type == 'meta-test':
                TEST_IMG_PATH = test_img_path
                TEST_BASE_STRING = 'test_X'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = test_id

                if self.sweep_param is not None:
                    TEST_ID = TEST_ID + '{}'.format(self.sweep_param)  # '{}-{}'.format(self.sweep_method, self.sweep_param)
                    TEST_RESULTS_PATH = join(base_dir, 'results/test-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = [0, 500, 10]
                TRUE_NUM_PARTICLES_PER_IMAGE = 250

                # test processing parameters
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [1, 2]
                TEST_PARTICLE_ID_IMAGE = 'test_X1.tif'
                TEST_TEMPLATE_PADDING = 16  # used 6 for membrane curvature measurements
                TEST_CROPPING_SPECS = cropping_specs
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 3
                TEST_PROCESSING = {'none': None}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 10
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.9  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
                SUB_IMAGE_INTERPOLATION = True
                ASSESS_SIMILARITY_FOR_ALL_STACKS = False

                # display options
                INSPECT_TEST_CONTOURS = False
                SHOW_PLOTS = False
                SAVE_PLOTS = True

                if self.collection_type == 'meta-test':
                    TEST_BASE_STRING = 'calib_'
                    TEST_GROUND_TRUTH_PATH = None
                    TEST_CROPPING_SPECS = cropping_specs

                    TEST_SUBSET = None
                    IF_TEST_IMAGE_STACK = 'subset'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [metaset_i, metaset_f]
                    TEST_PARTICLE_ID_IMAGE = 'calib_58.tif'

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == 'zipper':

            base_dir = join('/Users/mackenzie/Desktop/gdpyt-characterization/experiments', self.dataset)

            bkg_mean = 350  # 262
            bkg_noise = 25  # 23
            calib_img_path = join(base_dir, 'images/calibration')
            calib_id = self.dataset + '_calib'
            subset_i, subset_f = 0, 10  # averages all 10 calibration images per z-step

            if self.collection_type == 'meta-test':
                test_img_path = calib_img_path
                meta_test_id = self.dataset + '_calibration-meta-assessment_'
                test_id = meta_test_id
                subset_i, subset_f = 0, 9  # averages the first 9 calibration images per z-step
                metaset_i, metaset_f = subset_f, subset_f + 1
                XY_DISPLACEMENT = [[0, 0]]
            else:
                if self.sweep_method == 'testset':
                    test_img_path = join(base_dir, 'images/tests/test_{}V'.format(self.sweep_param))
                    test_id = self.dataset + '_test_'
                    XY_DISPLACEMENT = [[0, 0]]

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # image pre-processing
            cropping_specs = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
            DILATE = None  # None or True
            SHAPE_TOL = 0.25  # None == take any shape; 1 == take perfectly circular shape only.
            MIN_P_AREA = 50  # minimum particle size (area: units are in pixels) (recommended: 20)
            MAX_P_AREA = 5000     # (750) maximum particle size (area: units are in pixels) (recommended: 200)
            SAME_ID_THRESH = 7  # maximum tolerance in x- and y-directions for particle to have the same ID between images
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 5

            # optics
            PARTICLE_DIAMETER = 5.61e-6
            DEMAG = 1
            MAGNIFICATION = 10
            NA = 0.3
            FOCAL_LENGTH = 500
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 512
            PIXEL_DIM_Y = 512
            BKG_MEAN = bkg_mean
            BKG_NOISES = bkg_noise
            POINTS_PER_PIXEL = None
            N_RAYS = None
            GAIN = 7
            CYL_FOCAL_LENGTH = 0
            WAVELENGTH = 600e-9
            Z_RANGE = 150

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

            if self.collection_type == 'calibration':
                CALIB_IMG_PATH = calib_img_path
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = calib_id

                if self.sweep_param is not None:
                    CALIB_ID = CALIB_ID + '_{}'.format(self.sweep_param)
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = None  # used [30, 59] for membrane curvature measurements
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                BASELINE_IMAGE = 'calib_30.tif'  # NOTES: ~'calib_57-60.tif' is the peak intensity image.

                # image stack averaging
                IF_CALIB_IMAGE_STACK = 'subset'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # averages up to subset_f but not including

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 8  # used 9 for membrane curvature measurements
                CALIB_CROPPING_SPECS = cropping_specs
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 3
                CALIB_PROCESSING = {'none': None}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 10
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.9  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = False

            if self.collection_type == 'test' or self.collection_type == 'meta-test':
                TEST_IMG_PATH = test_img_path
                TEST_BASE_STRING = 'test_X'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = test_id

                if self.sweep_param is not None:
                    TEST_ID = TEST_ID + '{}'.format(self.sweep_param)  # '{}-{}'.format(self.sweep_method, self.sweep_param)
                    TEST_RESULTS_PATH = join(base_dir, 'results/test-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = None
                TRUE_NUM_PARTICLES_PER_IMAGE = 250

                # test processing parameters
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [1, 2]
                TEST_PARTICLE_ID_IMAGE = 'test_X1.tif'
                TEST_TEMPLATE_PADDING = 6  # used 6 for membrane curvature measurements
                TEST_CROPPING_SPECS = cropping_specs
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 3
                TEST_PROCESSING = {'none': None}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 10
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.9  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
                SUB_IMAGE_INTERPOLATION = True
                ASSESS_SIMILARITY_FOR_ALL_STACKS = False

                # display options
                INSPECT_TEST_CONTOURS = False
                SHOW_PLOTS = False
                SAVE_PLOTS = True

                if self.collection_type == 'meta-test':
                    TEST_BASE_STRING = 'calib_'
                    TEST_GROUND_TRUTH_PATH = None
                    TEST_CROPPING_SPECS = cropping_specs

                    TEST_SUBSET = None
                    IF_TEST_IMAGE_STACK = 'subset'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [metaset_i, metaset_f]
                    TEST_PARTICLE_ID_IMAGE = 'calib_58.tif'

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

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

        elif self.collection_type == 'test' or self.collection_type == 'meta-test':

            if self.collection_type == 'meta-test':
                test_type = 'meta-test'
            else:
                test_type = 'test'

            test_inputs = GdpytSetup.inputs(dataset=self.dataset,
                                            image_collection_type=test_type,
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
                                            true_number_of_particles=TRUE_NUM_PARTICLES_PER_IMAGE,
                                            known_z=self.known_z)

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
                                                    zero_stacks_offset=ZERO_STACKS_OFFSET,
                                                    xy_displacement=XY_DISPLACEMENT
                                                    )

            test_z_assessment = GdpytSetup.z_assessment(infer_method=INFER_METHODS,
                                                        min_cm=MIN_CM,
                                                        sub_image_interpolation=SUB_IMAGE_INTERPOLATION,
                                                        use_stack_id=self.use_stack_id)

            return GdpytSetup.GdpytSetup(test_inputs, test_outputs, test_processing,
                                         z_assessment=test_z_assessment, optics=optics)

# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------