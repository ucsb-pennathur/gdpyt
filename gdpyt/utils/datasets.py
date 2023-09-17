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

        # shared variables
        filetype_ground_truth = '.txt'

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
            MIN_P_AREA = 15  # minimum particle size (area: units are in pixels) (recommended: 5)
            MAX_P_AREA = 5000  # maximum particle size (area: units are in pixels) (recommended: 200)
            SAME_ID_THRESH = 10  # 8  # maximum tolerance in x- and y-directions for particle to have the same ID between images
            OVERLAP_THRESHOLD = 0.1
            BACKGROUND_SUBTRACTION = None

            if self.collection_type == 'calibration':

                if len(str(self.number_of_images)) == 3:
                    leading_zeros = '00'
                else:
                    leading_zeros = '000'

                CALIB_IMG_PATH = join(base_dir,
                                      'Calibration/Calibration-noise-level{}/Calib-{}{}'.format(self.noise_level,
                                                                                                leading_zeros[:-1],
                                                                                                self.number_of_images))
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
                    CALIB_PROCESSING = {
                        CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
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
                TEST_ID = 'JP-EXF01-{}-pnum{}-nl{}'.format(self.particle_distribution, self.particle_density,
                                                           self.noise_level)
                TEST_RESULTS_PATH = join(base_dir, 'Results/{}'.format(self.particle_distribution))

                if self.particle_distribution == 'Dataset_I':
                    TEST_IMG_PATH = join(base_dir,
                                         '{}/Measurement-grid-noise-level{}/Images'.format(self.particle_distribution,
                                                                                           self.noise_level))
                    TEST_GROUND_TRUTH_PATH = join(base_dir, '{}/Measurement-grid-noise-level{}/Coordinates'.format(
                        self.particle_distribution, self.noise_level))

                elif self.particle_distribution == 'Dataset_II':
                    TEST_IMG_PATH = join(base_dir, '{}/Images/Part-per-image-{}'.format(self.particle_distribution,
                                                                                        self.particle_density))
                    TEST_GROUND_TRUTH_PATH = join(base_dir,
                                                  '{}/Coordinates/Part-per-image-{}'.format(self.particle_distribution,
                                                                                            self.particle_density))
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
                    TEST_PROCESSING = {
                        TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
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
                threshold_modifier = 575  # 3000  # 650  # 575  # 585-600 w/o filter
                baseline_image = 'calib_-10.0.tif'
                test_particle_id_image = 'B0000.tif'
            elif nl == 2:
                bkg_noise = 5
                calibration_z_step_size = 1.0
                if self.single_particle_calibration is True:
                    threshold_modifier = 135
                else:
                    threshold_modifier = 135
                baseline_image = 'calib_-4.0.tif'
                test_particle_id_image = 'test_000.tif'
            elif nl == 15:
                nl = 2
                bkg_noise = 15
                calibration_z_step_size = 1.0
                if self.single_particle_calibration is True:
                    threshold_modifier = 170
                    MIN_P_AREA = 12
                else:
                    threshold_modifier = 200  # 135
                    MIN_P_AREA = 3
                baseline_image = 'calib_-4.0.tif'
                test_particle_id_image = 'calib_-4.01804.tif'  # 'test_000.tif'
            else:
                raise ValueError("Noise level {} not in dataset (yet)".format(nl))

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/synthetic_overlap_noise-level{}'.format(
                nl)

            if self.particle_distribution == 'single' and SINGLE_PARTICLE_CALIBRATION is False:
                calib_id = 'single_calib_nll{}_'.format(nl)
                calib_base_dir = join(base_dir, 'single')
                base_dir = join(base_dir, 'single')
                calib_cropping_specs = None  # {'xmin': 0, 'xmax': 75, 'ymin': 0, 'ymax': 75, 'pad': 0}
                calib_baseline_image = baseline_image
                calib_true_num_particles = 1  # 440
                overlap_scaling = 1
                true_num_particles = 1  # 440
                test_base_string = 'calib_'
                cropping_specs = calib_cropping_specs
            elif self.particle_distribution == 'single' and SINGLE_PARTICLE_CALIBRATION is True:
                calib_id = 'grid-dz_calib_nll{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid-dz')
                base_dir = join(base_dir, 'grid-dz')
                calib_cropping_specs = None  # {'xmin': 0, 'xmax': 62, 'ymin': 0, 'ymax': 512, 'pad': 0}
                calib_baseline_image = baseline_image
                calib_true_num_particles = 12
                overlap_scaling = 1
                true_num_particles = 12
                test_base_string = 'test_'
                cropping_specs = {'xmin': 0, 'xmax': 1024, 'ymin': 0, 'ymax': 512, 'pad': 0}
            elif self.particle_distribution == 'grid-random-z' and SINGLE_PARTICLE_CALIBRATION is False:
                calib_id = 'Grid-random-z_calib_nl{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid-random-z')
                base_dir = join(base_dir, 'grid-random-z')
                calib_cropping_specs = {'xmin': 0, 'xmax': 1024, 'ymin': 0, 'ymax': 1024, 'pad': 0}
                calib_baseline_image = baseline_image
                calib_true_num_particles = 1  # 170
                overlap_scaling = 5
                test_base_string = 'B0'
                true_num_particles = calib_true_num_particles
                cropping_specs = calib_cropping_specs
            elif self.particle_distribution == 'grid-random-z' and SINGLE_PARTICLE_CALIBRATION is True:
                calib_id = 'Grid-random-z_calib_nl{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid-random-z')
                base_dir = join(base_dir, 'grid-random-z')
                calib_cropping_specs = {'xmin': 20, 'xmax': 150, 'ymin': 0, 'ymax': 1024, 'pad': 0}
                calib_baseline_image = None
                calib_true_num_particles = 1  # 10
                overlap_scaling = 5
                test_base_string = 'B0'
                cropping_specs = {'xmin': 0, 'xmax': 1024, 'ymin': 0, 'ymax': 1024, 'pad': 0}
                true_num_particles = 1  # 170
            elif self.particle_distribution == 'grid-no-overlap-random-z' and SINGLE_PARTICLE_CALIBRATION is False:
                calib_id = 'Grid_calib_nl{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid-no-overlap-random-z')
                base_dir = join(base_dir, 'grid-no-overlap-random-z')
                calib_cropping_specs = {'xmin': 0, 'xmax': 150, 'ymin': 0, 'ymax': 1024, 'pad': 0}
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
                calib_cropping_specs = {'xmin': 0, 'xmax': 150, 'ymin': 0, 'ymax': 1024, 'pad': 0}
                calib_baseline_image = None
                calib_true_num_particles = 10
                overlap_scaling = None
                test_base_string = 'B00'
                cropping_specs = {'xmin': 0, 'xmax': 1024, 'ymin': 0, 'ymax': 1024, 'pad': 0}
                true_num_particles = 10
            elif self.particle_distribution == 'grid' and SINGLE_PARTICLE_CALIBRATION is True:
                calib_id = 'grid_calib_nll{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid-dz')
                base_dir = join(base_dir, 'grid')
                calib_cropping_specs = {'xmin': 0, 'xmax': 60, 'ymin': 0, 'ymax': 256, 'pad': 0}
                calib_baseline_image = baseline_image
                calib_true_num_particles = 1
                overlap_scaling = 0.75
                true_num_particles = 1
                test_base_string = 'test_'
                cropping_specs = None
            elif self.particle_distribution == 'grid' and SINGLE_PARTICLE_CALIBRATION is False:
                calib_id = 'grid_calib_nll{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid-dz')
                base_dir = join(base_dir, 'grid')
                calib_cropping_specs = None
                calib_baseline_image = baseline_image
                calib_true_num_particles = 1
                overlap_scaling = 0.75
                true_num_particles = 1
                test_base_string = 'test_'
                cropping_specs = None
            elif self.particle_distribution == 'grid-wide' and SINGLE_PARTICLE_CALIBRATION is True:
                calib_id = 'grid_calib_nll{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid-dz-wide')
                base_dir = join(base_dir, 'grid-wide')
                calib_cropping_specs = {'xmin': 0, 'xmax': 60, 'ymin': 0, 'ymax': 256, 'pad': 0}
                calib_baseline_image = baseline_image
                calib_true_num_particles = 1
                overlap_scaling = 1.5
                true_num_particles = 1
                test_base_string = 'test_'
                cropping_specs = {'xmin': 0, 'xmax': 520, 'ymin': 0, 'ymax': 256, 'pad': 0}
            elif self.particle_distribution == 'grid-wide' and SINGLE_PARTICLE_CALIBRATION is False:
                calib_id = 'grid_calib_nll{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid-dz-wide')
                base_dir = join(base_dir, 'grid-wide')
                calib_cropping_specs = {'xmin': 0, 'xmax': 520, 'ymin': 0, 'ymax': 256, 'pad': 0}
                calib_baseline_image = baseline_image
                calib_true_num_particles = 1
                overlap_scaling = 1.5
                true_num_particles = 1
                test_base_string = 'test_'
                cropping_specs = calib_cropping_specs
            elif self.particle_distribution == 'grid-dz' and SINGLE_PARTICLE_CALIBRATION is False:
                calib_id = 'grid-dz_calib_nll{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid-dz')
                base_dir = join(base_dir, 'grid-dz')
                calib_cropping_specs = {'xmin': 0, 'xmax': 1024, 'ymin': 0, 'ymax': 512, 'pad': 0}
                calib_baseline_image = baseline_image
                calib_true_num_particles = 1  # 440
                overlap_scaling = 1
                true_num_particles = 1  # 440
                test_base_string = 'test_'
                cropping_specs = calib_cropping_specs
            elif self.particle_distribution == 'grid-dz' and SINGLE_PARTICLE_CALIBRATION is True:
                calib_id = 'grid-dz_calib_nll{}_'.format(nl)
                calib_base_dir = join(base_dir, 'grid-dz')
                base_dir = join(base_dir, 'grid-dz')
                calib_cropping_specs = {'xmin': 0, 'xmax': 62, 'ymin': 0, 'ymax': 512, 'pad': 0}
                calib_baseline_image = baseline_image
                calib_true_num_particles = 12
                overlap_scaling = 1
                true_num_particles = 12
                test_base_string = 'test_'
                cropping_specs = {'xmin': 0, 'xmax': 1024, 'ymin': 0, 'ymax': 512, 'pad': 0}
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
                calib_cropping_specs = {'xmin': 220, 'xmax': 580, 'ymin': 670, 'ymax': 1024,
                                        'pad': 10}  # {'xmin': 623, 'xmax': 698, 'ymin': 186, 'ymax': 261, 'pad': 30}
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
            if self.single_particle_calibration is True:
                test_template_padding = 0  # self.sweep_param  # 1
            else:
                test_template_padding = 15
            calib_image_subset = None  # [-19, 15, 1]
            test_image_subset = None  # [1701, 2000, 1]  # self.sweep_param

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 2.15e-6
            MAGNIFICATION = 10
            DEMAG = 1
            NA = 0.3
            FOCAL_LENGTH = 75
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16
            PIXEL_DIM_X = 75
            PIXEL_DIM_Y = 75
            BKG_MEAN = 100
            BKG_NOISES = bkg_noise
            POINTS_PER_PIXEL = 20
            N_RAYS = 500
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
            # MIN_P_AREA = 5  # minimum particle size (area: units are in pixels) (recommended: 5)
            MAX_P_AREA = 1100  # maximum particle size (area: units are in pixels) (recommended: 200)
            SAME_ID_THRESH = 5
            OVERLAP_THRESHOLD = 0.1
            BACKGROUND_SUBTRACTION = None

            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(calib_base_dir, 'calibration_images')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = join(calib_base_dir, 'calibration_input')

                if self.sweep_param is not None:
                    if self.single_particle_calibration is True:
                        CALIB_ID = calib_id + 'spct_{}'.format(self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-spct-{}-{}'.format(self.sweep_method,
                                                                                                    self.sweep_param))
                    else:
                        CALIB_ID = calib_id + 'idpt_{}'.format(self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-idpt-{}-{}'.format(self.sweep_method,
                                                                                                    self.sweep_param))
                else:
                    CALIB_ID = calib_id
                    if self.single_particle_calibration is True:
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-spct')
                    else:
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-idpt')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = calib_image_subset
                CALIBRATION_Z_STEP_SIZE = calibration_z_step_size
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = calib_true_num_particles
                IF_CALIB_IMAGE_STACK = 'first'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = []
                BASELINE_IMAGE = calib_baseline_image

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = test_template_padding + 5
                CALIB_CROPPING_SPECS = calib_cropping_specs  # cropping_specs
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 5
                CALIB_PROCESSING_METHOD2 = 'gaussian'
                CALIB_PROCESSING_FILTER_TYPE2 = None
                CALIB_PROCESSING_FILTER_SIZE2 = 1
                if self.single_particle_calibration is True:
                    CALIB_PROCESSING = {
                        'none': None}  # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                else:
                    CALIB_PROCESSING = {'none': None}
                # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = threshold_modifier
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = True

            if self.collection_type == 'test':
                TEST_IMG_PATH = join(base_dir, self.sweep_param, 'test_images')
                TEST_BASE_STRING = test_base_string  # test_base_string  # 'B00' or 'calib_'
                TEST_GROUND_TRUTH_PATH = join(base_dir, self.sweep_param, 'test_input')

                if self.sweep_param is not None:
                    if self.single_particle_calibration is True:
                        TEST_ID = '{}_test_SPC_nl{}_sweep{}_'.format(self.particle_distribution, nl, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir,
                                                 'results/test-SPC-{}-{}'.format(self.sweep_method, self.sweep_param))
                    else:
                        TEST_ID = '{}_test_idpt_nl{}_sweep{}_'.format(self.particle_distribution, nl, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir,
                                                 'results/test-idpt-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    TEST_ID = '{}_test_nl{}_'.format(self.particle_distribution, nl)
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = test_image_subset
                TRUE_NUM_PARTICLES_PER_IMAGE = true_num_particles
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = []
                TEST_PARTICLE_ID_IMAGE = test_particle_id_image

                # test processing parameters
                TEST_TEMPLATE_PADDING = test_template_padding
                TEST_CROPPING_SPECS = cropping_specs
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 5
                TEST_PROCESSING_METHOD2 = 'gaussian'
                TEST_PROCESSING_FILTER_TYPE2 = None
                TEST_PROCESSING_FILTER_SIZE2 = 1
                if self.single_particle_calibration is True:
                    TEST_PROCESSING = {
                        'none': None}  # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                else:
                    TEST_PROCESSING = {'none': None}
                # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                # TEST_PROCESSING_METHOD2: {'args': [], 'kwargs': dict(sigma=TEST_PROCESSING_FILTER_SIZE2, preserve_range=True)}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = threshold_modifier
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5  # percent of calibration stack
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

        if self.dataset == 'synthetic_example':

            XY_DISPLACEMENT = [[0, 0]]

            # organize noise-dependent variables
            nl = 2
            bkg_noise = 0
            calibration_z_step_size = 1.0
            threshold_modifier = 200
            MIN_P_AREA = 3
            baseline_image = 'calib_-2.5.tif'  # 'calib_-5.0.tif'
            test_particle_id_image = 'calib_-2.5.tif'  # 'calib_-5.78947.tif'  # 'test_000.tif'

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/synthetic_example'

            calib_id = 'ex_'
            calib_base_dir = base_dir
            base_dir = base_dir
            calib_cropping_specs = {'xmin': 180, 'xmax': 275, 'ymin': 320, 'ymax': 430, 'pad': 10}
            calib_baseline_image = baseline_image
            calib_true_num_particles = 1  # 440
            overlap_scaling = 1
            true_num_particles = 1  # 440
            test_base_string = 'calib_'
            cropping_specs = calib_cropping_specs

            test_template_padding = 21
            calib_image_subset = None  # [-19, 15, 1]
            test_image_subset = None  # [1701, 2000, 1]  # self.sweep_param

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 2.15e-6
            MAGNIFICATION = 10
            DEMAG = 1
            NA = 0.3
            FOCAL_LENGTH = 75
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 6.5
            PIXEL_DIM_X = 256
            PIXEL_DIM_Y = 256
            BKG_MEAN = 100
            BKG_NOISES = bkg_noise
            POINTS_PER_PIXEL = 20
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
            # MIN_P_AREA = 5  # minimum particle size (area: units are in pixels) (recommended: 5)
            MAX_P_AREA = 1100  # maximum particle size (area: units are in pixels) (recommended: 200)
            SAME_ID_THRESH = 20
            OVERLAP_THRESHOLD = 0.1
            BACKGROUND_SUBTRACTION = None

            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(calib_base_dir, 'test_images')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = join(calib_base_dir, 'test_input')

                if self.sweep_param is not None:
                    if self.single_particle_calibration is True:
                        CALIB_ID = calib_id + 'spct_{}'.format(self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-spct-{}-{}'.format(self.sweep_method,
                                                                                                    self.sweep_param))
                    else:
                        CALIB_ID = calib_id + 'idpt_{}'.format(self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-idpt-{}-{}'.format(self.sweep_method,
                                                                                                    self.sweep_param))
                else:
                    CALIB_ID = calib_id
                    if self.single_particle_calibration is True:
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-spct')
                    else:
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-idpt')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = calib_image_subset
                CALIBRATION_Z_STEP_SIZE = calibration_z_step_size
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = calib_true_num_particles
                IF_CALIB_IMAGE_STACK = 'first'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = []
                BASELINE_IMAGE = calib_baseline_image

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = test_template_padding + 5
                CALIB_CROPPING_SPECS = calib_cropping_specs  # cropping_specs
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 5
                CALIB_PROCESSING_METHOD2 = 'gaussian'
                CALIB_PROCESSING_FILTER_TYPE2 = None
                CALIB_PROCESSING_FILTER_SIZE2 = 1
                CALIB_PROCESSING = {'none': None}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = threshold_modifier
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = True

            if self.collection_type == 'test':
                TEST_IMG_PATH = join(base_dir, 'test_images')
                TEST_BASE_STRING = test_base_string  # test_base_string  # 'B00' or 'calib_'
                TEST_GROUND_TRUTH_PATH = join(base_dir, 'test_input')

                if self.sweep_param is not None:
                    if self.single_particle_calibration is True:
                        TEST_ID = '{}_test_SPC_nl{}_sweep{}_'.format(self.particle_distribution, nl, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir,
                                                 'results/test-SPC-{}-{}'.format(self.sweep_method, self.sweep_param))
                    else:
                        TEST_ID = '{}_test_idpt_nl{}_sweep{}_'.format(self.particle_distribution, nl, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir,
                                                 'results/test-idpt-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    TEST_ID = '{}_test_nl{}_'.format(self.particle_distribution, nl)
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = test_image_subset
                TRUE_NUM_PARTICLES_PER_IMAGE = true_num_particles
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = []
                TEST_PARTICLE_ID_IMAGE = test_particle_id_image

                # test processing parameters
                TEST_TEMPLATE_PADDING = test_template_padding
                TEST_CROPPING_SPECS = cropping_specs
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 5
                TEST_PROCESSING_METHOD2 = 'gaussian'
                TEST_PROCESSING_FILTER_TYPE2 = None
                TEST_PROCESSING_FILTER_SIZE2 = 1
                TEST_PROCESSING = {'none': None}
                # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                # TEST_PROCESSING_METHOD2: {'args': [], 'kwargs': dict(sigma=TEST_PROCESSING_FILTER_SIZE2, preserve_range=True)}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = threshold_modifier
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5  # percent of calibration stack
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

        elif self.dataset == 'synthetic_experiment':

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

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/synthetic_experiment/{}'.format(
                self.particle_distribution)

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
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-SPC-{}-{}'.format(self.sweep_method,
                                                                                                   self.sweep_param))
                    else:
                        CALIB_ID = calib_id + 'Static_{}'.format(self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-static-{}-{}'.format(self.sweep_method,
                                                                                                      self.sweep_param))
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
                    CALIB_PROCESSING = {
                        CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
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
                        TEST_RESULTS_PATH = join(base_dir,
                                                 'results/test-SPC-{}-{}'.format(self.sweep_method, self.sweep_param))
                    else:
                        TEST_ID = '{}_test_static_sweep{}_'.format(self.particle_distribution, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir, 'results/test-static-{}-{}'.format(self.sweep_method,
                                                                                              self.sweep_param))
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
                    TEST_PROCESSING = {
                        TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
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
            NA = 0.4
            FOCAL_LENGTH = 50
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
            MIN_P_AREA = 45  # minimum particle size (area: units are in pixels) (recommended: 5)
            MAX_P_AREA = 2500  # maximum particle size (area: units are in pixels) (recommended: 200)
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
                CALIB_SUBSET = None
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 13
                IF_CALIB_IMAGE_STACK = 'subset'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, 25]
                # ---

                BASELINE_IMAGE = 'calib_42.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 3  # 19
                CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 2}
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 3
                CALIB_PROCESSING = {'none': None}
                # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 2
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

            elif self.collection_type == 'test' or self.collection_type == 'meta-test':
                TEST_IMG_PATH = join(base_dir, 'images/calibration')
                TEST_BASE_STRING = 'calib_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = self.dataset + '_Meta-Test'

                # ---
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
                TEST_SUBSET = None
                TRUE_NUM_PARTICLES_PER_IMAGE = 13
                IF_TEST_IMAGE_STACK = 'subset'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 10]
                TEST_PARTICLE_ID_IMAGE = 'calib_42.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 1  # 16
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 2}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 3
                TEST_PROCESSING = {'none': None}
                # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = optics.bkg_mean + optics.bkg_noise * 2
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
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
            MAX_P_AREA = 2000  # maximum particle size (area: units are in pixels) (recommended: 200)
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
                TEST_PROCESSING = {
                    'none': None}  # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
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
            BKG_MEAN = 102
            BKG_NOISES = 5
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
            DILATE = None
            SHAPE_TOL = 0.5
            MIN_P_AREA = 10
            MAX_P_AREA = 1075
            SAME_ID_THRESH = 8
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 10

            # shared variables
            calibration_z_step_size = 1
            baseline_image = 'calib_075.tif'
            threshold_modifier = optics.bkg_mean + optics.bkg_noise * 3
            threshold_step_size = 1
            image_subset = None

            theoretical_threshold = {'theory': threshold_modifier,
                                     'thresh_min': optics.bkg_mean + optics.bkg_noise * 6,
                                     'thresh_max': optics.bkg_mean * 3.5,
                                     'frame_max': 75,
                                     'dz_per_frame': threshold_step_size,
                                     }

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

                CALIBRATION_Z_STEP_SIZE = calibration_z_step_size
                CALIB_SUBSET = image_subset
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                IF_CALIB_IMAGE_STACK = 'mean'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, 10]
                BASELINE_IMAGE = baseline_image

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 1
                CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 0}
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                CALIB_PROCESSING = None  # {'none': None}
                # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = threshold_modifier
                CALIB_THRESHOLD_PARAMS = theoretical_threshold  # {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = True

            elif self.collection_type == 'test':
                TEST_IMG_PATH = join(base_dir, 'images/calibration')
                TEST_BASE_STRING = 'calib_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = self.dataset + '_meta'

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
                TEST_SUBSET = image_subset
                TRUE_NUM_PARTICLES_PER_IMAGE = 1
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = []
                TEST_PARTICLE_ID_IMAGE = baseline_image

                # test processing parameters
                TEST_TEMPLATE_PADDING = 0
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 0}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = int(np.ceil(optics.pixels_per_particle_in_focus))
                TEST_PROCESSING = None  # {'none': None}
                # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = threshold_modifier
                TEST_THRESHOLD_PARAMS = theoretical_threshold  # {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5
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
            MIN_P_AREA = 40
            MAX_P_AREA = 900
            SAME_ID_THRESH = 5
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
                CALIB_SUBSET = [25, 151, 2]
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 45
                IF_CALIB_IMAGE_STACK = 'subset'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, 10]
                BASELINE_IMAGE = 'calib_92.tif'
                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 4  # 10
                CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 0}
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 3
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
                MIN_CM = 0.0
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = True

            elif self.collection_type == 'test' or self.collection_type == 'meta-test':
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
                TEST_SUBSET = [26, 145, 2]
                TRUE_NUM_PARTICLES_PER_IMAGE = 40
                IF_TEST_IMAGE_STACK = 'subset'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 10]
                TEST_PARTICLE_ID_IMAGE = 'calib_92.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 2  # 7
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 0}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 3
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
                MIN_CM = 0.0
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
            NA = 0.3
            Z_RANGE = 120
            FOCAL_LENGTH = 50
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
            MIN_P_AREA = 8
            MAX_P_AREA = 2500
            SAME_ID_THRESH = 5
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
                CALIB_SUBSET = None
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 60

                IF_CALIB_IMAGE_STACK = 'subset'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, 10]

                BASELINE_IMAGE = 'calib_60.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 3  # 16
                CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 0}
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 3
                CALIB_PROCESSING = {'none': None}
                # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual_percent'
                CALIB_THRESHOLD_MODIFIER = 0.99  # optics.bkg_mean + optics.bkg_noise * 3
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = True

            elif self.collection_type == 'test' or self.collection_type == 'meta-test':
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
                TEST_SUBSET = [15, 93, 1]
                TRUE_NUM_PARTICLES_PER_IMAGE = 60
                IF_TEST_IMAGE_STACK = 'subset'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 10]
                TEST_PARTICLE_ID_IMAGE = 'calib_60.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 1  # 14
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 0}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 3
                TEST_PROCESSING = {'none': None}
                # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual_percent'
                TEST_THRESHOLD_MODIFIER = 0.99  # optics.bkg_mean + optics.bkg_noise  # 0.99
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.9  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
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

            # image pre-processing
            DILATE = None  # None or True
            SHAPE_TOL = 0.5  # None == take any shape; 1 == take perfectly circular shape only.
            SAME_ID_THRESH = 5
            OVERLAP_THRESHOLD = 8

            # optics
            PARTICLE_DIAMETER = 0.87e-6
            MAGNIFICATION = 20
            DEMAG = 1
            NA = 0.45
            Z_RANGE = 50
            FOCAL_LENGTH = 350
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 512
            PIXEL_DIM_Y = 512
            BKG_MEAN = 122
            BKG_NOISES = 7
            BKG_MAX = 144
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

            # shared variables
            MIN_P_AREA = 3
            MAX_P_AREA = 850
            BACKGROUND_SUBTRACTION = None
            cropping_specs = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 0}
            # a single good particle: {'xmin': 80, 'xmax': 120, 'ymin': 360, 'ymax': 390, 'pad': 0}
            image_subset = None  # [1, 37, 5]
            test_image_subset = None  # [1, 100, 5]
            baseline_image = 'calib_17.tif'
            threshold_modifier = BKG_MAX + 1
            idpt_template_padding = 4

            theoretical_threshold = {'theory': threshold_modifier,
                                     'thresh_min': BKG_MEAN + BKG_NOISES,  # 108,  # 130
                                     'thresh_max': BKG_MEAN * 2,  # 220,
                                     'frame_max': 17 * 1,  # 50
                                     'dz_per_frame': 1,
                                     }

            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(base_dir, 'images/calibration')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = self.dataset + '_calib'

                if self.sweep_param is not None:
                    if self.single_particle_calibration:
                        CALIB_ID = CALIB_ID + '_spct_{}-{}'.format(self.sweep_method, self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/spct-calibration-{}-{}'.format(self.sweep_method,
                                                                                                    self.sweep_param))
                    else:
                        CALIB_ID = CALIB_ID + '_idpt_{}-{}'.format(self.sweep_method, self.sweep_param)
                        CALIB_RESULTS_PATH = join(base_dir, 'results/idpt-calibration-{}-{}'.format(self.sweep_method,
                                                                                                    self.sweep_param))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = image_subset
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                IF_CALIB_IMAGE_STACK = 'mean'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, 1]
                BASELINE_IMAGE = baseline_image

                # calibration processing parameters
                if self.single_particle_calibration:
                    CALIB_TEMPLATE_PADDING = 1
                else:
                    CALIB_TEMPLATE_PADDING = idpt_template_padding + 3

                CALIB_CROPPING_SPECS = cropping_specs
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 3
                CALIB_PROCESSING = None
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = threshold_modifier - 3
                CALIB_THRESHOLD_PARAMS = theoretical_threshold
                # {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = True

            elif self.collection_type == 'test' or self.collection_type == 'meta-test':
                TEST_IMG_PATH = join(base_dir, 'images/calibration')
                TEST_BASE_STRING = 'calib_'  # 'nilered870nm_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = self.dataset + '_meta'

                if self.sweep_param is not None:
                    if self.single_particle_calibration:
                        TEST_ID = TEST_ID + '_spct_{}-{}'.format(self.sweep_method, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir,
                                                 'results/spct_test-{}-{}'.format(self.sweep_method, self.sweep_param))
                    else:
                        TEST_ID = TEST_ID + '_idpt_{}-{}'.format(self.sweep_method, self.sweep_param)
                        TEST_RESULTS_PATH = join(base_dir, 'results/idpt_test-{}-{}'.format(self.sweep_method,
                                                                                            self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = test_image_subset
                TRUE_NUM_PARTICLES_PER_IMAGE = 1
                IF_TEST_IMAGE_STACK = 'mean'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 1]
                TEST_PARTICLE_ID_IMAGE = baseline_image

                # test processing parameters
                if self.single_particle_calibration:
                    TEST_TEMPLATE_PADDING = 0
                else:
                    TEST_TEMPLATE_PADDING = idpt_template_padding
                TEST_CROPPING_SPECS = cropping_specs
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 3
                TEST_PROCESSING = None
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = threshold_modifier
                TEST_THRESHOLD_PARAMS = theoretical_threshold
                # {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5
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

        if self.dataset == '20X_1Xmag_0.87umNR_on_Silpuran':

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/20X_1Xmag_0.87umNR_on_Silpuran'

            # shared variables
            XY_DISPLACEMENT = [[0, 0]]

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 0.87e-6
            MAGNIFICATION = 20
            DEMAG = 1
            NA = 0.45
            Z_RANGE = 30
            FOCAL_LENGTH = 200
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
            MIN_P_AREA = 10
            MAX_P_AREA = 400
            SAME_ID_THRESH = 4
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 8
            cropping_specs = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 5}

            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(base_dir, 'images/calibration')
                CALIB_BASE_STRING = 'nr870silpuran_'
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
                CALIB_SUBSET = None
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 40
                IF_CALIB_IMAGE_STACK = 'subset'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, 100]
                BASELINE_IMAGE = 'nr870silpuran_12.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 3
                CALIB_CROPPING_SPECS = cropping_specs
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 3
                CALIB_PROCESSING = {'none': None}
                # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                CALIB_THRESHOLD_METHOD = 'manual_percent'
                CALIB_THRESHOLD_MODIFIER = 0.99  # optics.bkg_mean + optics.bkg_noise * 2.25
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.9  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = True

            elif self.collection_type == 'test' or self.collection_type == 'meta-test':
                TEST_IMG_PATH = join(base_dir, 'images/calibration')
                TEST_BASE_STRING = 'nr870silpuran_'
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
                TEST_SUBSET = None
                TRUE_NUM_PARTICLES_PER_IMAGE = 40
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = None
                TEST_PARTICLE_ID_IMAGE = 'nr870silpuran_12.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 1
                TEST_CROPPING_SPECS = cropping_specs
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 3
                TEST_PROCESSING = {'none': None}
                # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual_percent'
                TEST_THRESHOLD_MODIFIER = 0.99  # optics.bkg_mean + optics.bkg_noise * 2.25
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.9  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
                SUB_IMAGE_INTERPOLATION = True
                ASSESS_SIMILARITY_FOR_ALL_STACKS = False

                # display options
                INSPECT_TEST_CONTOURS = False
                SHOW_PLOTS = False
                SAVE_PLOTS = True

        if self.dataset == '20X_1Xmag_2.15umNR_HighInt_0.03XHg':

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/20X_1Xmag_2.15umNR_HighInt_0.03XHg'

            # shared variables
            if self.particle_distribution == 'glass':
                bkg_mean = 95
                bkg_noise = 4
                calib_img_path = join(base_dir, 'images/calibration')
                calib_id = self.dataset + '_cGlass'
                subset_i, subset_f = 0, 10

                cropping_specs = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 0}

                if self.collection_type == 'meta-test':
                    test_img_path = calib_img_path
                    meta_test_id = self.dataset + '_meta_'
                    test_id = meta_test_id
                    metaset_i, metaset_f = 0, 10  # subset_f, subset_f + 1
                    XY_DISPLACEMENT = [[0, 0]]

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 2.15e-6
            MAGNIFICATION = 20
            DEMAG = 1
            NA = 0.45
            FOCAL_LENGTH = 50
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
            Z_RANGE = 80

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
            MIN_P_AREA = 8
            MAX_P_AREA = 2250
            SAME_ID_THRESH = 10  # maximum tolerance in x- and y-directions for particle to have the same ID between images
            TEST_SAME_ID_THRESH = 10
            OVERLAP_THRESHOLD = 5
            BACKGROUND_SUBTRACTION = None  # 'min_value'

            baseline_image = 'calib_35.tif'
            threshold_modifier = bkg_mean + bkg_noise * 3
            threshold_step_size = 1
            image_subset = None  # [5, 75, threshold_step_size]

            theoretical_threshold = {'theory': threshold_modifier,
                                     'thresh_min': bkg_mean + bkg_noise * 4,  # 108,  # 130
                                     'thresh_max': bkg_mean * 3,  # 220,
                                     'frame_max': 35,  # 50
                                     'dz_per_frame': threshold_step_size,
                                     }

            if self.collection_type == 'calibration':
                CALIB_IMG_PATH = calib_img_path
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = calib_id

                if self.sweep_param is not None:
                    CALIB_ID = CALIB_ID + '_{}'.format(self.sweep_param)
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-{}-{}'.format(self.sweep_method,
                                                                                           self.sweep_param))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = image_subset  # [10, 70, 3]
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                BASELINE_IMAGE = baseline_image  # 'calib_30.tif'

                # image stack averaging
                IF_CALIB_IMAGE_STACK = 'mean'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # averages up to 2 but not including)

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 1
                CALIB_CROPPING_SPECS = cropping_specs
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 3
                CALIB_PROCESSING_METHOD2 = 'gaussian'
                CALIB_PROCESSING_FILTER_TYPE2 = None
                CALIB_PROCESSING_FILTER_SIZE2 = 1
                CALIB_PROCESSING = None
                # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                # {'none': None}

                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = 120
                CALIB_THRESHOLD_PARAMS = theoretical_threshold
                # {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = True

            elif self.collection_type == 'test' or self.collection_type == 'meta-test':
                TEST_IMG_PATH = join(base_dir, 'images/calibration')
                TEST_BASE_STRING = 'calib_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = self.dataset + '_Meta-Test'

                if self.sweep_param is not None:
                    TEST_ID = TEST_ID + '_{}'.format(self.sweep_param)
                    TEST_RESULTS_PATH = join(base_dir, 'results/test-{}-{}'.format(self.sweep_method,
                                                                                   self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')

                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = None
                TRUE_NUM_PARTICLES_PER_IMAGE = 1
                IF_TEST_IMAGE_STACK = 'mean'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 3]

                # test processing parameters
                TEST_TEMPLATE_PADDING = 1
                TEST_CROPPING_SPECS = cropping_specs
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 3
                TEST_PROCESSING = None
                # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = 120

                TEST_THRESHOLD_PARAMS = theoretical_threshold
                # {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
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
                    IF_TEST_IMAGE_STACK = 'mean'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [metaset_i, metaset_f]
                    TEST_PARTICLE_ID_IMAGE = 'calib_35.tif'

        if self.dataset == '20X_1Xmag_5.1umNR_HighInt_0.03XHg':

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/20X_1Xmag_5.1umNR_HighInt_0.03XHg'

            # shared variables
            XY_DISPLACEMENT = [[0, 0]]

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # optics
            PARTICLE_DIAMETER = 5.1e-6
            MAGNIFICATION = 20
            DEMAG = 1
            NA = 0.45
            Z_RANGE = 100
            FOCAL_LENGTH = 100
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
            MIN_P_AREA = 45
            MAX_P_AREA = 3000
            SAME_ID_THRESH = 5
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 8

            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(base_dir, 'images/calibration')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = self.dataset + '_Calib'

                if self.sweep_param is not None:
                    CALIB_ID = CALIB_ID + '_{}'.format(self.sweep_param)
                    CALIB_RESULTS_PATH = join(base_dir,
                                              'results/calibration-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = [1, 76, 3]
                CALIB_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 0}
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 40
                IF_CALIB_IMAGE_STACK = 'subset'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, 10]
                BASELINE_IMAGE = 'calib_33.tif'

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 3
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 3
                CALIB_PROCESSING = {
                    CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                # {'none': None}
                #
                CALIB_THRESHOLD_METHOD = 'manual_percent'
                CALIB_THRESHOLD_MODIFIER = 0.75  # optics.bkg_mean + optics.bkg_noise * 2.25
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.85  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = True

            elif self.collection_type == 'test' or self.collection_type == 'meta-test':
                TEST_IMG_PATH = join(base_dir, 'images/calibration')
                TEST_BASE_STRING = 'calib_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = self.dataset + '_Meta-Test'

                if self.sweep_param is not None:
                    TEST_ID = TEST_ID + '{}'.format(
                        self.sweep_param)  # '{}-{}'.format(self.sweep_method, self.sweep_param)
                    TEST_RESULTS_PATH = join(base_dir, 'results/test-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = None
                TRUE_NUM_PARTICLES_PER_IMAGE = 40
                IF_TEST_IMAGE_STACK = 'subset'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 3]
                TEST_PARTICLE_ID_IMAGE = 'calib_33.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = 1
                TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 0}
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 3
                TEST_PROCESSING = {
                    TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                # {'none': None}
                # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual_percent'
                TEST_THRESHOLD_MODIFIER = 0.25  # optics.bkg_mean + optics.bkg_noise * 2.25
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.0  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0  # TODO: should be derived or defined by a plane of interest
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
                    TEST_CROPPING_SPECS = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 0}

                    TEST_SUBSET = None
                    IF_TEST_IMAGE_STACK = 'subset'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 5]
                    TEST_PARTICLE_ID_IMAGE = 'calib_33.tif'

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == '10.07.21-BPE_Pressure_Deflection':

            base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/10.07.21-BPE_Pressure_Deflection_20X'

            # shared variables
            XY_DISPLACEMENT = [[0, 0]]

            # file types
            filetype = '.tif'

            # optics
            PARTICLE_DIAMETER = 2.15e-6
            MAGNIFICATION = 20
            DEMAG = 0.5
            NA = 0.35
            FOCAL_LENGTH = 350
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 512
            PIXEL_DIM_Y = 512
            BKG_MEAN = 117
            BKG_NOISES = 5
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
            MIN_P_AREA = 3  # 8  # minimum particle size (area: units are in pixels) (recommended: 5)
            MAX_P_AREA = 200  # for IDPT, use 40; max measured = 55 for calib_41.tif
            SAME_ID_THRESH = 4  # maximum tolerance in x- and y-directions for particle to have the same ID between images
            OVERLAP_THRESHOLD = 0.1
            BACKGROUND_SUBTRACTION = None  # 'min_value'

            cropping_specs = {'xmin': 100, 'xmax': 360, 'ymin': 50, 'ymax': 190,
                              'pad': 0}  # {'xmin': 150, 'xmax': 370, 'ymin': 0, 'ymax': 512, 'pad': 0}
            test_cropping_specs = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 0}
            image_subset = [36, 71, 1]
            test_template_padding = 0
            calib_template_padding = 2
            thresholding = 'manual'  # 'manual_percent' 'median_percent' 'manual'
            threshold_value = BKG_MEAN + BKG_NOISES * 5  # most recent IDPT: BKG_MEAN * BKG_NOISES * 1
            threshold_modifier = 0.3
            baseline_image = 'calib_41.tif'

            if self.collection_type == 'calibration':

                CALIB_IMG_PATH = join(base_dir, 'images/calibration')
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = '10.07.21-BPE_Pressure_Deflection'

                if self.sweep_param is not None:
                    CALIB_ID = CALIB_ID + '_{}'.format(
                        self.sweep_param)  # '_{}-{}'.format(self.sweep_method, self.sweep_param)
                    CALIB_RESULTS_PATH = join(base_dir,
                                              'results/calibration-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = image_subset
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                IF_CALIB_IMAGE_STACK = 'mean'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, 10]
                BASELINE_IMAGE = baseline_image

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = calib_template_padding  # test_template_padding + 3  # test_template_padding + 2  # 14 # z=75, ct >= 11, z=70, calib temp >= 8
                CALIB_CROPPING_SPECS = cropping_specs  # cropping_specs
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 2
                CALIB_PROCESSING_METHOD2 = 'gaussian'
                CALIB_PROCESSING_FILTER_TYPE2 = None
                CALIB_PROCESSING_FILTER_SIZE2 = 1
                CALIB_PROCESSING = {'none': None}
                # {'none': None}
                # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                # CALIB_PROCESSING_METHOD2: {'args': [], 'kwargs': dict(sigma=CALIB_PROCESSING_FILTER_SIZE2, preserve_range=True)}}
                CALIB_THRESHOLD_METHOD = thresholding
                CALIB_THRESHOLD_MODIFIER = threshold_modifier
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [threshold_value]}  # , CALIB_THRESHOLD_MODIFIER
                # {CALIB_THRESHOLD_METHOD: [threshold_value, CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.975  # percent of calibration stack
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

                if self.sweep_method == 'test-set':
                    TEST_IMG_PATH = join(base_dir, 'images/test/pos/{}'.format(self.sweep_param[0]))
                else:
                    TEST_IMG_PATH = join(base_dir, 'images/test')
                TEST_BASE_STRING = 'test_'
                TEST_GROUND_TRUTH_PATH = None  # join(base_dir, 'test_input')
                TEST_ID = '10.07.21_BPE_'

                if self.sweep_param is not None:
                    TEST_ID = TEST_ID + '{}'.format(
                        self.sweep_param)  # '{}-{}'.format(self.sweep_method, self.sweep_param)
                    TEST_RESULTS_PATH = join(base_dir, 'results/test-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = [-1, 25, 1]
                TRUE_NUM_PARTICLES_PER_IMAGE = 1
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = []
                TEST_PARTICLE_ID_IMAGE = 'test_-001.tif'

                # test processing parameters
                TEST_TEMPLATE_PADDING = test_template_padding  # 11
                TEST_CROPPING_SPECS = test_cropping_specs
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 3
                TEST_PROCESSING_METHOD2 = 'gaussian'
                TEST_PROCESSING_FILTER_TYPE2 = None
                TEST_PROCESSING_FILTER_SIZE2 = 1
                TEST_PROCESSING = {'none': None}
                # {'none': None}
                # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                # TEST_PROCESSING_METHOD2: {'args': [], 'kwargs': dict(sigma=TEST_PROCESSING_FILTER_SIZE2, preserve_range=True)}}
                TEST_THRESHOLD_METHOD = thresholding
                TEST_THRESHOLD_MODIFIER = threshold_modifier
                test_thresh = BKG_MEAN + BKG_NOISES * 9
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [test_thresh]}  # , TEST_THRESHOLD_MODIFIER
                # {TEST_THRESHOLD_METHOD: [threshold_value, TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.975  # percent of calibration stack
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
                    TEST_IMG_PATH = join(base_dir, 'images/calibration')
                    TEST_BASE_STRING = 'calib_'
                    TEST_GROUND_TRUTH_PATH = None
                    TEST_CROPPING_SPECS = cropping_specs

                    TEST_SUBSET = image_subset
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 1]
                    TEST_PARTICLE_ID_IMAGE = baseline_image

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == '11.06.21_z-micrometer-v2':

            base_dir = join('/Users/mackenzie/Desktop/gdpyt-characterization/experiments', self.dataset)

            """
            IMPORTANT NOTE:
            Calibrate using 1-micron steps: calib_1umSteps_microscope
            Calibrate using 5-micron steps: calib_5umSteps_micrometer
            """

            if self.particle_distribution == 'SILPURAN':
                bkg_mean = 100
                bkg_noise = 4
                calib_img_path = join(base_dir, 'images/calibration/calib_1umSteps_microscope')
                calib_id = self.dataset + '_1umSteps_'
                subset_i, subset_f = 0, 3

                if self.collection_type == 'meta-test':
                    test_img_path = calib_img_path  # join(base_dir, 'images/calibration_noise', self.sweep_param)
                    meta_test_id = self.dataset + '_Meta_'
                    test_id = meta_test_id
                    metaset_i, metaset_f = 0, 1  # subset_f, subset_f + 1
                    XY_DISPLACEMENT = [[0, 0]]

                if self.sweep_method == 'testset':
                    test_img_path = join(base_dir, 'images/tests/{}'.format(self.sweep_param[0]))
                    test_id = self.dataset + '_tcSILPURAN_'
                    XY_DISPLACEMENT = [[0, 0]]  # self.sweep_param[1]

                elif self.sweep_method == 'calib_1um':
                    test_img_path = join(base_dir, 'images/tests/calib_1umSteps_microscope')
                    test_id = self.dataset + '_1umSteps_'
                    XY_DISPLACEMENT = [[0, 0]]

                elif self.sweep_method == 'micrometer_5um':
                    test_img_path = join(base_dir, 'images/tests/calib_5umSteps_micrometer')
                    test_id = self.dataset + '_5umSteps_'
                    XY_DISPLACEMENT = [[-2, -6]]

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
            NA = 0.25  # 0.35 (best); 0.4
            FOCAL_LENGTH = 24.47
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
            SHAPE_TOL = 0.25
            TEST_MIN_P_AREA = 7
            MAX_P_AREA = 1000  # original: 780;      --> changed on 7/12/23 for P2P sim. Using aspect ratio to filter instead.
            SAME_ID_THRESH = 3  # 2
            BACKGROUND_SUBTRACTION = None  # 'median'  # 'min_value'  # 'min_value'
            OVERLAP_THRESHOLD = 8

            # shared setup
            dc = 0  # 16
            cropping_specs = {'xmin': 0 + dc, 'xmax': 512 - dc, 'ymin': 0 + dc, 'ymax': 512 - dc, 'pad': 5 + dc}
            calib_step_size = self.sweep_param
            threshold_step_size = self.sweep_param
            image_subset = [1, 106, calib_step_size]  # None  # [1, 106, calib_step_size]  # [15, 85]
            test_image_subset = None  # [1, 79, 4]  # [15, 57, 10]  # [40, 70, 1]
            calib_baseline_image = 'calib_50.tif'  # 'calib_50.tif'  # 1-micron calib: 'calib_50.tif'; for 5-micron calib: 'calib_13.tif'
            test_baseline_image = 'test_39.tif'

            if self.static_templates:
                MIN_P_AREA = 1  # 7
                test_template_padding = 14
                calib_template_padding = test_template_padding + 3  # self.sweep_param

                thresholding = 'manual'
                threshold_modifier = 1300
                test_threshold_modifier = 1300
                spct_calib_method = None

            else:
                MIN_P_AREA = 3  # 5
                calib_template_padding = 1  # NOTE: changed from "5" to "1" for P2P similarity evaluation (7/17/23).
                test_template_padding = 0

                thresholding = 'manual'  # 'manual'  # 'median_percent'  # general: 'manual':115;  # SPCT: 'median_percent'
                threshold_modifier = 500  # 1000  # 130  # 200  # 108  # 0.5  # 0.5  # if background subtraction == 'min_value': IDPT: 12; --> ? SPCT: 0.62
                test_threshold_modifier = 250  # 130

                spct_calib_method = 'theory'  # 'theory'

                # DON'T TOUCH THESE SETTINGS. THEY ARE PERFECT!
                calib_theory_threshold = {'theory': threshold_modifier,
                                          'thresh_min': 103,  # 110 (best), 124
                                          'thresh_max': 500,  # 550 (best),  # 220,
                                          'frame_max': 50,  # 50
                                          'dz_per_frame': threshold_step_size,
                                          }
                # -------------------------------------------

                cal_5um_theory_threshold = {'theory': threshold_modifier,
                                            'thresh_min': 110,  # 108,  # 125
                                            'thresh_max': 250,  # 220,
                                            'frame_max': 14,  # 50
                                            'dz_per_frame': threshold_step_size,
                                            }

                test_theory_threshold = {'theory': threshold_modifier,
                                         'thresh_min': 125,  # 108,  # 130
                                         'thresh_max': 250,  # 220,
                                         'frame_max': 41,  # 50
                                         'dz_per_frame': threshold_step_size,
                                         }

            if self.collection_type == 'calibration':
                CALIB_IMG_PATH = calib_img_path
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = calib_id

                if self.sweep_param is not None:
                    CALIB_ID = CALIB_ID + '_{}'.format(self.sweep_param)
                    # '_{}-{}'.format(self.sweep_method, self.sweep_param)
                    if self.static_templates:
                        method = 'idpt'
                    else:
                        method = 'spct'
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-{}_{}-{}'.format(method,
                                                                                              self.sweep_method,
                                                                                              self.sweep_param,
                                                                                              ))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = image_subset  #
                CALIBRATION_Z_STEP_SIZE = calib_step_size
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                BASELINE_IMAGE = calib_baseline_image  # 'calib_50.tif'

                # image stack averaging
                IF_CALIB_IMAGE_STACK = 'mean'  # 'subset', 'mean'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # up to 2 but not including subset_f

                # calibration processing parameters
                if self.static_templates:
                    CALIB_TEMPLATE_PADDING = calib_template_padding
                else:
                    CALIB_TEMPLATE_PADDING = calib_template_padding

                CALIB_CROPPING_SPECS = cropping_specs
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 3
                CALIB_PROCESSING_METHOD2 = 'gaussian'
                CALIB_PROCESSING_FILTER_TYPE2 = None
                CALIB_PROCESSING_FILTER_SIZE2 = 1
                # if self.sweep_param[0] == 0:
                if self.static_templates:
                    CALIB_PROCESSING = None
                else:
                    CALIB_PROCESSING = {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}

                # else:
                #    CALIB_PROCESSING = {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                # {'none': None}
                # {CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                # CALIB_PROCESSING_METHOD2: {'args': [], 'kwargs': dict(sigma=CALIB_PROCESSING_FILTER_SIZE2, preserve_range=True)}

                CALIB_THRESHOLD_METHOD = thresholding  # 'manual'

                if BACKGROUND_SUBTRACTION is not None:
                    CALIB_THRESHOLD_MODIFIER = 0.25  # optics.bkg_noise * 3  # optics.bkg_mean - optics.bkg_noise * 1  # 3
                elif CALIB_THRESHOLD_METHOD == 'manual_percent':
                    CALIB_THRESHOLD_MODIFIER = threshold_modifier
                elif CALIB_THRESHOLD_METHOD == 'median_percent':
                    CALIB_THRESHOLD_MODIFIER = threshold_modifier
                else:
                    CALIB_THRESHOLD_MODIFIER = threshold_modifier  # optics.bkg_mean + optics.bkg_noise * 10

                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                if spct_calib_method == 'theory':
                    # CALIB_THRESHOLD_PARAMS = {'manual': [500]}
                    CALIB_THRESHOLD_PARAMS = calib_theory_threshold

                # similarity
                # if self.sweep_param[0] == 0:
                STACKS_USE_RAW = True
                # else:
                #    STACKS_USE_RAW = False
                MIN_STACKS = 0.15  # percent of calibration stack
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
                TEST_BASE_STRING = 'test_'  # 'test_X'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = test_id

                if self.sweep_param is not None:
                    TEST_ID = TEST_ID + '{}'.format(
                        self.sweep_param)  # '{}-{}'.format(self.sweep_method, self.sweep_param)
                    if self.static_templates:
                        method = 'idpt'
                    else:
                        method = 'spct'
                    TEST_RESULTS_PATH = join(base_dir, 'results/test-{}_{}-{}'.format(method,
                                                                                      self.sweep_method,
                                                                                      self.sweep_param,
                                                                                      ))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = test_image_subset  # image_subset  #
                TRUE_NUM_PARTICLES_PER_IMAGE = 1
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 1]

                TEST_PARTICLE_ID_IMAGE = test_baseline_image  # 'test_X045.tif'  # 'test_39.tif'
                # 1-micron calib: 50; 5-micron test: 39; 1-um calib flat test: 'test_149.tif', test: 'test_X001.tif'

                # test processing parameters
                if self.static_templates:
                    TEST_TEMPLATE_PADDING = test_template_padding
                else:
                    TEST_TEMPLATE_PADDING = 0

                TEST_CROPPING_SPECS = cropping_specs

                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 3  # self.sweep_param[1]
                TEST_PROCESSING_METHOD2 = 'gaussian'
                TEST_PROCESSING_FILTER_TYPE2 = None
                TEST_PROCESSING_FILTER_SIZE2 = 1
                # if self.sweep_param[1] == 0:
                if self.static_templates:
                    TEST_PROCESSING = None
                else:
                    TEST_PROCESSING = {
                        TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}

                # else:
                #    TEST_PROCESSING = {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                # {'none': None}
                # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                # {TEST_PROCESSING_METHOD2: {'args': [], 'kwargs': dict(sigma=TEST_PROCESSING_FILTER_SIZE2, preserve_range=True)}}

                TEST_THRESHOLD_METHOD = thresholding

                if BACKGROUND_SUBTRACTION is not None:
                    TEST_THRESHOLD_MODIFIER = 0.25  # optics.bkg_noise * 3
                elif TEST_THRESHOLD_METHOD == 'median_percent':
                    TEST_THRESHOLD_MODIFIER = threshold_modifier
                else:
                    TEST_THRESHOLD_MODIFIER = test_threshold_modifier  # optics.bkg_mean + optics.bkg_noise * 10  # tbkg_mean + tbkg_noise * 3

                TEST_THRESHOLD_PARAMS = test_theory_threshold  # {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}  # test_theory_threshold  #

                # similarity
                # if self.sweep_param[1] == 0:
                STACKS_USE_RAW = True
                # else:
                #    STACKS_USE_RAW = False
                MIN_STACKS = 0.35  # percent of calibration stack
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
                TEST_IMG_PATH = calib_img_path
                TEST_BASE_STRING = 'calib_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_CROPPING_SPECS = cropping_specs

                TEST_SUBSET = image_subset  # [0, 106, 5]
                IF_TEST_IMAGE_STACK = 'mean'  # 'first'  # 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 3]  # [0, 1]  # [metaset_i, metaset_f]
                TEST_PARTICLE_ID_IMAGE = calib_baseline_image  # was calib_30.tif
                TEST_TEMPLATE_PADDING = 0  # SPCT: 1, IDPT: 16

                if spct_calib_method == 'theory':
                    TEST_THRESHOLD_PARAMS = theoretical_threshold

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
            MAX_P_AREA = 1700  # maximum particle size (area: units are in pixels) (recommended: 200)
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
                    CALIB_RESULTS_PATH = join(base_dir,
                                              'results/calibration-{}-{}'.format(self.sweep_method, self.sweep_param))
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
                CALIB_CROPPING_SPECS = None  # {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 10}
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
                TEST_PROCESSING = {
                    'none': None}  # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
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

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # image pre-processing
            DILATE = None
            SHAPE_TOL = 0.25
            OVERLAP_THRESHOLD = 5

            # optics
            PARTICLE_DIAMETER = 2.15e-6
            DEMAG = 1
            MAGNIFICATION = 20
            NA = 0.3
            FOCAL_LENGTH = 100
            REF_INDEX_MEDIUM = 1
            REF_INDEX_LENS = 1.5
            PIXEL_SIZE = 16e-6
            PIXEL_DIM_X = 512
            PIXEL_DIM_Y = 512
            BKG_MEAN = 240
            BKG_MAX = 280
            BKG_NOISES = 13
            POINTS_PER_PIXEL = None
            N_RAYS = None
            GAIN = 1
            CYL_FOCAL_LENGTH = 0
            WAVELENGTH = 600e-9
            Z_RANGE = 70

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

            # shared variables
            CALIB_IMG_PATH = join(base_dir, 'images/calibration', self.sweep_param[0])
            CALIB_BASE_STRING = 'calib_'
            TEST_IMG_PATH = join(base_dir, 'images/calibration', self.sweep_param[1])
            TEST_BASE_STRING = 'calib_'

            MIN_P_AREA = 5
            MAX_P_AREA = 500
            SAME_ID_THRESH = 4
            BACKGROUND_SUBTRACTION = None
            XY_DISPLACEMENT = [[0, 0]]
            cropping_specs = {'xmin': 180, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 5}

            calib_image_subset = None
            test_image_subset = calib_image_subset
            calib_baseline_image = 'calib_35.tif'
            test_baseline_image = 'calib_35.tif'

            if self.single_particle_calibration is True:
                test_template_padding = 1
                threshold_modifier = optics.bkg_mean - optics.bkg_noise * 6
                # on BPE: + optics.bkg_noise * 2.5;
                # off BPE: - optics.bkg_noise * 5.5
            else:
                test_template_padding = 10
                threshold_modifier = BKG_MAX + optics.bkg_noise * 5

            # collection specific

            if self.collection_type == 'calibration':

                # files
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = self.dataset + '_c-{}_t-{}'.format(self.sweep_param[0], self.sweep_param[1])
                CALIB_RESULTS_PATH = join(base_dir,
                                          'results/cal-{}_test-{}'.format(self.sweep_param[0], self.sweep_param[1]))
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = calib_image_subset
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                IF_CALIB_IMAGE_STACK = 'mean'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [0, 5]
                BASELINE_IMAGE = calib_baseline_image

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 1  # test_template_padding + 2
                CALIB_CROPPING_SPECS = cropping_specs
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 3
                CALIB_PROCESSING = {'none': None}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = threshold_modifier
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.35  # percent of calibration stack
                ZERO_CALIB_STACKS = False
                ZERO_STACKS_OFFSET = 0.0
                INFER_METHODS = 'sknccorr'
                MIN_CM = 0.0
                SUB_IMAGE_INTERPOLATION = True

                # display options
                INSPECT_CALIB_CONTOURS = False
                SHOW_CALIB_PLOTS = False
                SAVE_CALIB_PLOTS = True

            if self.collection_type == 'test':

                # files
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = self.dataset + '_t-{}_c-{}'.format(self.sweep_param[1], self.sweep_param[0])
                TEST_RESULTS_PATH = join(base_dir,
                                         'results/test-{}_cal-{}'.format(self.sweep_param[1], self.sweep_param[0]))
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = test_image_subset
                TRUE_NUM_PARTICLES_PER_IMAGE = 1
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 1]
                TEST_PARTICLE_ID_IMAGE = test_baseline_image

                # test processing parameters
                TEST_TEMPLATE_PADDING = test_template_padding
                TEST_CROPPING_SPECS = cropping_specs
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 3
                TEST_PROCESSING = {'none': None}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = threshold_modifier
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.35  # percent of calibration stack
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
            MAX_P_AREA = 1500  # (750) maximum particle size (area: units are in pixels) (recommended: 200)
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
                    CALIB_ID = CALIB_ID + '_{}'.format(
                        self.sweep_param[0])  # '_{}-{}'.format(self.sweep_method, self.sweep_param)
                    CALIB_RESULTS_PATH = join(base_dir,
                                              'results/calibration-{}-{}'.format(self.sweep_method, self.sweep_param))
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
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i,
                                                subset_f]  # averages the first two images (i.e. up to 2 but not including)
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
                    TEST_ID = TEST_ID + '{}'.format(
                        self.sweep_param)  # '{}-{}'.format(self.sweep_method, self.sweep_param)
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
            subset_i, subset_f = 0, 10  # averages all 10 calibration images per z-step

            if self.collection_type == 'meta-test':
                test_img_path = calib_img_path
                meta_test_id = self.dataset + '_calibration-meta-assessment_'
                test_id = meta_test_id
                subset_i, subset_f = 0, 9  # 9; averages the first 9 calibration images per z-step
                metaset_i, metaset_f = 9, 10
                XY_DISPLACEMENT = [[0, 0]]
            else:
                if self.sweep_method == 'testset':
                    if self.particle_distribution == 'pos':
                        test_img_path = join(base_dir, 'images/tests/pos/test_{}mm_pos'.format(self.sweep_param))
                    elif self.particle_distribution == 'dynamic':
                        test_img_path = join(base_dir, 'images/tests/dynamic/{}'.format(self.sweep_param))
                    test_id = self.dataset + '_test_'
                    XY_DISPLACEMENT = [[0, 0]]

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # image pre-processing
            cropping_specs = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 0}
            DILATE = None  # None or True
            SHAPE_TOL = 0.25  # None == take any shape; 1 == take perfectly circular shape only.
            MIN_P_AREA = 30  # minimum particle size (area: units are in pixels) (recommended: 20)
            MAX_P_AREA = 500  # (750) maximum particle size (area: units are in pixels) (recommended: 200)
            SAME_ID_THRESH = 7  # maximum tolerance in x- and y-directions for particle to have the same ID between images
            BACKGROUND_SUBTRACTION = None
            OVERLAP_THRESHOLD = 5

            # optics
            PARTICLE_DIAMETER = 2.15e-6
            DEMAG = 0.5
            MAGNIFICATION = 20
            NA = 0.3
            FOCAL_LENGTH = 50
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
                    CALIB_RESULTS_PATH = join(base_dir,
                                              'results/calibration-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = [20, 95]  # [20, 95, 5]
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                BASELINE_IMAGE = 'calib_58.tif'  # NOTES: ~'calib_57-60.tif' is the peak intensity image.

                # image stack averaging
                IF_CALIB_IMAGE_STACK = 'subset'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # averages up to subset_f but not including

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 3  # 12
                CALIB_CROPPING_SPECS = cropping_specs
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 3
                CALIB_PROCESSING = {
                    CALIB_PROCESSING_METHOD: {'args': [square(CALIB_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                # {'none': None}
                CALIB_THRESHOLD_METHOD = 'manual'
                CALIB_THRESHOLD_MODIFIER = 150  # optics.bkg_mean + optics.bkg_noise * 10
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.75  # percent of calibration stack
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
                    TEST_ID = TEST_ID + '{}'.format(
                        self.sweep_param)  # '{}-{}'.format(self.sweep_method, self.sweep_param)
                    TEST_RESULTS_PATH = join(base_dir, 'results/test-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = None  # [20, 95]
                TRUE_NUM_PARTICLES_PER_IMAGE = 250

                # test processing parameters
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 1]
                TEST_PARTICLE_ID_IMAGE = 'test_X1.tif'
                TEST_TEMPLATE_PADDING = 1  # 9
                TEST_CROPPING_SPECS = cropping_specs
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 3
                TEST_PROCESSING = {
                    TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                # {'none': None}
                TEST_THRESHOLD_METHOD = 'manual'
                TEST_THRESHOLD_MODIFIER = 150  # optics.bkg_mean + optics.bkg_noise * 10
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.75  # percent of calibration stack
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

                    TEST_SUBSET = [20, 95]
                    IF_TEST_IMAGE_STACK = 'subset'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [metaset_i, metaset_f]
                    TEST_PARTICLE_ID_IMAGE = 'calib_58.tif'

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == '02.06.22_membrane_characterization':

            base_dir = join('/Users/mackenzie/Desktop/gdpyt-characterization/experiments', self.dataset)

            if self.particle_distribution == '20X':
                """
                Membrane region (dark): mean = 104, std = 2.5, max = 115
                Surface region (brighter): mean = 130, std = 4.5, max = 140
                """
                bkg_mean = 105  # 130
                bkg_noise = 4.5
                calib_img_path = join(base_dir, 'images/20X/calibration')
                calib_z_step_size = 1.0
                calib_baseline_image = 'calib_90.tif'
                magnification = 20
                numerical_aperture = 0.45

            elif self.particle_distribution == '10X':
                """
                Membrane region (very dark): mean = 107, std = 2.5, max = 115
                Surface region (brighter): mean = 140, std = 4.5, max = 160
                """
                bkg_mean = 140
                bkg_max = 160
                bkg_noise = 4.5
                calib_img_path = join(base_dir, 'images/10X/calibration_2umSteps')
                calib_z_step_size = 2.0
                calib_baseline_image = 'calib_67.tif'
                magnification = 10
                numerical_aperture = 0.3

            calib_id = self.dataset + '_calib'
            subset_i, subset_f = 0, 10  # averages all 10 calibration images per z-step

            if self.collection_type == 'meta-test':
                test_img_path = calib_img_path
                meta_test_id = self.dataset + '_calibration-meta-assessment_'
                test_id = meta_test_id
                subset_i, subset_f = 0, 9  # 9; averages the first 9 calibration images per z-step
                metaset_i, metaset_f = subset_f, subset_f + 1
                XY_DISPLACEMENT = [[0, 0]]
            elif self.sweep_method == 'testset':
                test_img_path = join(base_dir, 'images/{}/tests/{}'.format(self.particle_distribution,
                                                                           self.sweep_param[0]))
                test_id = self.dataset + '_test_'
                XY_DISPLACEMENT = [[0, 0]]

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # image pre-processing
            DILATE = None
            SHAPE_TOL = 0.25
            SAME_ID_THRESH = 5  # 8
            OVERLAP_THRESHOLD = 5

            # optics
            PARTICLE_DIAMETER = 2.15e-6
            DEMAG = 0.5
            MAGNIFICATION = magnification
            NA = numerical_aperture
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

            # shared variables
            MIN_P_AREA = 15  # 10X = 4, 20X = 5
            MAX_P_AREA = 600
            BACKGROUND_SUBTRACTION = None  # {'method': 'baseline_image_subtraction', 'param': 100}
            cropping_specs = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 15}
            image_subset = [0, 275, self.sweep_param[2]]
            test_image_subset = None  # [20, 81, 1]  # [33, 42, 1]  # None  # [40, 200, 1]
            baseline_image = calib_baseline_image
            idpt_test_template_padding = self.sweep_param[1]  # 13
            idpt_calib_template_padding = idpt_test_template_padding + 5  # 11
            threshold_modifier = optics.bkg_mean + optics.bkg_noise * 100  # 200  # 7; IDPT = 50; SPCT large memb = -6

            if self.single_particle_calibration is True:
                spct_calib_threshold_modifier = optics.bkg_mean + optics.bkg_noise * -4
                spct_test_threshold_modifier = optics.bkg_mean + optics.bkg_noise * -3
                print("SPCT calib and test threshold values = {}, {}".format(spct_calib_threshold_modifier,
                                                                             spct_test_threshold_modifier))
            else:
                print("Threshold value = {}".format(threshold_modifier))

            # ---

            if self.collection_type == 'calibration':
                CALIB_IMG_PATH = calib_img_path
                CALIB_BASE_STRING = 'calib_'
                CALIB_GROUND_TRUTH_PATH = None
                CALIB_ID = calib_id

                if self.sweep_param is not None:
                    CALIB_ID = CALIB_ID + '_{}'.format(self.sweep_param)
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-{}-{}'.format(self.sweep_method,
                                                                                           self.sweep_param))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = image_subset
                CALIBRATION_Z_STEP_SIZE = calib_z_step_size
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                BASELINE_IMAGE = baseline_image

                # image stack averaging
                IF_CALIB_IMAGE_STACK = 'subset'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # averages up to subset_f but not including

                # calibration processing parameters
                if self.single_particle_calibration is True:
                    CALIB_TEMPLATE_PADDING = 3
                    threshold_modifier = spct_calib_threshold_modifier
                    """MIN_P_AREA = 3  # 5
                    calib_template_padding = 2
                    test_template_padding = 0

                    thresholding = 'manual'  # 'manual'  # 'median_percent'  # general: 'manual':115;  # SPCT: 'median_percent'
                    threshold_modifier = 500  # 1000  # 130  # 200  # 108  # 0.5  # 0.5  # if background subtraction == 'min_value': IDPT: 12; --> ? SPCT: 0.62
                    test_threshold_modifier = 250  # 130

                    spct_calib_method = 'theory'  # 'theory'

                    # DON'T TOUCH THESE SETTINGS. THEY ARE PERFECT!
                    calib_theory_threshold = {'theory': threshold_modifier,
                                              'thresh_min': 103,  # 110 (best), 124
                                              'thresh_max': 500,  # 550 (best),  # 220,
                                              'frame_max': 50,  # 50
                                              'dz_per_frame': threshold_step_size,
                                              }"""
                else:
                    CALIB_TEMPLATE_PADDING = idpt_calib_template_padding

                CALIB_CROPPING_SPECS = cropping_specs
                CALIB_PROCESSING_METHOD = 'median'
                CALIB_PROCESSING_FILTER_TYPE = 'square'
                CALIB_PROCESSING_FILTER_SIZE = 3
                CALIB_PROCESSING = None
                CALIB_THRESHOLD_METHOD = 'manual'
                if BACKGROUND_SUBTRACTION is None:
                    CALIB_THRESHOLD_MODIFIER = threshold_modifier
                else:
                    CALIB_THRESHOLD_MODIFIER = optics.bkg_noise * 8
                CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5  # percent of calibration stack
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
                    TEST_ID = TEST_ID + '{}'.format(
                        self.sweep_param)  # '{}-{}'.format(self.sweep_method, self.sweep_param)
                    TEST_RESULTS_PATH = join(base_dir,
                                             'results/test-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = test_image_subset
                TRUE_NUM_PARTICLES_PER_IMAGE = 1

                # test processing parameters
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 1]
                TEST_PARTICLE_ID_IMAGE = 'test_X1.tif'

                if self.single_particle_calibration is True:
                    TEST_TEMPLATE_PADDING = 0
                else:
                    TEST_TEMPLATE_PADDING = idpt_test_template_padding

                TEST_CROPPING_SPECS = cropping_specs
                TEST_PROCESSING_METHOD = 'median'
                TEST_PROCESSING_FILTER_TYPE = 'square'
                TEST_PROCESSING_FILTER_SIZE = 3
                TEST_PROCESSING = None
                # {TEST_PROCESSING_METHOD: {'args': [square(TEST_PROCESSING_FILTER_SIZE), None, 'wrap']}}
                TEST_THRESHOLD_METHOD = 'manual'
                if BACKGROUND_SUBTRACTION is None:
                    if self.single_particle_calibration is True:
                        threshold_modifier = spct_test_threshold_modifier
                    TEST_THRESHOLD_MODIFIER = threshold_modifier
                else:
                    TEST_THRESHOLD_MODIFIER = optics.bkg_noise * 25  # 8
                TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}

                # similarity
                STACKS_USE_RAW = True
                MIN_STACKS = 0.5  # percent of calibration stack
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

                    TEST_SUBSET = image_subset
                    IF_TEST_IMAGE_STACK = 'subset'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [metaset_i, metaset_f]
                    TEST_PARTICLE_ID_IMAGE = baseline_image

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == 'zipper_old':

            base_dir = join('/Users/mackenzie/Desktop/gdpyt-characterization/experiments', self.dataset)

            bkg_mean = 310  # 350  # 262
            bkg_noise = 25  # 23
            calib_img_path = join(base_dir, 'images/iter2/calibration')
            calib_id = self.dataset + '_calib'
            subset_i, subset_f = 0, 1  # averages all 10 calibration images per z-step

            if self.collection_type == 'meta-test':
                test_img_path = calib_img_path
                meta_test_id = self.dataset + '_calibration-meta-assessment_'
                test_id = meta_test_id
                subset_i, subset_f = 0, 1  # averages the first 9 calibration images per z-step
                metaset_i, metaset_f = 0, 1  # subset_f, subset_f + 1
                XY_DISPLACEMENT = [[0, 0]]
            else:
                if self.sweep_method == 'testset':
                    test_img_path = join(base_dir, 'images/iter2/tests/test_{}V'.format(self.sweep_param))
                    test_id = self.dataset + '_test_'
                    XY_DISPLACEMENT = [[32, 3]]

            # shared variables

            # file types
            filetype = '.tif'
            filetype_ground_truth = '.txt'

            # image pre-processing
            cropping_specs = {'xmin': 0, 'xmax': 512, 'ymin': 0, 'ymax': 512, 'pad': 30}
            DILATE = None  # None or True
            SHAPE_TOL = 0.25  # None == take any shape; 1 == take perfectly circular shape only.
            MIN_P_AREA = 50  # minimum particle size (area: units are in pixels) (recommended: 20)
            MAX_P_AREA = 5000  # (750) maximum particle size (area: units are in pixels) (recommended: 200)
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
                    CALIB_RESULTS_PATH = join(base_dir,
                                              'results/calibration-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                if not os.path.exists(CALIB_RESULTS_PATH):
                    os.makedirs(CALIB_RESULTS_PATH)

                # calib dataset information
                CALIB_SUBSET = [0, 79, 2]  # used [30, 59] for membrane curvature measurements
                CALIBRATION_Z_STEP_SIZE = 1.0
                TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                BASELINE_IMAGE = 'calib_022.tif'  # NOTES: ~'calib_57-60.tif' is the peak intensity image.

                # image stack averaging
                IF_CALIB_IMAGE_STACK = 'first'  # 'subset'
                TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # averages up to subset_f but not including

                # calibration processing parameters
                CALIB_TEMPLATE_PADDING = 12  # used 9 for membrane curvature measurements
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
                SAVE_CALIB_PLOTS = True

            if self.collection_type == 'test' or self.collection_type == 'meta-test':
                TEST_IMG_PATH = test_img_path
                TEST_BASE_STRING = 'test_'
                TEST_GROUND_TRUTH_PATH = None
                TEST_ID = test_id

                if self.sweep_param is not None:
                    TEST_ID = TEST_ID + '{}'.format(
                        self.sweep_param)  # '{}-{}'.format(self.sweep_method, self.sweep_param)
                    TEST_RESULTS_PATH = join(base_dir, 'results/test-{}-{}'.format(self.sweep_method, self.sweep_param))
                else:
                    TEST_RESULTS_PATH = join(base_dir, 'results/test')
                if not os.path.exists(TEST_RESULTS_PATH):
                    os.makedirs(TEST_RESULTS_PATH)

                # test dataset information
                TEST_SUBSET = [0, 300, 15]
                TRUE_NUM_PARTICLES_PER_IMAGE = 250

                # test processing parameters
                IF_TEST_IMAGE_STACK = 'first'
                TAKE_TEST_IMAGE_SUBSET_MEAN = [1, 2]
                TEST_PARTICLE_ID_IMAGE = 'test_001.tif'
                TEST_TEMPLATE_PADDING = 10  # used 6 for membrane curvature measurements
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

                    TEST_SUBSET = [1, 78, 2]
                    IF_TEST_IMAGE_STACK = 'first'  # 'subset'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [metaset_i, metaset_f]
                    TEST_PARTICLE_ID_IMAGE = 'calib_022.tif'

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == 'wA2':

            base_dir = join('/Users/mackenzie/Desktop/Zipper/Testing', self.dataset)

            bkg_mean = 360
            bkg_noise = 45
            calib_img_path = join(base_dir, 'images/calibration/cal-from-test')
            calib_step_size = 1
            calib_id = self.dataset + '_cal-from-test_'
            subset_i, subset_f = 0, 10

            test_img_path = join(base_dir, 'images/tests/{}/{}'.format(self.sweep_param[0], self.sweep_param[1]))
            test_id = self.dataset + '_071923_'
            print(test_img_path)

            # shared setup
            dc = 5  # 16
            cropping_specs = {'xmin': 0 + dc, 'xmax': 512 - dc, 'ymin': 0 + dc, 'ymax': 512 - dc, 'pad': 5 + dc}

            calib_image_subset = [47, 88, 1]  # [47, 136, 10]  # None  # [1, 106, calib_step_size]  # [15, 85]
            calib_base_string = 'calib_'
            calib_baseline_image = 'calib_47.tif'

            test_base_string = 'test_X'  # 'test_X' 'test_'
            test_baseline_image = 'test_X1.tif'
            test_image_subset = [1, 200, 1]  # None, [10, 79, 4]

            test_template_padding = 4
            calib_template_padding = test_template_padding + 4  # self.sweep_param
            thresholding = 'manual'
            threshold_modifier = 3000
            test_threshold_modifier = threshold_modifier
            MIN_P_AREA = 4
            MAX_P_AREA = 50
            XY_DISPLACEMENT = [[0, 0]]

            # shared variables
            hide_variables = True
            if hide_variables:

                # file types
                filetype = '.tif'
                filetype_ground_truth = '.txt'

                # optics
                PARTICLE_DIAMETER = 2.15e-6
                DEMAG = 1
                MAGNIFICATION = 10
                NA = 0.3
                FOCAL_LENGTH = None
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
                SHAPE_TOL = 0.25
                SAME_ID_THRESH = 3
                BACKGROUND_SUBTRACTION = None  # 'median'  # 'min_value'  # 'min_value'
                OVERLAP_THRESHOLD = 8

                if self.collection_type == 'calibration':
                    CALIB_IMG_PATH = calib_img_path
                    CALIB_BASE_STRING = calib_base_string
                    CALIB_GROUND_TRUTH_PATH = None
                    CALIB_ID = calib_id

                    if self.sweep_param is not None:
                        CALIB_ID = CALIB_ID + '_{}'.format(self.sweep_param)
                        method = 'idpt'
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-{}_{}-{}'.format(method,
                                                                                                  self.sweep_method,
                                                                                                  self.sweep_param,
                                                                                                  ))
                    else:
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                    if not os.path.exists(CALIB_RESULTS_PATH):
                        os.makedirs(CALIB_RESULTS_PATH)

                    # calib dataset information
                    CALIB_SUBSET = calib_image_subset
                    CALIBRATION_Z_STEP_SIZE = calib_step_size
                    TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                    BASELINE_IMAGE = calib_baseline_image
                    IF_CALIB_IMAGE_STACK = 'mean'  # 'subset', 'mean'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # up to 2 but not including subset_f
                    CALIB_TEMPLATE_PADDING = calib_template_padding
                    CALIB_CROPPING_SPECS = cropping_specs
                    CALIB_PROCESSING_METHOD = None
                    CALIB_PROCESSING_FILTER_TYPE = None
                    CALIB_PROCESSING_FILTER_SIZE = None
                    CALIB_PROCESSING_METHOD2 = None
                    CALIB_PROCESSING_FILTER_TYPE2 = None
                    CALIB_PROCESSING_FILTER_SIZE2 = None
                    CALIB_PROCESSING = None
                    CALIB_THRESHOLD_METHOD = thresholding
                    CALIB_THRESHOLD_MODIFIER = threshold_modifier
                    CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
                    TEST_BASE_STRING = test_base_string
                    TEST_GROUND_TRUTH_PATH = None
                    TEST_ID = test_id

                    if self.sweep_param is not None:
                        TEST_ID = TEST_ID + '{}'.format(self.sweep_param)
                        method = 'idpt'
                        TEST_RESULTS_PATH = join(base_dir, 'results/test-{}_{}-{}'.format(method,
                                                                                          self.sweep_method,
                                                                                          self.sweep_param,
                                                                                          ))
                    else:
                        TEST_RESULTS_PATH = join(base_dir, 'results/test')
                    if not os.path.exists(TEST_RESULTS_PATH):
                        os.makedirs(TEST_RESULTS_PATH)

                    # test dataset information
                    TEST_SUBSET = test_image_subset
                    TRUE_NUM_PARTICLES_PER_IMAGE = 1
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 1]

                    TEST_PARTICLE_ID_IMAGE = test_baseline_image
                    TEST_TEMPLATE_PADDING = test_template_padding
                    TEST_CROPPING_SPECS = cropping_specs
                    TEST_PROCESSING_METHOD = None
                    TEST_PROCESSING_FILTER_TYPE = None
                    TEST_PROCESSING_FILTER_SIZE = None
                    TEST_PROCESSING_METHOD2 = None
                    TEST_PROCESSING_FILTER_TYPE2 = None
                    TEST_PROCESSING_FILTER_SIZE2 = None
                    TEST_PROCESSING = None
                    TEST_THRESHOLD_METHOD = thresholding
                    TEST_THRESHOLD_MODIFIER = test_threshold_modifier
                    TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}  # test_theory_threshold

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
            else:
                raise ValueError("Need those variables though...")

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == 'wA_b1_test2':

            base_dir = join('/Users/mackenzie/Desktop/Zipper/Testing/wA', self.dataset)

            bkg_mean = 360
            bkg_noise = 45
            calib_img_path = join(base_dir, 'images/calibration')
            calib_step_size = 1
            calib_id = self.dataset + '_cal_'
            subset_i, subset_f = 0, 10

            test_img_path = join(base_dir, 'images/tests2/{}/{}'.format(self.sweep_param[0], self.sweep_param[1]))
            # IF "tests2" must include X,Y offset:
            XY_DISPLACEMENT = [[-5, 0]]
            # ELSE,
            #   XY_DISPLACEMENT = [[0, 0]]
            test_id = self.dataset + '_072123_'

            # shared setup
            dc = 0  # 16
            cropping_specs = {'xmin': 0 + dc, 'xmax': 512 - dc, 'ymin': 0 + dc, 'ymax': 512 - dc, 'pad': 10 + dc}

            calib_image_subset = [71, 135, 1]  # [47, 136, 10]  # None  # [1, 106, calib_step_size]  # [15, 85]
            calib_base_string = 'calib_'
            calib_baseline_image = 'calib_71.tif'

            test_base_string = 'test_X'  # 'test_X' 'test_'
            test_baseline_image = 'test_X25.tif'
            test_image_subset = [25, 120, 1]  # None, [10, 79, 4]

            test_template_padding = 7
            calib_template_padding = test_template_padding + 6  # self.sweep_param
            thresholding = 'manual'
            threshold_modifier = 3000
            test_threshold_modifier = threshold_modifier
            MIN_P_AREA = 3
            MAX_P_AREA = 100

            # shared variables
            hide_variables = True
            if hide_variables:

                # file types
                filetype = '.tif'
                filetype_ground_truth = '.txt'

                # optics
                PARTICLE_DIAMETER = 2.15e-6
                DEMAG = 1
                MAGNIFICATION = 10
                NA = 0.3
                FOCAL_LENGTH = None
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
                SHAPE_TOL = 0.25
                SAME_ID_THRESH = 3
                BACKGROUND_SUBTRACTION = None  # 'median'  # 'min_value'  # 'min_value'
                OVERLAP_THRESHOLD = 8

                if self.collection_type == 'calibration':
                    CALIB_IMG_PATH = calib_img_path
                    CALIB_BASE_STRING = calib_base_string
                    CALIB_GROUND_TRUTH_PATH = None
                    CALIB_ID = calib_id

                    if self.sweep_param is not None:
                        CALIB_ID = CALIB_ID + '_{}'.format(self.sweep_param)
                        method = 'idpt'
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-{}_{}-{}'.format(method,
                                                                                                  self.sweep_method,
                                                                                                  self.sweep_param,
                                                                                                  ))
                    else:
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                    if not os.path.exists(CALIB_RESULTS_PATH):
                        os.makedirs(CALIB_RESULTS_PATH)

                    # calib dataset information
                    CALIB_SUBSET = calib_image_subset
                    CALIBRATION_Z_STEP_SIZE = calib_step_size
                    TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                    BASELINE_IMAGE = calib_baseline_image
                    IF_CALIB_IMAGE_STACK = 'mean'  # 'subset', 'mean'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # up to 2 but not including subset_f
                    CALIB_TEMPLATE_PADDING = calib_template_padding
                    CALIB_CROPPING_SPECS = cropping_specs
                    CALIB_PROCESSING_METHOD = None
                    CALIB_PROCESSING_FILTER_TYPE = None
                    CALIB_PROCESSING_FILTER_SIZE = None
                    CALIB_PROCESSING_METHOD2 = None
                    CALIB_PROCESSING_FILTER_TYPE2 = None
                    CALIB_PROCESSING_FILTER_SIZE2 = None
                    CALIB_PROCESSING = None
                    CALIB_THRESHOLD_METHOD = thresholding
                    CALIB_THRESHOLD_MODIFIER = threshold_modifier
                    CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
                    TEST_BASE_STRING = test_base_string
                    TEST_GROUND_TRUTH_PATH = None
                    TEST_ID = test_id

                    if self.sweep_param is not None:
                        TEST_ID = TEST_ID + '{}'.format(self.sweep_param)
                        method = 'idpt'
                        TEST_RESULTS_PATH = join(base_dir, 'results/test-{}_{}-{}'.format(method,
                                                                                          self.sweep_method,
                                                                                          self.sweep_param,
                                                                                          ))
                    else:
                        TEST_RESULTS_PATH = join(base_dir, 'results/test')
                    if not os.path.exists(TEST_RESULTS_PATH):
                        os.makedirs(TEST_RESULTS_PATH)

                    # test dataset information
                    TEST_SUBSET = test_image_subset
                    TRUE_NUM_PARTICLES_PER_IMAGE = 1
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 1]

                    TEST_PARTICLE_ID_IMAGE = test_baseline_image
                    TEST_TEMPLATE_PADDING = test_template_padding
                    TEST_CROPPING_SPECS = cropping_specs
                    TEST_PROCESSING_METHOD = None
                    TEST_PROCESSING_FILTER_TYPE = None
                    TEST_PROCESSING_FILTER_SIZE = None
                    TEST_PROCESSING_METHOD2 = None
                    TEST_PROCESSING_FILTER_TYPE2 = None
                    TEST_PROCESSING_FILTER_SIZE2 = None
                    TEST_PROCESSING = None
                    TEST_THRESHOLD_METHOD = thresholding
                    TEST_THRESHOLD_MODIFIER = test_threshold_modifier
                    TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}  # test_theory_threshold

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
            else:
                raise ValueError("Need those variables though...")

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == 'w18_b1_test3':

            base_dir = join('/Users/mackenzie/Desktop/Zipper/Testing/Wafer18', self.dataset)

            bkg_mean = 360
            bkg_noise = 45

            calib_dir = join('/Users/mackenzie/Desktop/Zipper/Testing/Wafer18/w18_b1_test2')
            calib_img_path = join(calib_dir, 'images/calibration')
            calib_step_size = 1
            calib_id = self.dataset + '_cal_'
            subset_i, subset_f = 0, 5

            test_img_path = join(base_dir, 'images/tests/{}'.format(self.sweep_param[0]))
            test_id = self.dataset + '_080223_'

            # shared setup
            dc = 0  # 16
            cropping_specs = {'xmin': 0 + dc, 'xmax': 512 - dc, 'ymin': 0 + dc, 'ymax': 512 - dc, 'pad': 10 + dc}

            calib_image_subset = [70, 138, 1]  # [47, 136, 10]  # None  # [1, 106, calib_step_size]  # [15, 85]
            calib_base_string = 'calib_'
            calib_baseline_image = 'calib_70.tif'

            test_base_string = 'test_X'  # 'test_X' 'test_'
            test_baseline_image = 'test_X1.tif'
            test_image_subset = None  # None, [10, 79, 4]

            test_template_padding = 8
            calib_template_padding = test_template_padding + 6
            thresholding = 'manual'
            threshold_modifier = 3000
            test_threshold_modifier = threshold_modifier
            MIN_P_AREA = 3
            MAX_P_AREA = 200
            XY_DISPLACEMENT = [[-2, 3]]  # the calib image must shift: 2 pixels LEFT; 3 pixels DOWN
            # NOTE: [[X, Y]] is applied to the calibration baseline locations. So, what dx/dy shifts calib to test?
            SAME_ID_THRESH = 8

            # shared variables
            hide_variables = True
            if hide_variables:

                # file types
                filetype = '.tif'
                filetype_ground_truth = '.txt'

                # optics
                PARTICLE_DIAMETER = 2.15e-6
                DEMAG = 1
                MAGNIFICATION = 10
                NA = 0.3
                FOCAL_LENGTH = None
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
                SHAPE_TOL = 0.25
                BACKGROUND_SUBTRACTION = None  # 'median'  # 'min_value'  # 'min_value'
                OVERLAP_THRESHOLD = 8

                if self.collection_type == 'calibration':
                    CALIB_IMG_PATH = calib_img_path
                    CALIB_BASE_STRING = calib_base_string
                    CALIB_GROUND_TRUTH_PATH = None
                    CALIB_ID = calib_id

                    if self.sweep_param is not None:
                        CALIB_ID = CALIB_ID + '_{}'.format(self.sweep_param)
                        method = 'idpt'
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-{}_{}-{}'.format(method,
                                                                                                  self.sweep_method,
                                                                                                  self.sweep_param,
                                                                                                  ))
                    else:
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                    if not os.path.exists(CALIB_RESULTS_PATH):
                        os.makedirs(CALIB_RESULTS_PATH)

                    # calib dataset information
                    CALIB_SUBSET = calib_image_subset
                    CALIBRATION_Z_STEP_SIZE = calib_step_size
                    TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                    BASELINE_IMAGE = calib_baseline_image
                    IF_CALIB_IMAGE_STACK = 'mean'  # 'subset', 'mean'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # up to 2 but not including subset_f
                    CALIB_TEMPLATE_PADDING = calib_template_padding
                    CALIB_CROPPING_SPECS = cropping_specs
                    CALIB_PROCESSING_METHOD = None
                    CALIB_PROCESSING_FILTER_TYPE = None
                    CALIB_PROCESSING_FILTER_SIZE = None
                    CALIB_PROCESSING_METHOD2 = None
                    CALIB_PROCESSING_FILTER_TYPE2 = None
                    CALIB_PROCESSING_FILTER_SIZE2 = None
                    CALIB_PROCESSING = None
                    CALIB_THRESHOLD_METHOD = thresholding
                    CALIB_THRESHOLD_MODIFIER = threshold_modifier
                    CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
                    TEST_BASE_STRING = test_base_string
                    TEST_GROUND_TRUTH_PATH = None
                    TEST_ID = test_id

                    if self.sweep_param is not None:
                        TEST_ID = TEST_ID + '{}'.format(self.sweep_param)
                        method = 'idpt'
                        TEST_RESULTS_PATH = join(base_dir, 'results/test-{}_{}-{}'.format(method,
                                                                                          self.sweep_method,
                                                                                          self.sweep_param,
                                                                                          ))
                    else:
                        TEST_RESULTS_PATH = join(base_dir, 'results/test')
                    if not os.path.exists(TEST_RESULTS_PATH):
                        os.makedirs(TEST_RESULTS_PATH)

                    # test dataset information
                    TEST_SUBSET = test_image_subset
                    TRUE_NUM_PARTICLES_PER_IMAGE = 1
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 1]

                    TEST_PARTICLE_ID_IMAGE = test_baseline_image
                    TEST_TEMPLATE_PADDING = test_template_padding
                    TEST_CROPPING_SPECS = cropping_specs
                    TEST_PROCESSING_METHOD = None
                    TEST_PROCESSING_FILTER_TYPE = None
                    TEST_PROCESSING_FILTER_SIZE = None
                    TEST_PROCESSING_METHOD2 = None
                    TEST_PROCESSING_FILTER_TYPE2 = None
                    TEST_PROCESSING_FILTER_SIZE2 = None
                    TEST_PROCESSING = None
                    TEST_THRESHOLD_METHOD = thresholding
                    TEST_THRESHOLD_MODIFIER = test_threshold_modifier
                    TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}  # test_theory_threshold

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
            else:
                raise ValueError("Need those variables though...")

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == 'wA_a1_test1':

            base_dir = join('/Users/mackenzie/Desktop/Zipper/Testing/wA', self.dataset)

            bkg_mean = 360
            bkg_noise = 45
            calib_img_path = join(base_dir, 'images/calibration')
            calib_step_size = 1
            calib_id = self.dataset + '_cal_'
            subset_i, subset_f = 0, 5

            test_img_path = join(base_dir, 'images/tests/{}/{}'.format(self.sweep_param[0], self.sweep_param[1]))
            test_id = self.dataset + '_072223_'

            # shared setup
            dc = 0  # 16
            cropping_specs = {'xmin': 0 + dc, 'xmax': 512 - dc, 'ymin': 0 + dc, 'ymax': 512 - dc, 'pad': 10 + dc}

            calib_image_subset = [55, 140, 2]  # [47, 136, 10]  # None  # [1, 106, calib_step_size]  # [15, 85]
            calib_base_string = 'calib_'
            calib_baseline_image = 'calib_65.tif'

            test_base_string = 'test_X'  # 'test_X' 'test_'
            test_baseline_image = 'test_X25.tif'
            test_image_subset = [25, 130, 1]  # None, [10, 79, 4]

            test_template_padding = 7
            calib_template_padding = test_template_padding + 6  # self.sweep_param
            thresholding = 'manual'
            threshold_modifier = 3000
            test_threshold_modifier = threshold_modifier
            MIN_P_AREA = 3
            MAX_P_AREA = 100
            XY_DISPLACEMENT = [[0, 0]]

            # shared variables
            hide_variables = True
            if hide_variables:

                # file types
                filetype = '.tif'
                filetype_ground_truth = '.txt'

                # optics
                PARTICLE_DIAMETER = 2.15e-6
                DEMAG = 1
                MAGNIFICATION = 10
                NA = 0.3
                FOCAL_LENGTH = None
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
                SHAPE_TOL = 0.25
                SAME_ID_THRESH = 3
                BACKGROUND_SUBTRACTION = None  # 'median'  # 'min_value'  # 'min_value'
                OVERLAP_THRESHOLD = 8

                if self.collection_type == 'calibration':
                    CALIB_IMG_PATH = calib_img_path
                    CALIB_BASE_STRING = calib_base_string
                    CALIB_GROUND_TRUTH_PATH = None
                    CALIB_ID = calib_id

                    if self.sweep_param is not None:
                        CALIB_ID = CALIB_ID + '_{}'.format(self.sweep_param)
                        method = 'idpt'
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-{}_{}-{}'.format(method,
                                                                                                  self.sweep_method,
                                                                                                  self.sweep_param,
                                                                                                  ))
                    else:
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                    if not os.path.exists(CALIB_RESULTS_PATH):
                        os.makedirs(CALIB_RESULTS_PATH)

                    # calib dataset information
                    CALIB_SUBSET = calib_image_subset
                    CALIBRATION_Z_STEP_SIZE = calib_step_size
                    TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                    BASELINE_IMAGE = calib_baseline_image
                    IF_CALIB_IMAGE_STACK = 'mean'  # 'subset', 'mean'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # up to 2 but not including subset_f
                    CALIB_TEMPLATE_PADDING = calib_template_padding
                    CALIB_CROPPING_SPECS = cropping_specs
                    CALIB_PROCESSING_METHOD = None
                    CALIB_PROCESSING_FILTER_TYPE = None
                    CALIB_PROCESSING_FILTER_SIZE = None
                    CALIB_PROCESSING_METHOD2 = None
                    CALIB_PROCESSING_FILTER_TYPE2 = None
                    CALIB_PROCESSING_FILTER_SIZE2 = None
                    CALIB_PROCESSING = None
                    CALIB_THRESHOLD_METHOD = thresholding
                    CALIB_THRESHOLD_MODIFIER = threshold_modifier
                    CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
                    TEST_BASE_STRING = test_base_string
                    TEST_GROUND_TRUTH_PATH = None
                    TEST_ID = test_id

                    if self.sweep_param is not None:
                        TEST_ID = TEST_ID + '{}'.format(self.sweep_param)
                        method = 'idpt'
                        TEST_RESULTS_PATH = join(base_dir, 'results/test-{}_{}-{}'.format(method,
                                                                                          self.sweep_method,
                                                                                          self.sweep_param,
                                                                                          ))
                    else:
                        TEST_RESULTS_PATH = join(base_dir, 'results/test')
                    if not os.path.exists(TEST_RESULTS_PATH):
                        os.makedirs(TEST_RESULTS_PATH)

                    # test dataset information
                    TEST_SUBSET = test_image_subset
                    TRUE_NUM_PARTICLES_PER_IMAGE = 1
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 1]

                    TEST_PARTICLE_ID_IMAGE = test_baseline_image
                    TEST_TEMPLATE_PADDING = test_template_padding
                    TEST_CROPPING_SPECS = cropping_specs
                    TEST_PROCESSING_METHOD = None
                    TEST_PROCESSING_FILTER_TYPE = None
                    TEST_PROCESSING_FILTER_SIZE = None
                    TEST_PROCESSING_METHOD2 = None
                    TEST_PROCESSING_FILTER_TYPE2 = None
                    TEST_PROCESSING_FILTER_SIZE2 = None
                    TEST_PROCESSING = None
                    TEST_THRESHOLD_METHOD = thresholding
                    TEST_THRESHOLD_MODIFIER = test_threshold_modifier
                    TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}  # test_theory_threshold

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
            else:
                raise ValueError("Need those variables though...")

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == 'wA_c1_test1':

            base_dir = join('/Users/mackenzie/Desktop/Zipper/Testing/wA', self.dataset)

            bkg_mean = 360
            bkg_noise = 45
            calib_img_path = join(base_dir, 'images/calibration')
            calib_step_size = 1
            calib_id = self.dataset + '_cal_'
            subset_i, subset_f = 0, 5

            test_img_path = join(base_dir, 'images/tests/{}/{}'.format(self.sweep_param[0], self.sweep_param[1]))
            test_id = self.dataset + '_072223_'

            # shared setup
            dc = 0  # 16
            cropping_specs = {'xmin': 0 + dc, 'xmax': 512 - dc, 'ymin': 0 + dc, 'ymax': 512 - dc, 'pad': 10 + dc}

            calib_image_subset = [90, 155, 1]  # [47, 136, 10]  # None  # [1, 106, calib_step_size]  # [15, 85]
            calib_base_string = 'calib_'
            calib_baseline_image = 'calib_95.tif'

            test_base_string = 'test_X'  # 'test_X' 'test_'
            test_baseline_image = 'test_X25.tif'
            test_image_subset = [25, 130, 1]  # None, [10, 79, 4]

            test_template_padding = 7
            calib_template_padding = test_template_padding + 6  # self.sweep_param
            thresholding = 'manual'
            threshold_modifier = 3000
            test_threshold_modifier = threshold_modifier
            MIN_P_AREA = 3
            MAX_P_AREA = 100
            XY_DISPLACEMENT = [[0, 0]]

            # shared variables
            hide_variables = True
            if hide_variables:

                # file types
                filetype = '.tif'
                filetype_ground_truth = '.txt'

                # optics
                PARTICLE_DIAMETER = 2.15e-6
                DEMAG = 1
                MAGNIFICATION = 10
                NA = 0.3
                FOCAL_LENGTH = None
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
                SHAPE_TOL = 0.25
                SAME_ID_THRESH = 3
                BACKGROUND_SUBTRACTION = None  # 'median'  # 'min_value'  # 'min_value'
                OVERLAP_THRESHOLD = 8

                if self.collection_type == 'calibration':
                    CALIB_IMG_PATH = calib_img_path
                    CALIB_BASE_STRING = calib_base_string
                    CALIB_GROUND_TRUTH_PATH = None
                    CALIB_ID = calib_id

                    if self.sweep_param is not None:
                        CALIB_ID = CALIB_ID + '_{}'.format(self.sweep_param)
                        method = 'idpt'
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-{}_{}-{}'.format(method,
                                                                                                  self.sweep_method,
                                                                                                  self.sweep_param,
                                                                                                  ))
                    else:
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                    if not os.path.exists(CALIB_RESULTS_PATH):
                        os.makedirs(CALIB_RESULTS_PATH)

                    # calib dataset information
                    CALIB_SUBSET = calib_image_subset
                    CALIBRATION_Z_STEP_SIZE = calib_step_size
                    TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                    BASELINE_IMAGE = calib_baseline_image
                    IF_CALIB_IMAGE_STACK = 'mean'  # 'subset', 'mean'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # up to 2 but not including subset_f
                    CALIB_TEMPLATE_PADDING = calib_template_padding
                    CALIB_CROPPING_SPECS = cropping_specs
                    CALIB_PROCESSING_METHOD = None
                    CALIB_PROCESSING_FILTER_TYPE = None
                    CALIB_PROCESSING_FILTER_SIZE = None
                    CALIB_PROCESSING_METHOD2 = None
                    CALIB_PROCESSING_FILTER_TYPE2 = None
                    CALIB_PROCESSING_FILTER_SIZE2 = None
                    CALIB_PROCESSING = None
                    CALIB_THRESHOLD_METHOD = thresholding
                    CALIB_THRESHOLD_MODIFIER = threshold_modifier
                    CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
                    TEST_BASE_STRING = test_base_string
                    TEST_GROUND_TRUTH_PATH = None
                    TEST_ID = test_id

                    if self.sweep_param is not None:
                        TEST_ID = TEST_ID + '{}'.format(self.sweep_param)
                        method = 'idpt'
                        TEST_RESULTS_PATH = join(base_dir, 'results/test-{}_{}-{}'.format(method,
                                                                                          self.sweep_method,
                                                                                          self.sweep_param,
                                                                                          ))
                    else:
                        TEST_RESULTS_PATH = join(base_dir, 'results/test')
                    if not os.path.exists(TEST_RESULTS_PATH):
                        os.makedirs(TEST_RESULTS_PATH)

                    # test dataset information
                    TEST_SUBSET = test_image_subset
                    TRUE_NUM_PARTICLES_PER_IMAGE = 1
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 1]

                    TEST_PARTICLE_ID_IMAGE = test_baseline_image
                    TEST_TEMPLATE_PADDING = test_template_padding
                    TEST_CROPPING_SPECS = cropping_specs
                    TEST_PROCESSING_METHOD = None
                    TEST_PROCESSING_FILTER_TYPE = None
                    TEST_PROCESSING_FILTER_SIZE = None
                    TEST_PROCESSING_METHOD2 = None
                    TEST_PROCESSING_FILTER_TYPE2 = None
                    TEST_PROCESSING_FILTER_SIZE2 = None
                    TEST_PROCESSING = None
                    TEST_THRESHOLD_METHOD = thresholding
                    TEST_THRESHOLD_MODIFIER = test_threshold_modifier
                    TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}  # test_theory_threshold

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
            else:
                raise ValueError("Need those variables though...")

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == 'w18_c1_test3':

            base_dir = join('/Users/mackenzie/Desktop/Zipper/Testing/Wafer18', self.dataset)

            bkg_mean = 360
            bkg_noise = 45

            calib_dir = join('/Users/mackenzie/Desktop/Zipper/Testing/Wafer18', 'w18_c1_test1')
            calib_img_path = join(calib_dir, 'images/calibration')
            calib_step_size = 5
            calib_id = self.dataset + '_cal_'
            subset_i, subset_f = 0, 5

            test_img_path = join(base_dir, 'images/tests/{}'.format(self.sweep_param[0]))
            test_id = self.dataset + '_080323_'

            # shared setup
            dc = 0  # 16
            cropping_specs = {'xmin': 0 + dc, 'xmax': 512 - dc, 'ymin': 0 + dc, 'ymax': 512 - dc, 'pad': 10 + dc}

            calib_image_subset = [95, 175, 1]  # [47, 136, 10]  # None  # [1, 106, calib_step_size]  # [15, 85]
            calib_base_string = 'calib_'
            calib_baseline_image = 'calib_95.tif'

            test_base_string = 'test_X'  # 'test_X' 'test_'
            test_baseline_image = 'test_X1.tif'
            test_image_subset = None  # None, [10, 79, 4]

            test_template_padding = 8
            calib_template_padding = test_template_padding + 7  # self.sweep_param
            thresholding = 'manual'
            threshold_modifier = 3000
            test_threshold_modifier = threshold_modifier
            MIN_P_AREA = 3
            MAX_P_AREA = 120
            XY_DISPLACEMENT = [[-2, 0]]  # [[-3, 2]]  # the calib image must shift: 3 pixels LEFT; 2 pixels DOWN
            # NOTE: [[X, Y]] is applied to the calibration baseline locations. So, what dx/dy shifts calib to test?
            SAME_ID_THRESH = 9

            # shared variables
            hide_variables = True
            if hide_variables:

                # file types
                filetype = '.tif'
                filetype_ground_truth = '.txt'

                # optics
                PARTICLE_DIAMETER = 2.15e-6
                DEMAG = 1
                MAGNIFICATION = 10
                NA = 0.3
                FOCAL_LENGTH = None
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
                SHAPE_TOL = 0.25
                BACKGROUND_SUBTRACTION = None  # 'median'  # 'min_value'  # 'min_value'
                OVERLAP_THRESHOLD = 8

                if self.collection_type == 'calibration':
                    CALIB_IMG_PATH = calib_img_path
                    CALIB_BASE_STRING = calib_base_string
                    CALIB_GROUND_TRUTH_PATH = None
                    CALIB_ID = calib_id

                    if self.sweep_param is not None:
                        CALIB_ID = CALIB_ID + '_{}'.format(self.sweep_param)
                        method = 'idpt'
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-{}_{}-{}'.format(method,
                                                                                                  self.sweep_method,
                                                                                                  self.sweep_param,
                                                                                                  ))
                    else:
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                    if not os.path.exists(CALIB_RESULTS_PATH):
                        os.makedirs(CALIB_RESULTS_PATH)

                    # calib dataset information
                    CALIB_SUBSET = calib_image_subset
                    CALIBRATION_Z_STEP_SIZE = calib_step_size
                    TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                    BASELINE_IMAGE = calib_baseline_image
                    IF_CALIB_IMAGE_STACK = 'mean'  # 'subset', 'mean'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # up to 2 but not including subset_f
                    CALIB_TEMPLATE_PADDING = calib_template_padding
                    CALIB_CROPPING_SPECS = cropping_specs
                    CALIB_PROCESSING_METHOD = None
                    CALIB_PROCESSING_FILTER_TYPE = None
                    CALIB_PROCESSING_FILTER_SIZE = None
                    CALIB_PROCESSING_METHOD2 = None
                    CALIB_PROCESSING_FILTER_TYPE2 = None
                    CALIB_PROCESSING_FILTER_SIZE2 = None
                    CALIB_PROCESSING = None
                    CALIB_THRESHOLD_METHOD = thresholding
                    CALIB_THRESHOLD_MODIFIER = threshold_modifier
                    CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
                    TEST_BASE_STRING = test_base_string
                    TEST_GROUND_TRUTH_PATH = None
                    TEST_ID = test_id

                    if self.sweep_param is not None:
                        TEST_ID = TEST_ID + '{}'.format(self.sweep_param)
                        method = 'idpt'
                        TEST_RESULTS_PATH = join(base_dir, 'results/test-{}_{}-{}'.format(method,
                                                                                          self.sweep_method,
                                                                                          self.sweep_param,
                                                                                          ))
                    else:
                        TEST_RESULTS_PATH = join(base_dir, 'results/test')
                    if not os.path.exists(TEST_RESULTS_PATH):
                        os.makedirs(TEST_RESULTS_PATH)

                    # test dataset information
                    TEST_SUBSET = test_image_subset
                    TRUE_NUM_PARTICLES_PER_IMAGE = 1
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 1]

                    TEST_PARTICLE_ID_IMAGE = test_baseline_image
                    TEST_TEMPLATE_PADDING = test_template_padding
                    TEST_CROPPING_SPECS = cropping_specs
                    TEST_PROCESSING_METHOD = None
                    TEST_PROCESSING_FILTER_TYPE = None
                    TEST_PROCESSING_FILTER_SIZE = None
                    TEST_PROCESSING_METHOD2 = None
                    TEST_PROCESSING_FILTER_TYPE2 = None
                    TEST_PROCESSING_FILTER_SIZE2 = None
                    TEST_PROCESSING = None
                    TEST_THRESHOLD_METHOD = thresholding
                    TEST_THRESHOLD_MODIFIER = test_threshold_modifier
                    TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}  # test_theory_threshold

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
            else:
                raise ValueError("Need those variables though...")

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == 'w18_c1_5pT_test2':

            base_dir = join('/Users/mackenzie/Desktop/Zipper/Testing/Wafer18', self.dataset)

            bkg_mean = 360
            bkg_noise = 45

            calib_img_path = join(base_dir, 'calibration')  # , 'images/calibration'
            calib_step_size = 1
            calib_id = self.dataset + '_cal_'
            subset_i, subset_f = 0, 5

            test_img_path = join(base_dir, 'fin_tests/{}'.format(self.sweep_param[0]))
            test_id = self.dataset + '_080923_'

            """ 
            if test image set: "...fin_tests2...", the following parameters were used:
            
            calib_image_subset = [45, 125, 1]
            calib_baseline_image = 'calib_55.tif'
            test_template_padding = 6
            calib_template_padding = test_template_padding + 13
            threshold_modifier = 1300
            test_threshold_modifier = 850
            MIN_P_AREA = 30
            MAX_P_AREA = 450
            XY_DISPLACEMENT = [[0, 0]]
            SAME_ID_THRESH = 9
            """

            # shared setup
            dc = 0  # 16
            cropping_specs = {'xmin': 0 + dc, 'xmax': 512 - dc, 'ymin': 0 + dc, 'ymax': 512 - dc, 'pad': 10 + dc}

            calib_image_subset = [45, 125, 1]  # [47, 136, 10]  # None  # [1, 106, calib_step_size]  # [15, 85]
            calib_base_string = 'calib_'
            calib_baseline_image = 'calib_55.tif'

            test_base_string = 'test_X'  # 'test_X' 'test_'
            test_baseline_image = 'test_X1.tif'
            test_image_subset = None  # None, [10, 79, 4]

            test_template_padding = 6
            calib_template_padding = test_template_padding + 13  # self.sweep_param
            thresholding = 'manual'

            # NOTE: you should be identifying ~23-26 particles per image. If not, you should ajdust threshold params.
            threshold_modifier = 1300
            test_threshold_modifier = 850  # 1500  # threshold_modifier
            MIN_P_AREA = 30
            MAX_P_AREA = 450
            XY_DISPLACEMENT = [[0, 0]]  # [[-3, 2]]  # the calib image must shift: 3 pixels LEFT; 2 pixels DOWN
            # NOTE: [[X, Y]] is applied to the calibration baseline locations. So, what dx/dy shifts calib to test?
            SAME_ID_THRESH = 9

            # shared variables
            hide_variables = True
            if hide_variables:

                # file types
                filetype = '.tif'
                filetype_ground_truth = '.txt'

                # optics
                PARTICLE_DIAMETER = 2.15e-6
                DEMAG = 1
                MAGNIFICATION = 10
                NA = 0.3
                FOCAL_LENGTH = None
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
                SHAPE_TOL = 0.25
                BACKGROUND_SUBTRACTION = None  # 'median'  # 'min_value'  # 'min_value'
                OVERLAP_THRESHOLD = 8

                if self.collection_type == 'calibration':
                    CALIB_IMG_PATH = calib_img_path
                    CALIB_BASE_STRING = calib_base_string
                    CALIB_GROUND_TRUTH_PATH = None
                    CALIB_ID = calib_id

                    if self.sweep_param is not None:
                        CALIB_ID = CALIB_ID + '_{}'.format(self.sweep_param)
                        method = 'idpt'
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-{}_{}-{}'.format(method,
                                                                                                  self.sweep_method,
                                                                                                  self.sweep_param,
                                                                                                  ))
                    else:
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                    if not os.path.exists(CALIB_RESULTS_PATH):
                        os.makedirs(CALIB_RESULTS_PATH)

                    # calib dataset information
                    CALIB_SUBSET = calib_image_subset
                    CALIBRATION_Z_STEP_SIZE = calib_step_size
                    TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                    BASELINE_IMAGE = calib_baseline_image
                    IF_CALIB_IMAGE_STACK = 'mean'  # 'subset', 'mean'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # up to 2 but not including subset_f
                    CALIB_TEMPLATE_PADDING = calib_template_padding
                    CALIB_CROPPING_SPECS = cropping_specs
                    CALIB_PROCESSING_METHOD = None
                    CALIB_PROCESSING_FILTER_TYPE = None
                    CALIB_PROCESSING_FILTER_SIZE = None
                    CALIB_PROCESSING_METHOD2 = None
                    CALIB_PROCESSING_FILTER_TYPE2 = None
                    CALIB_PROCESSING_FILTER_SIZE2 = None
                    CALIB_PROCESSING = None
                    CALIB_THRESHOLD_METHOD = thresholding
                    CALIB_THRESHOLD_MODIFIER = threshold_modifier
                    CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
                    TEST_BASE_STRING = test_base_string
                    TEST_GROUND_TRUTH_PATH = None
                    TEST_ID = test_id

                    if self.sweep_param is not None:
                        TEST_ID = TEST_ID + '{}'.format(self.sweep_param)
                        method = 'idpt'
                        TEST_RESULTS_PATH = join(base_dir, 'results/test-{}_{}-{}'.format(method,
                                                                                          self.sweep_method,
                                                                                          self.sweep_param,
                                                                                          ))
                    else:
                        TEST_RESULTS_PATH = join(base_dir, 'results/test')
                    if not os.path.exists(TEST_RESULTS_PATH):
                        os.makedirs(TEST_RESULTS_PATH)

                    # test dataset information
                    TEST_SUBSET = test_image_subset
                    TRUE_NUM_PARTICLES_PER_IMAGE = 1
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 1]

                    TEST_PARTICLE_ID_IMAGE = test_baseline_image
                    TEST_TEMPLATE_PADDING = test_template_padding
                    TEST_CROPPING_SPECS = cropping_specs
                    TEST_PROCESSING_METHOD = None
                    TEST_PROCESSING_FILTER_TYPE = None
                    TEST_PROCESSING_FILTER_SIZE = None
                    TEST_PROCESSING_METHOD2 = None
                    TEST_PROCESSING_FILTER_TYPE2 = None
                    TEST_PROCESSING_FILTER_SIZE2 = None
                    TEST_PROCESSING = None
                    TEST_THRESHOLD_METHOD = thresholding
                    TEST_THRESHOLD_MODIFIER = test_threshold_modifier
                    TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}  # test_theory_threshold

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
            else:
                raise ValueError("Need those variables though...")

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == 'w18_c1_0pT_test1':

            base_dir = join('/Users/mackenzie/Desktop/Zipper/Testing/Wafer18', self.dataset)

            bkg_mean = 360
            bkg_noise = 45

            calib_img_path = join(base_dir, 'images/calibration')
            calib_step_size = 1
            calib_id = self.dataset + '_cal_'
            subset_i, subset_f = 0, 5

            test_img_path = join(base_dir, 'images/tests/{}'.format(self.sweep_param))
            test_id = self.dataset + '_080923_'

            # shared setup
            dc = 0  # 16
            cropping_specs = {'xmin': 0 + dc, 'xmax': 512 - dc, 'ymin': 0 + dc, 'ymax': 512 - dc, 'pad': 10 + dc}

            calib_image_subset = [10, 78, 1]  # [47, 136, 10]  # None  # [1, 106, calib_step_size]  # [15, 85]
            calib_base_string = 'calib_'
            calib_baseline_image = 'calib_19.tif'

            test_base_string = 'test_X'  # 'test_X' 'test_'
            test_baseline_image = 'test_X1.tif'
            test_image_subset = None  # None, [10, 79, 4]

            test_template_padding = 5
            calib_template_padding = test_template_padding + 12  # self.sweep_param
            thresholding = 'manual'

            # NOTE: you should be identifying ~23-26 particles per image. If not, you should ajdust threshold params.
            threshold_modifier = 2000
            test_threshold_modifier = threshold_modifier  # 1500  # threshold_modifier
            MIN_P_AREA = 15
            MAX_P_AREA = 250
            XY_DISPLACEMENT = [[0, 0]]  # [[-3, 2]]  # the calib image must shift: 3 pixels LEFT; 2 pixels DOWN
            # NOTE: [[X, Y]] is applied to the calibration baseline locations. So, what dx/dy shifts calib to test?
            SAME_ID_THRESH = 9

            # shared variables
            hide_variables = True
            if hide_variables:

                # file types
                filetype = '.tif'
                filetype_ground_truth = '.txt'

                # optics
                PARTICLE_DIAMETER = 2.15e-6
                DEMAG = 1
                MAGNIFICATION = 10
                NA = 0.3
                FOCAL_LENGTH = None
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
                SHAPE_TOL = 0.25
                BACKGROUND_SUBTRACTION = None  # 'median'  # 'min_value'  # 'min_value'
                OVERLAP_THRESHOLD = 8

                if self.collection_type == 'calibration':
                    CALIB_IMG_PATH = calib_img_path
                    CALIB_BASE_STRING = calib_base_string
                    CALIB_GROUND_TRUTH_PATH = None
                    CALIB_ID = calib_id

                    if self.sweep_param is not None:
                        CALIB_ID = CALIB_ID + '_{}'.format(self.sweep_param)
                        method = 'idpt'
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-{}_{}-{}'.format(method,
                                                                                                  self.sweep_method,
                                                                                                  self.sweep_param,
                                                                                                  ))
                    else:
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                    if not os.path.exists(CALIB_RESULTS_PATH):
                        os.makedirs(CALIB_RESULTS_PATH)

                    # calib dataset information
                    CALIB_SUBSET = calib_image_subset
                    CALIBRATION_Z_STEP_SIZE = calib_step_size
                    TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                    BASELINE_IMAGE = calib_baseline_image
                    IF_CALIB_IMAGE_STACK = 'mean'  # 'subset', 'mean'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # up to 2 but not including subset_f
                    CALIB_TEMPLATE_PADDING = calib_template_padding
                    CALIB_CROPPING_SPECS = cropping_specs
                    CALIB_PROCESSING_METHOD = None
                    CALIB_PROCESSING_FILTER_TYPE = None
                    CALIB_PROCESSING_FILTER_SIZE = None
                    CALIB_PROCESSING_METHOD2 = None
                    CALIB_PROCESSING_FILTER_TYPE2 = None
                    CALIB_PROCESSING_FILTER_SIZE2 = None
                    CALIB_PROCESSING = None
                    CALIB_THRESHOLD_METHOD = thresholding
                    CALIB_THRESHOLD_MODIFIER = threshold_modifier
                    CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
                    TEST_BASE_STRING = test_base_string
                    TEST_GROUND_TRUTH_PATH = None
                    TEST_ID = test_id

                    if self.sweep_param is not None:
                        TEST_ID = TEST_ID + '{}'.format(self.sweep_param)
                        method = 'idpt'
                        TEST_RESULTS_PATH = join(base_dir, 'results/test-{}_{}-{}'.format(method,
                                                                                          self.sweep_method,
                                                                                          self.sweep_param,
                                                                                          ))
                    else:
                        TEST_RESULTS_PATH = join(base_dir, 'results/test')
                    if not os.path.exists(TEST_RESULTS_PATH):
                        os.makedirs(TEST_RESULTS_PATH)

                    # test dataset information
                    TEST_SUBSET = test_image_subset
                    TRUE_NUM_PARTICLES_PER_IMAGE = 1
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 1]

                    TEST_PARTICLE_ID_IMAGE = test_baseline_image
                    TEST_TEMPLATE_PADDING = test_template_padding
                    TEST_CROPPING_SPECS = cropping_specs
                    TEST_PROCESSING_METHOD = None
                    TEST_PROCESSING_FILTER_TYPE = None
                    TEST_PROCESSING_FILTER_SIZE = None
                    TEST_PROCESSING_METHOD2 = None
                    TEST_PROCESSING_FILTER_TYPE2 = None
                    TEST_PROCESSING_FILTER_SIZE2 = None
                    TEST_PROCESSING = None
                    TEST_THRESHOLD_METHOD = thresholding
                    TEST_THRESHOLD_MODIFIER = test_threshold_modifier
                    TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}  # test_theory_threshold

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
            else:
                raise ValueError("Need those variables though...")

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == 'wA_c1_test2':
            # NOTE: this test location is near the edge so it has a unique calibration image collection.

            base_dir = join('/Users/mackenzie/Desktop/Zipper/Testing/wA', self.dataset)

            bkg_mean = 360
            bkg_noise = 45
            calib_img_path = join(base_dir, 'images/calibration')
            calib_step_size = 1
            calib_id = self.dataset + '_cal_'
            subset_i, subset_f = 0, 5

            test_img_path = join(base_dir, 'images/tests/{}/{}'.format(self.sweep_param[0], self.sweep_param[1]))
            test_id = self.dataset + '_072323_'

            # shared setup
            dc = 0  # 16
            cropping_specs = {'xmin': 0 + dc, 'xmax': 512 - dc, 'ymin': 0 + dc, 'ymax': 512 - dc, 'pad': 10 + dc}

            calib_image_subset = [90, 130, 1]  # [47, 136, 10]  # None  # [1, 106, calib_step_size]  # [15, 85]
            calib_base_string = 'calib_'
            calib_baseline_image = 'calib_90.tif'

            test_base_string = 'test_X'  # 'test_X' 'test_'
            test_baseline_image = 'test_X1.tif'
            test_image_subset = [1, 150, 1]  # None, [10, 79, 4]

            test_template_padding = 3
            calib_template_padding = test_template_padding + 7  # self.sweep_param
            thresholding = 'manual'
            threshold_modifier = 3000
            test_threshold_modifier = threshold_modifier
            MIN_P_AREA = 3
            MAX_P_AREA = 200
            XY_DISPLACEMENT = [[0, 0]]

            # shared variables
            hide_variables = True
            if hide_variables:

                # file types
                filetype = '.tif'
                filetype_ground_truth = '.txt'

                # optics
                PARTICLE_DIAMETER = 2.15e-6
                DEMAG = 1
                MAGNIFICATION = 10
                NA = 0.3
                FOCAL_LENGTH = None
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
                SHAPE_TOL = 0.25
                SAME_ID_THRESH = 3
                BACKGROUND_SUBTRACTION = None  # 'median'  # 'min_value'  # 'min_value'
                OVERLAP_THRESHOLD = 8

                if self.collection_type == 'calibration':
                    CALIB_IMG_PATH = calib_img_path
                    CALIB_BASE_STRING = calib_base_string
                    CALIB_GROUND_TRUTH_PATH = None
                    CALIB_ID = calib_id

                    if self.sweep_param is not None:
                        CALIB_ID = CALIB_ID + '_{}'.format(self.sweep_param)
                        method = 'idpt'
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-{}_{}-{}'.format(method,
                                                                                                  self.sweep_method,
                                                                                                  self.sweep_param,
                                                                                                  ))
                    else:
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                    if not os.path.exists(CALIB_RESULTS_PATH):
                        os.makedirs(CALIB_RESULTS_PATH)

                    # calib dataset information
                    CALIB_SUBSET = calib_image_subset
                    CALIBRATION_Z_STEP_SIZE = calib_step_size
                    TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                    BASELINE_IMAGE = calib_baseline_image
                    IF_CALIB_IMAGE_STACK = 'mean'  # 'subset', 'mean'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # up to 2 but not including subset_f
                    CALIB_TEMPLATE_PADDING = calib_template_padding
                    CALIB_CROPPING_SPECS = cropping_specs
                    CALIB_PROCESSING_METHOD = None
                    CALIB_PROCESSING_FILTER_TYPE = None
                    CALIB_PROCESSING_FILTER_SIZE = None
                    CALIB_PROCESSING_METHOD2 = None
                    CALIB_PROCESSING_FILTER_TYPE2 = None
                    CALIB_PROCESSING_FILTER_SIZE2 = None
                    CALIB_PROCESSING = None
                    CALIB_THRESHOLD_METHOD = thresholding
                    CALIB_THRESHOLD_MODIFIER = threshold_modifier
                    CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
                    TEST_BASE_STRING = test_base_string
                    TEST_GROUND_TRUTH_PATH = None
                    TEST_ID = test_id

                    if self.sweep_param is not None:
                        TEST_ID = TEST_ID + '{}'.format(self.sweep_param)
                        method = 'idpt'
                        TEST_RESULTS_PATH = join(base_dir, 'results/test-{}_{}-{}'.format(method,
                                                                                          self.sweep_method,
                                                                                          self.sweep_param,
                                                                                          ))
                    else:
                        TEST_RESULTS_PATH = join(base_dir, 'results/test')
                    if not os.path.exists(TEST_RESULTS_PATH):
                        os.makedirs(TEST_RESULTS_PATH)

                    # test dataset information
                    TEST_SUBSET = test_image_subset
                    TRUE_NUM_PARTICLES_PER_IMAGE = 1
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 1]

                    TEST_PARTICLE_ID_IMAGE = test_baseline_image
                    TEST_TEMPLATE_PADDING = test_template_padding
                    TEST_CROPPING_SPECS = cropping_specs
                    TEST_PROCESSING_METHOD = None
                    TEST_PROCESSING_FILTER_TYPE = None
                    TEST_PROCESSING_FILTER_SIZE = None
                    TEST_PROCESSING_METHOD2 = None
                    TEST_PROCESSING_FILTER_TYPE2 = None
                    TEST_PROCESSING_FILTER_SIZE2 = None
                    TEST_PROCESSING = None
                    TEST_THRESHOLD_METHOD = thresholding
                    TEST_THRESHOLD_MODIFIER = test_threshold_modifier
                    TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}  # test_theory_threshold

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
            else:
                raise ValueError("Need those variables though...")

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

        if self.dataset == 'w25_a1_5pT_test1':

            base_dir = join('/Users/mackenzie/Desktop/Zipper/Testing/Wafer25', self.dataset)

            bkg_mean = 360
            bkg_noise = 45

            calib_img_path = join(base_dir, 'images/calibration')
            calib_step_size = 1
            calib_id = self.dataset + '_cal_'
            subset_i, subset_f = 0, 5

            test_img_path = join(base_dir, 'images/tests/{}'.format(self.sweep_param))
            test_id = self.dataset + '_081023_'

            # shared setup
            dc = 0  # 16
            cropping_specs = {'xmin': 0 + dc, 'xmax': 512 - dc, 'ymin': 0 + dc, 'ymax': 512 - dc, 'pad': 10 + dc}

            calib_image_subset = [35, 80, 1]  # [47, 136, 10]  # None  # [1, 106, calib_step_size]  # [15, 85]
            calib_base_string = 'calib_'
            calib_baseline_image = 'calib_40.tif'

            test_base_string = 'test_X'  # 'test_X' 'test_'
            test_baseline_image = 'test_X1.tif'
            test_image_subset = None  # None, [10, 79, 4]

            test_template_padding = 3
            calib_template_padding = test_template_padding + 4  # self.sweep_param
            thresholding = 'manual'

            # NOTE: FIJI analysis suggests: threshold = 3000; min_area = 10.
            threshold_modifier = 2000
            test_threshold_modifier = threshold_modifier  # 1500  # threshold_modifier
            MIN_P_AREA = 10
            MAX_P_AREA = 250
            XY_DISPLACEMENT = [[0, 0]]  # [[-3, 2]]  # the calib image must shift: 3 pixels LEFT; 2 pixels DOWN
            # NOTE: [[X, Y]] is applied to the calibration baseline locations. So, what dx/dy shifts calib to test?
            SAME_ID_THRESH = 7

            # shared variables
            hide_variables = True
            if hide_variables:

                # file types
                filetype = '.tif'
                filetype_ground_truth = '.txt'

                # optics
                PARTICLE_DIAMETER = 2.15e-6
                DEMAG = 1
                MAGNIFICATION = 10
                NA = 0.3
                FOCAL_LENGTH = None
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
                SHAPE_TOL = 0.25
                BACKGROUND_SUBTRACTION = None  # 'median'  # 'min_value'  # 'min_value'
                OVERLAP_THRESHOLD = 8

                if self.collection_type == 'calibration':
                    CALIB_IMG_PATH = calib_img_path
                    CALIB_BASE_STRING = calib_base_string
                    CALIB_GROUND_TRUTH_PATH = None
                    CALIB_ID = calib_id

                    if self.sweep_param is not None:
                        CALIB_ID = CALIB_ID + '_{}'.format(self.sweep_param)
                        method = 'idpt'
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration-{}_{}-{}'.format(method,
                                                                                                  self.sweep_method,
                                                                                                  self.sweep_param,
                                                                                                  ))
                    else:
                        CALIB_RESULTS_PATH = join(base_dir, 'results/calibration')
                    if not os.path.exists(CALIB_RESULTS_PATH):
                        os.makedirs(CALIB_RESULTS_PATH)

                    # calib dataset information
                    CALIB_SUBSET = calib_image_subset
                    CALIBRATION_Z_STEP_SIZE = calib_step_size
                    TRUE_NUM_PARTICLES_PER_CALIB_IMAGE = 1
                    BASELINE_IMAGE = calib_baseline_image
                    IF_CALIB_IMAGE_STACK = 'mean'  # 'subset', 'mean'
                    TAKE_CALIB_IMAGE_SUBSET_MEAN = [subset_i, subset_f]  # up to 2 but not including subset_f
                    CALIB_TEMPLATE_PADDING = calib_template_padding
                    CALIB_CROPPING_SPECS = cropping_specs
                    CALIB_PROCESSING_METHOD = None
                    CALIB_PROCESSING_FILTER_TYPE = None
                    CALIB_PROCESSING_FILTER_SIZE = None
                    CALIB_PROCESSING_METHOD2 = None
                    CALIB_PROCESSING_FILTER_TYPE2 = None
                    CALIB_PROCESSING_FILTER_SIZE2 = None
                    CALIB_PROCESSING = None
                    CALIB_THRESHOLD_METHOD = thresholding
                    CALIB_THRESHOLD_MODIFIER = threshold_modifier
                    CALIB_THRESHOLD_PARAMS = {CALIB_THRESHOLD_METHOD: [CALIB_THRESHOLD_MODIFIER]}

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
                    TEST_BASE_STRING = test_base_string
                    TEST_GROUND_TRUTH_PATH = None
                    TEST_ID = test_id

                    if self.sweep_param is not None:
                        TEST_ID = TEST_ID + '{}'.format(self.sweep_param)
                        method = 'idpt'
                        TEST_RESULTS_PATH = join(base_dir, 'results/test-{}_{}-{}'.format(method,
                                                                                          self.sweep_method,
                                                                                          self.sweep_param,
                                                                                          ))
                    else:
                        TEST_RESULTS_PATH = join(base_dir, 'results/test')
                    if not os.path.exists(TEST_RESULTS_PATH):
                        os.makedirs(TEST_RESULTS_PATH)

                    # test dataset information
                    TEST_SUBSET = test_image_subset
                    TRUE_NUM_PARTICLES_PER_IMAGE = 1
                    IF_TEST_IMAGE_STACK = 'first'
                    TAKE_TEST_IMAGE_SUBSET_MEAN = [0, 1]

                    TEST_PARTICLE_ID_IMAGE = test_baseline_image
                    TEST_TEMPLATE_PADDING = test_template_padding
                    TEST_CROPPING_SPECS = cropping_specs
                    TEST_PROCESSING_METHOD = None
                    TEST_PROCESSING_FILTER_TYPE = None
                    TEST_PROCESSING_FILTER_SIZE = None
                    TEST_PROCESSING_METHOD2 = None
                    TEST_PROCESSING_FILTER_TYPE2 = None
                    TEST_PROCESSING_FILTER_SIZE2 = None
                    TEST_PROCESSING = None
                    TEST_THRESHOLD_METHOD = thresholding
                    TEST_THRESHOLD_MODIFIER = test_threshold_modifier
                    TEST_THRESHOLD_PARAMS = {TEST_THRESHOLD_METHOD: [TEST_THRESHOLD_MODIFIER]}  # test_theory_threshold

                    # similarity
                    STACKS_USE_RAW = True
                    MIN_STACKS = 0.95  # percent of calibration stack
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
            else:
                raise ValueError("Need those variables though...")

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
                                             true_number_of_particles=TRUE_NUM_PARTICLES_PER_CALIB_IMAGE,
                                             use_stack_id=self.use_stack_id
                                             )

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