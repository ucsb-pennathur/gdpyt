"""
This program tests the GDPyT measurement accuracy on... DataSet I
"""

from gdpyt import GdpytImageCollection
from os.path import join
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.morphology import disk

# ----- ----- ----- ----- DEFINE PARAMETERS ----- ----- ----- ----- ----- ----- -----

# define filepaths
CALIB_PATH = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Calibration'
CALIB_RESULTS_PATH = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Results/Calibration'

TEST_PATH = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Dataset_I'


EXPORT_RESULTS_PATH = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Results'

# display options
SHOW_CALIB_PLOTS = True
SAVE_CALIB_PLOTS = True
SHOW_PLOTS = True
SAVE_PLOTS = True

# define sweep
NOISE_LEVELS = ['1']  #

# dataset information
N_CAL = 50.0
N_TEST = 6
MEASUREMENT_DEPTH = 86.0
MEASUREMENT_WIDTH = 2000
TRUE_NUM_PARTICLES_PER_IMAGE = 361
TRUE_NUM_PARTICLES_TOTAL = N_TEST * TRUE_NUM_PARTICLES_PER_IMAGE

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

# optics
PIXEL_TO_MICRON_SCALING = 0.8064516129  # units: microns per pixel for 20X objective measured from 25-um checkerboard
WAVELENGTH = 600e-9
LATERAL_RESOLLUTION = 16E-6
depth_of_focus = WAVELENGTH * REF_INDEX_MEDIUM / NA ** 2 + REF_INDEX_MEDIUM / (MAGNIFCATION * NA) * LATERAL_RESOLLUTION

# image pre-processing
SHAPE_TOL = 0.95  # None == take any shape; 1 == take perfectly circular shape only.
MIN_P_AREA = 15  # minimum particle size (area: units are in pixels) (recommended: 5)
MAX_P_AREA = 2000  # maximum particle size (area: units are in pixels) (recommended: 200)
SAME_ID_THRESH = 8  # maximum tolerance in x- and y-directions for particle to have the same ID between images
MEDIAN_DISK = 5  # size of median disk filter - [1, 1.5, 2, 2.5, 3, 4,]
CALIB_PROCESSING = {'none': {}}
PROCESSING = {'median': {'args': [disk(MEDIAN_DISK), None, 'wrap']}}
MEDIAN_PERCENT = 0.05  # percent additional threshold value from median value
THRESHOLD = {'median_percent': [MEDIAN_PERCENT]}

# similarity
ZERO_CALIB_STACKS = False
ZERO_STACKS_OFFSET = 0.5
INFER_METHODS = 'bccorr'
MIN_CM = 0.75
SUB_IMAGE_INTERPOLATION = True

# define filetypes
filetype = '.tif'

# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

# ----- ----- ----- ----- CALIBRATION ANALYSES - READ THESE NOTES CAREFULLY ----- ----- ----- -----
"""
The "META" variable toggles between using:
    1. the same calibration image collection as the test image collection
    or,
    2. a specific calibration image collection tested against test image collections from each noise level.

Option #1 is called the "calibration meta characterization"
Option #2 is called the "calibration characterization"
"""

META = False
CALIB_IDD = '0'

# ----- ----- ----- CALIBRATION ANALYSIS - SWEEP CALIBRATION GRID NOISE LEVEL ----- ----- -----

for nl in NOISE_LEVELS: # TODO: the loop is not working. Every particle on following test collections get negative Cm values. Maybe the images are being double filtered?


    if META is True:
        CALIB_IDD = nl
        CALIB_ID = 'Calibration-noise-level' + nl
        SAVE_CALIB_ID = 'Calib-noise-level-' + nl
    else:
        CALIB_ID = 'Calibration-noise-level' + CALIB_IDD
        SAVE_CALIB_ID = 'Calib-noise-level-' + CALIB_IDD

    CALIB_IMG_PATH = join(CALIB_PATH, CALIB_ID, 'Calib-0050')

    # create image collection
    calib_col = GdpytImageCollection(CALIB_IMG_PATH,
                                     filetype,
                                     stacks_use_raw=False,
                                     background_subtraction=None,
                                     processing_specs=CALIB_PROCESSING,
                                     thresholding_specs=THRESHOLD,
                                     min_particle_size=MIN_P_AREA,
                                     max_particle_size=MAX_P_AREA,
                                     shape_tol=SHAPE_TOL,
                                     folder_ground_truth='standard_gdpt')

    # uniformize particle id's
    calib_col.uniformize_particle_ids(threshold=SAME_ID_THRESH)

    # Calibration image filename to z position dictionary
    name_to_z = {}
    for image in calib_col.images.values():
        name_to_z.update(
            {image.filename: float(image.filename.split('B00')[-1].split('.')[0]) / N_CAL})  # 'calib_X.tif' to z = X

    # create the calibration set consisting of calibration stacks for each particle
    calib_set = calib_col.create_calibration(name_to_z, dilate=True)  # Dilate: dilate images to slide template

    # set zero plane of the particles
    if ZERO_CALIB_STACKS:
        calib_set.zero_stacks(offset=ZERO_STACKS_OFFSET)
        format_strings = True
    else:
        format_strings = False

    # get calibration stacks data
    calib_stack_data = calib_set.calibration_stacks[0].calculate_stats(true_num_particles=N_CAL,
                                                                       measurement_volume=MEASUREMENT_DEPTH)

    # plot calibration figures
    if SAVE_CALIB_PLOTS or SHOW_CALIB_PLOTS:

        # plot calibration images with identified particles
        plot_calib_col_imgs = ['B00005.tif', 'B00010.tif', 'B00015.tif', 'B00020.tif', 'B00030.tif', 'B00035.tif',
                               'B00040.tif', 'B00045.tif']
        fig, ax = plt.subplots(ncols=8, figsize=(16, 4))
        for i, img in enumerate(plot_calib_col_imgs):
            num_particles = len(calib_col.images[img].particles)
            ax[i].imshow(calib_col.images[img].draw_particles(raw=False, thickness=1, draw_id=False, draw_bbox=False))
            ax[i].set_title('z/h = {}'.format(calib_col.images[img]._z))
            ax[i].axis('off')
        plt.suptitle(SAVE_CALIB_ID)
        plt.tight_layout()
        if SAVE_CALIB_PLOTS is True:
            savepath = join(CALIB_RESULTS_PATH, SAVE_CALIB_ID + '_calib_col.png')
            plt.savefig(savepath)
        if SHOW_CALIB_PLOTS is True:
            plt.show()

        # plot calibration stack for a single particle
        plot_calib_stack_particle_ids = [0]
        for id in plot_calib_stack_particle_ids:
            calib_set.calibration_stacks[id].plot(imgs_per_row=9, fig=None, ax=None, format_string=format_strings)
        plt.suptitle(SAVE_CALIB_ID)
        plt.tight_layout()
        if SAVE_CALIB_PLOTS is True:
            savepath = join(CALIB_RESULTS_PATH, SAVE_CALIB_ID + '_calib_stack.png')
            plt.savefig(savepath)
        if SHOW_CALIB_PLOTS is True:
            plt.show()

        # plot calibration stack for a single particle
        for id in plot_calib_stack_particle_ids:
            calib_set.calibration_stacks[id].plot(imgs_per_row=9, fill_contours=True, fig=None, ax=None,
                                                  format_string=format_strings)
        plt.suptitle(SAVE_CALIB_ID)
        plt.tight_layout()
        if SAVE_CALIB_PLOTS is True:
            savepath = join(CALIB_RESULTS_PATH, SAVE_CALIB_ID + '_calib_stack_contours.png')
            plt.savefig(savepath)
        if SHOW_CALIB_PLOTS is True:
            plt.show()

        # plot 3D calibration stack for a single particle
        for id in plot_calib_stack_particle_ids:
            calib_set.calibration_stacks[id].plot_3d_stack(intensity_percentile=(10, 99), stepsize=5, aspect_ratio=2.5)
        plt.suptitle(SAVE_CALIB_ID)
        plt.tight_layout()
        if SAVE_CALIB_PLOTS is True:
            savepath = join(CALIB_RESULTS_PATH, SAVE_CALIB_ID + '_calib_stack_3d.png')
            plt.savefig(savepath, bbox_inches="tight")
        if SHOW_CALIB_PLOTS is True:
            plt.show()

        # plot calibration stack self similarity
        for id in plot_calib_stack_particle_ids:
            calib_set.calibration_stacks[id].plot_self_similarity()
        plt.suptitle(SAVE_CALIB_ID)
        plt.tight_layout()
        if SAVE_CALIB_PLOTS is True:
            savepath = join(CALIB_RESULTS_PATH, SAVE_CALIB_ID + '_calib_stack_self_similarity.png')
            plt.savefig(savepath)
        if SHOW_CALIB_PLOTS is True:
            plt.show()

    else:
        pass

    # Turn off plotting and saving so figures are not recreated
    if META is False:
        SAVE_CALIB_PLOTS = False
        SHOW_CALIB_PLOTS = False

    # ------ ------ ------ ------ ------ -------

    # ------ CREATE TEST IMAGE COLLECTION ------
    TEST_ID = 'Measurement-grid-noise-level' + nl
    SAVE_TEST_ID = 'Calib-noise-level-' + CALIB_IDD + '-Meas-grid-noise-level-' + nl

    if META is True:
        test_col = calib_col  # test images on identical calibration set (meta characterization)
        TEST_IMG_PATH = CALIB_IMG_PATH
    else:
        TEST_IMG_PATH = join(TEST_PATH, TEST_ID, 'Images')  # create a new image collection
        TEST_GROUND_TRUTH_PATH = join(TEST_PATH, TEST_ID, 'Coordinates')
        test_col = GdpytImageCollection(TEST_IMG_PATH,
                                        filetype,
                                        subset=['B00', 1, 4],
                                        stacks_use_raw=False,
                                        background_subtraction=None,
                                        processing_specs=PROCESSING,
                                        thresholding_specs=THRESHOLD,
                                        min_particle_size=MIN_P_AREA,
                                        max_particle_size=MAX_P_AREA,
                                        shape_tol=SHAPE_TOL,
                                        folder_ground_truth=TEST_GROUND_TRUTH_PATH,
                                        measurement_depth=MEASUREMENT_DEPTH)

        # uniformize particle id's
        test_col.uniformize_particle_ids(threshold=SAME_ID_THRESH)

    # Infer position using Barnkob's ('bccorr'), znccorr' (zero-normalized cross-corr.) or 'ncorr' (normalized cross-corr.)
    test_col.infer_z(calib_set, infer_sub_image=SUB_IMAGE_INTERPOLATION).bccorr(min_cm=MIN_CM)

    # Zero the test collection z-coordinate

    # get test collection stats
    test_col_stats = test_col.calculate_image_stats()

    # get test collection inference local uncertainties
    test_col_local_meas_quality = test_col.calculate_measurement_quality_local()

    # get test collection inference global uncertainties
    test_col_global_meas_quality = test_col.calculate_measurement_quality_global(local=test_col_local_meas_quality)

    # plot test collection figures
    if SAVE_PLOTS is True or SHOW_PLOTS is True:

        # plot calibration images with identified particles
        plot_test_col_imgs = ['B00001.tif', 'B00002.tif', 'B00003.tif', 'B00004.tif']
        fig, ax = plt.subplots(ncols=len(plot_test_col_imgs), figsize=(len(plot_test_col_imgs)*4, 4))
        for i, img in enumerate(plot_test_col_imgs):
            num_particles = len(test_col.images[img].particles)
            ax[i].imshow(test_col.images[img].draw_particles(raw=False, thickness=1, draw_id=False, draw_bbox=False))
            ax[i].set_title('z/h = {}'.format(test_col.images[img].particles[0].z_true))
            ax[i].axis('off')
        plt.suptitle(SAVE_TEST_ID)
        plt.tight_layout()
        if SAVE_PLOTS is True:
            savepath = join(CALIB_RESULTS_PATH, SAVE_TEST_ID + '_test_col.png')
            plt.savefig(savepath)
        if SHOW_PLOTS is True:
            plt.show()

        # plot interpolation curves for every particle
        P_INSPECT = 0
        fig = test_col.plot_similarity_curve(sub_image=SUB_IMAGE_INTERPOLATION, method=INFER_METHODS, min_cm=MIN_CM,
                                             particle_id=P_INSPECT, image_id=None)
        if fig is not None:
            fig.suptitle(SAVE_TEST_ID + ': particle id {}'.format(P_INSPECT))
            if SAVE_PLOTS is True:
                savefigpath = join(CALIB_RESULTS_PATH, SAVE_TEST_ID + '_correlation_pid{}.png'.format(P_INSPECT))
                fig.savefig(fname=savefigpath)
            if SHOW_PLOTS:
                fig.show()

        # plot interpolation curves for every particle on image_id N_CAL//2
        for IMG_INSPECT in [0, 1]: # np.arange(1, 2, dtype=int):
            fig = test_col.plot_similarity_curve(sub_image=SUB_IMAGE_INTERPOLATION, method=INFER_METHODS, min_cm=MIN_CM,
                                                 particle_id=None, image_id=IMG_INSPECT)
            if fig is not None:
                fig.suptitle(
                    SAVE_TEST_ID + ': ' + r'image #/$N_{cal}$' + ' = {}'.format(IMG_INSPECT))
                plt.tight_layout()
                if SAVE_PLOTS is True:
                    savefigpath = join(CALIB_RESULTS_PATH,
                                       SAVE_TEST_ID + '_correlation_particles_in_img_{}.png'.format(IMG_INSPECT))
                    fig.savefig(fname=savefigpath, bbox_inches='tight')
                if SHOW_PLOTS:
                    fig.show()
                plt.close(fig)


        # plot measurement accuracy against calibration stack
        fig = test_col.plot_particle_coordinate_calibration(measurement_quality=test_col_local_meas_quality,
                                                            measurement_depth=MEASUREMENT_DEPTH,
                                                            measurement_width=MEASUREMENT_WIDTH)
        fig.suptitle(SAVE_TEST_ID)
        if SAVE_PLOTS is True:
            savefigpath = join(CALIB_RESULTS_PATH, SAVE_TEST_ID + '_calibration_line.png')
            fig.savefig(fname=savefigpath)
        if SHOW_PLOTS:
            fig.show()

        # plot normalized local rmse uncertainty
        fig = test_col.plot_local_rmse_uncertainty(measurement_quality=test_col_local_meas_quality,
                                                   measurement_depth=MEASUREMENT_DEPTH,
                                                   measurement_width=MEASUREMENT_WIDTH)
        fig.suptitle(SAVE_TEST_ID)
        if SAVE_PLOTS is True:
            savefigpath = join(CALIB_RESULTS_PATH, SAVE_TEST_ID + '_rmse_depth_uncertainty.png')
            fig.savefig(fname=savefigpath, bbox_inches='tight')
        if SHOW_PLOTS:
            fig.show()

        # plot local rmse uncertainty in real-coordinates
        fig = test_col.plot_local_rmse_uncertainty(measurement_quality=test_col_local_meas_quality)
        fig.suptitle(SAVE_TEST_ID)
        if SAVE_PLOTS is True:
            savefigpath = join(CALIB_RESULTS_PATH, SAVE_TEST_ID + '_rmse_uncertainty.png')
            fig.savefig(fname=savefigpath, bbox_inches='tight')
        if SHOW_PLOTS:
            fig.show()

        # plot snr and area
        fig = test_col.plot_particle_snr_and(second_plot='area')
        fig.suptitle(SAVE_TEST_ID)
        if SAVE_PLOTS is True:
            savefigpath = join(CALIB_RESULTS_PATH, SAVE_TEST_ID + '_snr_area.png')
            fig.savefig(fname=savefigpath, bbox_inches='tight')
        if SHOW_PLOTS:
            fig.show()

        # plot snr and solidity
        fig = test_col.plot_particle_snr_and(second_plot='solidity')
        fig.suptitle(SAVE_TEST_ID)
        if SAVE_PLOTS is True:
            savefigpath = join(CALIB_RESULTS_PATH, SAVE_TEST_ID + '_snr_solidity.png')
            fig.savefig(fname=savefigpath, bbox_inches='tight')
        if SHOW_PLOTS:
            fig.show()

        # plot snr and percent of particles assigned valid z-coordinate
        fig = test_col.plot_particle_snr_and(second_plot='percent_measured')
        fig.suptitle(SAVE_TEST_ID)
        if SAVE_PLOTS is True:
            savefigpath = join(CALIB_RESULTS_PATH, SAVE_TEST_ID + '_snr_percent_measured.png')
            fig.savefig(fname=savefigpath, bbox_inches='tight')
        if SHOW_PLOTS:
            fig.show()

    # export data to text file
    export_data = {
        'date_and_time': datetime.now(),
        'calib_image_path': CALIB_IMG_PATH,
        'test_image_path': TEST_IMG_PATH,
        'n_cal': N_CAL,
        'noise_level': nl,
        'filter': 'median',
        'filter_disk': MEDIAN_DISK,
        'threshold': 'median_percent',
        'threshold_percent': MEDIAN_PERCENT,
        'zero_calib_stacks': ZERO_CALIB_STACKS,
        'infer': INFER_METHODS,
        'sub_image': SUB_IMAGE_INTERPOLATION,
        'xy_mean_uncertainty': test_col_global_meas_quality['rmse_xy'],
        'z_mean_uncertainty': test_col_global_meas_quality['rmse_z'],
        'percent_particles_idd': calib_stack_data['percent_particles_idd'],
        'percent_particles_measured': test_col_global_meas_quality['percent_measure'],
        'mean_pixel_density': test_col_stats['mean_pixel_density'],
        'mean_particle_density': test_col_stats['mean_particle_density'],
        'measurement_depth': MEASUREMENT_DEPTH,
        'measurement_width': MEASUREMENT_WIDTH,
        'depth_of_focus': depth_of_focus,
        'min_cm': MIN_CM,
        'shape_tol': SHAPE_TOL,
        'same_id_thresh': SAME_ID_THRESH,
        'min_p_area_input': MIN_P_AREA,
        'min_particle_area': calib_stack_data['min_particle_area'],
        'min_particle_dia': calib_stack_data['min_particle_dia'],
        'max_p_area_input': MAX_P_AREA,
        'max_particle_area': calib_stack_data['max_particle_area'],
        'max_particle_dia': calib_stack_data['max_particle_dia'],
        'calib_stack_avg_area': calib_stack_data['avg_area'],
        'calib_stack_avg_snr': calib_stack_data['avg_snr'],
        'test_col_avg_num_idd': test_col_global_meas_quality['num_idd'],
        'test_col_avg_num_invalid_z_measure': test_col_global_meas_quality['num_invalid_z_measure'],
        'test_col_mean_snr_filtered': test_col_stats['mean_snr_filtered'],
        'test_col_mean_signal': test_col_stats['mean_signal'],
        'test_col_mean_background': test_col_stats['mean_background'],
        'test_col_mean_std_background': test_col_stats['std_background'],
        'synthetic_img_bkg_mean': BKG_MEAN,
        'synthetic_img_bkg_noise': nl,
        'synthetic_img_gain': GAIN,
        'synthetic_img_focal_length': FOCAL_LENGTH,
        'synthetic_img_cyl_focal_length': CYL_FOCAL_LENGTH,
    }

    export_df = pd.DataFrame.from_dict(data=export_data, orient='index')
    savedata = join(EXPORT_RESULTS_PATH, SAVE_TEST_ID + '_gdpt_characterization_results.xlsx')
    export_df.to_excel(savedata)

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