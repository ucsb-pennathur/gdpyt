"""
This program tests the GDPyT --meta-- measurement accuracy..

Calibration Collection: 81 images of 5.61 um Pink SpheroTech High Intensity particles
Test Collection: 81 images of 5.61 um Pink SpheroTech High Intensity particles

Method:
1. Create calibration collection.
    1.1 assign 'z-true' from image filename.
    1.2 plot calibration figures and export statistics: particle area, SNR, etc.., image SNR, particle density, etc...
2. Copy calibration collection to test collection.
    2.1 For each calibration stack in the calibration set, infer the z-coordinate of all particles using that stack.
    2.2 plot test figures and export statistics: particle area, SNR, etc.., image SNR, particle density, etc...

Notes:
    1. The Otsu thresholding method is vastly superior to any median thresholding method I could come up with.
    2. There are options to plot every identified and passing contour to inspect contour identification. This is extremely useful in understanding what's going on.
    3. The results show a ridiculously high uncertainty due to the symmetry of the defocusing particle signature.

"""

from gdpyt import GdpytImageCollection
from os.path import join
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
import numpy as np
from skimage.morphology import disk

# ----- ----- ----- ----- DEFINE PARAMETERS ----- ----- ----- ----- ----- ----- -----

# define filepaths
CALIB_PATH = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/5.1umNR_HighInt_0.06XHg/10X_1Xmag'
CALIB_RESULTS_PATH = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/5.1umNR_HighInt_0.06XHg/results'

EXPORT_RESULTS_PATH = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/5.1umNR_HighInt_0.06XHg/results'

# display options
SHOW_CALIB_PLOTS = False
SAVE_CALIB_PLOTS = True
SHOW_PLOTS = False
SAVE_PLOTS = True

# dataset information
N_CAL = 181
MEASUREMENT_DEPTH = N_CAL
MEASUREMENT_WIDTH = 793.6
TRUE_NUM_PARTICLES_PER_IMAGE = 110

# synthetic particle generator data
MAGNIFCATION = 10
NA = 0.3
FOCAL_LENGTH = 350
REF_INDEX_MEDIUM = 1
REF_INDEX_LENS = 1.5
PIXEL_SIZE = 16
PIXEL_DIM_X = 512
PIXEL_DIM_Y = 512
BKG_MEAN = 110
BKG_NOISES = 6
GAIN = 5
CYL_FOCAL_LENGTH = 0

# optics
PIXEL_TO_MICRON_SCALING = 1.55 # units: microns per pixel; 20X objective w/ 0.5 demagnifier
    # Measured for 1X: 0.8064516129  # units: microns per pixel for 20X objective measured from 25-um checkerboard
WAVELENGTH = 600e-9
LATERAL_RESOLLUTION = 16E-6
depth_of_focus = WAVELENGTH * REF_INDEX_MEDIUM / NA ** 2 + REF_INDEX_MEDIUM / (MAGNIFCATION * NA) * LATERAL_RESOLLUTION

# image pre-processing
TEMPLATE_PADDING = 6
SHAPE_TOL = 0.25  # None == take any shape; 1 == take perfectly circular shape only.
MIN_P_AREA = 3  # minimum particle size (area: units are in pixels) (recommended: 5)
MAX_P_AREA = 1000  # maximum particle size (area: units are in pixels) (recommended: 500)
SAME_ID_THRESH = 10  # maximum distance=sqrt(x**2 + y**2) for particle to have the same ID between images
MEDIAN_DISK = 3  # size of median disk filter - [1, 1.5, 2, 2.5, 3, 4,]
CALIB_PROCESSING = {'median': {'args': [disk(MEDIAN_DISK), None, 'wrap']}} # {'none': {'args': []}}
PROCESSING = {'none': {'args': []}} # {'median': {'args': [disk(MEDIAN_DISK), None, 'wrap']}}
MEDIAN_PERCENT = 0.05  # percent additional threshold value from median value
THRESHOLD =  {'otsu': []}  # {'median_percent': [MEDIAN_PERCENT]} #

# similarity
MIN_STACKS = N_CAL * 1 // 2
ZERO_CALIB_STACKS = False
ZERO_STACKS_OFFSET = 0.5
INFER_METHODS = 'bccorr'
MIN_CM = 0.5
SUB_IMAGE_INTERPOLATION = True

# define filetypes
filetype = '.tif'

# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --------

# ----- ----- ----- ----- CALIBRATION ANALYSES - READ THESE NOTES CAREFULLY ----- ----- ----- -----
"""
Notes:
    here
"""

# ----- ----- ----- ----- ----- ----- CALIBRATION ANALYSIS ----- ----- ----- ----- ----- ----- ----- ----- -----

CALIB_ID = 'Calib-10X-5.1umNileRed-'
SAVE_CALIB_ID = 'Calib-10X-5.1umNileRed-'
CALIB_IMG_PATH = CALIB_PATH
CALIB_IMG_STRING = 'calib_'

# create image collection
calib_col = GdpytImageCollection(CALIB_IMG_PATH,
                                 filetype,
                                 file_basestring=CALIB_IMG_STRING,
                                 #subset=[20, 25],
                                 stacks_use_raw=False,
                                 background_subtraction=None,
                                 processing_specs=CALIB_PROCESSING,
                                 thresholding_specs=THRESHOLD,
                                 min_particle_size=MIN_P_AREA,
                                 max_particle_size=MAX_P_AREA,
                                 shape_tol=SHAPE_TOL,
                                 same_id_threshold=SAME_ID_THRESH,
                                 true_num_particles=TRUE_NUM_PARTICLES_PER_IMAGE,
                                 template_padding=TEMPLATE_PADDING,
                                 folder_ground_truth=None,
                                 if_img_stack_take='subset',
                                 take_subset_mean=[1, 2],
                                 inspect_contours_for_every_image=False)

# Calibration image filename to z position dictionary
name_to_z = {}
for image in calib_col.images.values():
    name_to_z.update(
        {image.filename: float(
            image.filename.split(CALIB_IMG_STRING)[-1].split('.')[0]) / N_CAL})  # 'calib_X.tif' to z = X

# create the calibration set consisting of calibration stacks for each particle
calib_set = calib_col.create_calibration(name_to_z, dilate=True, min_num_layers=MIN_STACKS)  # Dilate: dilate images to slide template

# set zero plane of the particles
if ZERO_CALIB_STACKS:
    calib_set.zero_stacks(offset=ZERO_STACKS_OFFSET)
    format_strings = True
else:
    format_strings = False

# get calibration collection image stats
img_keys = list(calib_col.images.keys())
img_keys = sorted(img_keys, key=lambda x: float(x.split(CALIB_IMG_STRING)[-1].split('.')[0]) / N_CAL)
img_zs = np.linspace(start=0, stop=len(img_keys), num=len(img_keys)) / len(img_keys)
dfs = pd.DataFrame(data=img_zs, index=img_keys, columns=['z'])
calib_col_image_stats = pd.concat([dfs, calib_col.image_stats], axis=1)

# get calibration collection mean stats
calib_col_stats = calib_col.calculate_image_stats()

# get calibration stacks data
calib_stack_data = calib_set.calculate_stacks_stats()

# plot calibration figures
if SAVE_CALIB_PLOTS or SHOW_CALIB_PLOTS:

    # plot number of particles identified in every image
    calib_col.plot_num_particles_per_image()
    plt.suptitle(SAVE_CALIB_ID)
    plt.tight_layout()
    if SAVE_CALIB_PLOTS is True:
        savefigpath = join(CALIB_RESULTS_PATH, SAVE_CALIB_ID + '_num_particles_per_image.png')
        plt.savefig(fname=savefigpath, bbox_inches='tight')
        plt.close()
    if SHOW_CALIB_PLOTS:
        plt.show()
    

    # plot particle area vs z/h for every particle
    calib_col.plot_particles_stats()
    plt.suptitle(SAVE_CALIB_ID)
    plt.tight_layout()
    if SAVE_CALIB_PLOTS is True:
        savefigpath = join(CALIB_RESULTS_PATH, SAVE_CALIB_ID + '_particles_area.png')
        plt.savefig(fname=savefigpath, bbox_inches='tight')
        plt.close()
    if SHOW_CALIB_PLOTS:
        plt.show()

    # plot snr and area from calibration collection image stats
    calib_col.plot_calib_col_image_stats(data=calib_col_image_stats)
    plt.suptitle(SAVE_CALIB_ID)
    plt.tight_layout()
    if SAVE_CALIB_PLOTS is True:
        savefigpath = join(CALIB_RESULTS_PATH, SAVE_CALIB_ID + '_calib_col_snr_area.png')
        plt.savefig(fname=savefigpath, bbox_inches='tight')
        plt.close()
    if SHOW_CALIB_PLOTS:
        plt.show()

    # plot snr and area
    calib_col.plot_particle_snr_and(second_plot='area')
    plt.suptitle(SAVE_CALIB_ID)
    plt.tight_layout()
    if SAVE_CALIB_PLOTS is True:
        savefigpath = join(CALIB_RESULTS_PATH, SAVE_CALIB_ID + '_snr_area.png')
        plt.savefig(fname=savefigpath, bbox_inches='tight')
        plt.close()
    if SHOW_CALIB_PLOTS:
        plt.show()

    # plot snr and solidity
    calib_col.plot_particle_snr_and(second_plot='solidity')
    plt.suptitle(SAVE_CALIB_ID)
    plt.tight_layout()
    if SAVE_CALIB_PLOTS is True:
        savefigpath = join(CALIB_RESULTS_PATH, SAVE_CALIB_ID + '_snr_solidity.png')
        plt.savefig(fname=savefigpath, bbox_inches='tight')
        plt.close()
    if SHOW_CALIB_PLOTS:
        plt.show()

    # plot snr and percent of particles assigned valid z-coordinate
    calib_col.plot_particle_snr_and(second_plot='percent_measured')
    plt.suptitle(SAVE_CALIB_ID)
    plt.tight_layout()
    if SAVE_CALIB_PLOTS is True:
        savefigpath = join(CALIB_RESULTS_PATH, SAVE_CALIB_ID + '_snr_percent_measured.png')
        plt.savefig(fname=savefigpath, bbox_inches='tight')
        plt.close()
    if SHOW_CALIB_PLOTS:
        plt.show()

    # plot calibration images with identified particles
    plot_calib_col_imgs = [10, 15, 20, 30, 40, 50, 60, 70]
    fig, ax = plt.subplots(ncols=8, figsize=(16, 4))
    for i, img in enumerate(plot_calib_col_imgs):
        img = CALIB_IMG_STRING + str(img) + '.tif'
        num_particles = len(calib_col.images[img].particles)
        ax[i].imshow(calib_col.images[img].draw_particles(raw=False, thickness=1, draw_id=True, draw_bbox=True))
        ax[i].set_title('z/h = {}'.format(np.round(calib_col.images[img].z, 2)))
        ax[i].axis('off')
    plt.suptitle(SAVE_CALIB_ID)
    plt.tight_layout()
    if SAVE_CALIB_PLOTS is True:
        savepath = join(CALIB_RESULTS_PATH, SAVE_CALIB_ID + '_calib_col.png')
        plt.savefig(savepath)
        plt.close()
    if SHOW_CALIB_PLOTS is True:
        plt.show()

    # plot calibration set's stack's self similarity
    fig = calib_set.plot_stacks_self_similarity(min_num_layers=MIN_STACKS)
    plt.suptitle(SAVE_CALIB_ID)
    plt.tight_layout()
    if SAVE_CALIB_PLOTS is True:
        savepath = join(CALIB_RESULTS_PATH, SAVE_CALIB_ID + '_calibset_stacks_self_similarity.png')
        plt.savefig(savepath)
        plt.close()
    if SHOW_CALIB_PLOTS is True:
        plt.show()

    # plot calibration stack for a single particle
    plot_calib_stack_particle_ids = calib_set.particle_ids
    for id in plot_calib_stack_particle_ids[1:5]:
        save_calib_pid = SAVE_CALIB_ID + 'pid{}'.format(id)

        calib_set.calibration_stacks[id].plot(imgs_per_row=9, fig=None, ax=None, format_string=format_strings)
        plt.suptitle(save_calib_pid)
        plt.tight_layout()
        if SAVE_CALIB_PLOTS is True:
            savepath = join(CALIB_RESULTS_PATH, save_calib_pid + '_calib_stack.png')
            plt.savefig(savepath)
            plt.close()
        if SHOW_CALIB_PLOTS is True:
            plt.show()

        # plot calibration stack for a single particle
        calib_set.calibration_stacks[id].plot(imgs_per_row=9, fill_contours=True, fig=None, ax=None,
                                              format_string=format_strings)
        plt.suptitle(save_calib_pid)
        plt.tight_layout()
        if SAVE_CALIB_PLOTS is True:
            savepath = join(CALIB_RESULTS_PATH, save_calib_pid + '_calib_stack_contours.png')
            plt.savefig(savepath)
            plt.close()
        if SHOW_CALIB_PLOTS is True:
            plt.show()

        # plot 3D calibration stack for a single particle
        calib_set.calibration_stacks[id].plot_3d_stack(intensity_percentile=(20, 99), stepsize=N_CAL//15, aspect_ratio=2.5)
        plt.suptitle(save_calib_pid)
        plt.tight_layout()
        if SAVE_CALIB_PLOTS is True:
            savepath = join(CALIB_RESULTS_PATH, save_calib_pid + '_calib_stack_3d.png')
            plt.savefig(savepath, bbox_inches="tight")
            plt.close()
        if SHOW_CALIB_PLOTS is True:
            plt.show()
        
        # plot calibration stack self similarity
        fig = calib_set.calibration_stacks[id].plot_self_similarity()
        plt.suptitle(save_calib_pid)
        plt.tight_layout()
        if SAVE_CALIB_PLOTS is True:
            savepath = join(CALIB_RESULTS_PATH, save_calib_pid + '_calib_stack_self_similarity.png')
            plt.savefig(savepath)
            plt.close()
        if SHOW_CALIB_PLOTS is True:
            plt.show()

else:
    pass

# ------ ------ ------ ------ ------ ------- ------ ------

# ------ ------ CREATE TEST IMAGE COLLECTION ------ ------ ------
TEST_IMG_PATH = CALIB_IMG_PATH
TEST_IMG_STRING = CALIB_IMG_STRING
TEST_RESULTS_PATH = CALIB_RESULTS_PATH
SAVE_TEST_ID = 'calib_test10X_inference'

# create copy of calibration collection for test collection
test_col = GdpytImageCollection(TEST_IMG_PATH,
                                filetype,
                                file_basestring=TEST_IMG_STRING,
                                #subset=[20, 25],
                                processing_specs=CALIB_PROCESSING,
                                thresholding_specs=THRESHOLD,
                                min_particle_size=MIN_P_AREA,
                                max_particle_size=MAX_P_AREA,
                                shape_tol=SHAPE_TOL,
                                same_id_threshold=SAME_ID_THRESH,
                                true_num_particles=TRUE_NUM_PARTICLES_PER_IMAGE,
                                template_padding=TEMPLATE_PADDING,
                                if_img_stack_take='first',
                                take_subset_mean=None)

test_col.infer_z(calib_set, infer_sub_image=SUB_IMAGE_INTERPOLATION).bccorr(min_cm=MIN_CM)

for img in test_col.images.values():
    z_true = np.round(float(img.filename.split(CALIB_IMG_STRING)[-1].split('.')[0]) / N_CAL, 2)
    for p in img.particles:
        p.set_true_z(z_true)

# get test collection stats
test_col_stats = test_col.calculate_image_stats()

# get test collection inference local uncertainties
test_col_local_meas_quality = test_col.calculate_measurement_quality_local(true_xy=False)

# get test collection inference global uncertainties
test_col_global_meas_quality = test_col.calculate_measurement_quality_global(local=test_col_local_meas_quality)


if SAVE_PLOTS or SHOW_PLOTS:
    # plot measurement accuracy against calibration stack
    fig = test_col.plot_particle_coordinate_calibration(measurement_quality=test_col_local_meas_quality,
                                                        measurement_depth=MEASUREMENT_DEPTH,
                                                        true_xy=False,
                                                        measurement_width=MEASUREMENT_WIDTH)
    fig.suptitle(SAVE_TEST_ID)
    plt.tight_layout()
    if SAVE_PLOTS is True:
        savefigpath = join(TEST_RESULTS_PATH, SAVE_TEST_ID + '_calibration_line.png')
        fig.savefig(fname=savefigpath, bbox_inches='tight')
    if SHOW_PLOTS:
        fig.show()

    # plot normalized local rmse uncertainty
    fig = test_col.plot_local_rmse_uncertainty(measurement_quality=test_col_local_meas_quality,
                                               measurement_depth=MEASUREMENT_DEPTH,
                                               measurement_width=MEASUREMENT_WIDTH)
    fig.suptitle(SAVE_TEST_ID)
    plt.tight_layout()
    if SAVE_PLOTS is True:
        savefigpath = join(TEST_RESULTS_PATH, SAVE_TEST_ID + '_rmse_depth_uncertainty.png')
        fig.savefig(fname=savefigpath, bbox_inches='tight')
    if SHOW_PLOTS:
        fig.show()

    # plot local rmse uncertainty in real-coordinates
    fig = test_col.plot_local_rmse_uncertainty(measurement_quality=test_col_local_meas_quality)
    fig.suptitle(SAVE_TEST_ID)
    plt.tight_layout()
    if SAVE_PLOTS is True:
        savefigpath = join(TEST_RESULTS_PATH, SAVE_TEST_ID + '_rmse_uncertainty.png')
        fig.savefig(fname=savefigpath, bbox_inches='tight')
    if SHOW_PLOTS:
        fig.show()

    # plot interpolation curves for every particle
    P_INSPECT = 0
    fig = test_col.plot_similarity_curve(sub_image=SUB_IMAGE_INTERPOLATION, method=INFER_METHODS, min_cm=MIN_CM,
                                         particle_id=P_INSPECT, image_id=None)
    if fig is not None:
        fig.suptitle(SAVE_TEST_ID + ': particle id {}'.format(P_INSPECT))
        plt.tight_layout()
        if SAVE_PLOTS is True:
            savefigpath = join(TEST_RESULTS_PATH, SAVE_TEST_ID + '_correlation_pid{}.png'.format(P_INSPECT))
            fig.savefig(fname=savefigpath)
        if SHOW_PLOTS:
            fig.show()

    # plot interpolation curves for every particle on image_id N_CAL//2
    for IMG_INSPECT in [15, 20]:  # np.arange(1, 2, dtype=int):
        fig = test_col.plot_similarity_curve(sub_image=SUB_IMAGE_INTERPOLATION, method=INFER_METHODS, min_cm=MIN_CM,
                                             particle_id=None, image_id=IMG_INSPECT)
        if fig is not None:
            fig.suptitle(SAVE_TEST_ID + ': ' + r'image #/$N_{cal}$' + ' = {}'.format(IMG_INSPECT))
            plt.tight_layout()
            if SAVE_PLOTS is True:
                savefigpath = join(TEST_RESULTS_PATH,
                                   SAVE_TEST_ID + '_correlation_particles_in_img_{}.png'.format(IMG_INSPECT))
                fig.savefig(fname=savefigpath, bbox_inches='tight')
            if SHOW_PLOTS:
                fig.show()
            plt.close(fig)

    # plot the stack similarity curve for every image, every particle, and every stack
    test_col.plot_every_image_particle_stack_similarity(calib_set=calib_set, plot=True, min_cm=MIN_CM,
                                                        save_results_path=CALIB_RESULTS_PATH,
                                                        infer_sub_image=SUB_IMAGE_INTERPOLATION,
                                                        measurement_depth=N_CAL)


# ------ ------ ------ ------ ------ ------- ------ ------

# ------------ ------ EXPORT RESULTS ------ ------ ------

# export data to text file
export_data = {
    'date_and_time': datetime.now(),
    'calib_image_path': CALIB_IMG_PATH,
    'test_image_path': TEST_IMG_PATH,
    'inference_stack_id': 'gdpyt-determined',
    'n_cal': N_CAL,
    'n_cal_averaged_per_z': 1,
    'filter': 'none',
    'threshold': 'otsu',
    'zero_calib_stacks': ZERO_CALIB_STACKS,
    'infer': INFER_METHODS,
    'sub_image': SUB_IMAGE_INTERPOLATION,
    'z_mean_uncertainty': test_col_global_meas_quality['rmse_z'],
    'percent_particles_idd': test_col_stats['percent_particles_idd'],
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
    'min_particle_area': calib_stack_data.min_particle_area.mean(),
    'min_particle_dia': calib_stack_data.min_particle_dia.mean(),
    'max_p_area_input': MAX_P_AREA,
    'max_particle_area': calib_stack_data.max_particle_area.mean(),
    'max_particle_dia': calib_stack_data.max_particle_dia.mean(),
    'calib_stack_avg_area': calib_stack_data.avg_area.mean(),
    'calib_stack_avg_snr': calib_stack_data.avg_snr.mean(),
    'test_col_avg_num_idd': test_col_global_meas_quality['num_idd'],
    'test_col_avg_num_invalid_z_measure': test_col_global_meas_quality['num_invalid_z_measure'],
    'test_col_mean_snr_filtered': test_col_stats['mean_snr_filtered'],
    'test_col_mean_signal': test_col_stats['mean_signal'],
    'test_col_mean_background': test_col_stats['mean_background'],
    'test_col_mean_std_background': test_col_stats['std_background'],
    'fiji_img_bkg_mean': BKG_MEAN,
    'fiji_img_bkg_noise': BKG_NOISES,
    'microscope_gain': GAIN,
    'microscope_focal_length': FOCAL_LENGTH,
    'microscope_cyl_focal_length': CYL_FOCAL_LENGTH,
}

export_df = pd.DataFrame.from_dict(data=export_data, orient='index')
savedata = join(EXPORT_RESULTS_PATH, 'gdpyt_characterization_stack{}_results.xlsx'.format('gdpyt-determined'))
export_df.to_excel(savedata)


# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# ----- ----- ----- ----- TEST IMAGE COLLECTION ----- ----- ----- ----- ----- ----- -----