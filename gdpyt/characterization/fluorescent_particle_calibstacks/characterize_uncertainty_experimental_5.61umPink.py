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
CALIB_PATH = '/Users/mackenzie/Desktop/BPE.Gen2/settings/optics/calib/particle3_5.61umPinkHighInt/calib'
CALIB_RESULTS_PATH = '/Users/mackenzie/Desktop/BPE.Gen2/settings/optics/calib/particle3_5.61umPinkHighInt/characterize_gdpyt'

EXPORT_RESULTS_PATH = '/Users/mackenzie/Desktop/BPE.Gen2/settings/optics/calib/particle3_5.61umPinkHighInt/characterize_gdpyt'

# display options
SHOW_CALIB_PLOTS = True
SAVE_CALIB_PLOTS = True
SHOW_PLOTS = True
SAVE_PLOTS = True

# dataset information
N_CAL = 81
MEASUREMENT_DEPTH = 81.0
MEASUREMENT_WIDTH = 412.9
TRUE_NUM_PARTICLES_PER_IMAGE = 13

# synthetic particle generator data
MAGNIFCATION = 20
NA = 0.45
FOCAL_LENGTH = 350
REF_INDEX_MEDIUM = 1
REF_INDEX_LENS = 1.5
PIXEL_SIZE = 16
PIXEL_DIM_X = 512
PIXEL_DIM_Y = 512
BKG_MEAN = 120
BKG_NOISES = 7
GAIN = 5
CYL_FOCAL_LENGTH = 0

# optics
PIXEL_TO_MICRON_SCALING = 0.8064516129  # units: microns per pixel for 20X objective measured from 25-um checkerboard
WAVELENGTH = 600e-9
LATERAL_RESOLLUTION = 16E-6
depth_of_focus = WAVELENGTH * REF_INDEX_MEDIUM / NA ** 2 + REF_INDEX_MEDIUM / (MAGNIFCATION * NA) * LATERAL_RESOLLUTION

# image pre-processing
SHAPE_TOL = 0.95  # None == take any shape; 1 == take perfectly circular shape only.
MIN_P_AREA = 6  # minimum particle size (area: units are in pixels) (recommended: 5)
MAX_P_AREA = 2500  # maximum particle size (area: units are in pixels) (recommended: 200)
SAME_ID_THRESH = 12  # maximum distance=sqrt(x**2 + y**2) for particle to have the same ID between images
MEDIAN_DISK = 5  # size of median disk filter - [1, 1.5, 2, 2.5, 3, 4,]
CALIB_PROCESSING = {'median': {'args': [disk(MEDIAN_DISK), None, 'wrap']}}
PROCESSING = {'median': {'args': [disk(MEDIAN_DISK), None, 'wrap']}}
MEDIAN_PERCENT = 0.1  # percent additional threshold value from median value
THRESHOLD = {'otsu': []}  # {'median_percent': [MEDIAN_PERCENT]} #

# similarity
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

CALIB_ID = 'Calib-5.61umPink-'
SAVE_CALIB_ID = 'Calib-5.61umPink-'
CALIB_IMG_PATH = CALIB_PATH
CALIB_IMG_STRING = 'calib_'

# create image collection
calib_col = GdpytImageCollection(CALIB_IMG_PATH,
                                 filetype,
                                 file_basestring=CALIB_IMG_STRING,
                                 # subset=[20, 27],
                                 stacks_use_raw=False,
                                 background_subtraction=None,
                                 processing_specs=CALIB_PROCESSING,
                                 thresholding_specs=THRESHOLD,
                                 min_particle_size=MIN_P_AREA,
                                 max_particle_size=MAX_P_AREA,
                                 shape_tol=SHAPE_TOL,
                                 same_id_threshold=SAME_ID_THRESH,
                                 true_num_particles=TRUE_NUM_PARTICLES_PER_IMAGE,
                                 template_padding=6,
                                 folder_ground_truth=None,
                                 if_img_stack_take='subset',
                                 take_subset_mean=[20, 23],
                                 inspect_contours_for_every_image=False)

# Calibration image filename to z position dictionary
name_to_z = {}
for image in calib_col.images.values():
    name_to_z.update(
        {image.filename: float(
            image.filename.split(CALIB_IMG_STRING)[-1].split('.')[0]) / N_CAL})  # 'calib_X.tif' to z = X

# create the calibration set consisting of calibration stacks for each particle
calib_set = calib_col.create_calibration(name_to_z, dilate=True, min_num_layers=75)  # Dilate: dilate images to slide template

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
    """
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
    if SAVE_CALIB_PLOTS is True:
        savefigpath = join(CALIB_RESULTS_PATH, SAVE_CALIB_ID + '_particles_area.png')
        plt.savefig(fname=savefigpath, bbox_inches='tight')
        plt.close()
    if SHOW_CALIB_PLOTS:
        plt.show()

    # plot snr and area from calibration collection image stats
    calib_col.plot_calib_col_image_stats(data=calib_col_image_stats)
    plt.suptitle(SAVE_CALIB_ID)
    if SAVE_CALIB_PLOTS is True:
        savefigpath = join(CALIB_RESULTS_PATH, SAVE_CALIB_ID + '_calib_col_snr_area.png')
        plt.savefig(fname=savefigpath, bbox_inches='tight')
        plt.close()
    if SHOW_CALIB_PLOTS:
        plt.show()

    # plot snr and area
    calib_col.plot_particle_snr_and(second_plot='area')
    plt.suptitle(SAVE_CALIB_ID)
    if SAVE_CALIB_PLOTS is True:
        savefigpath = join(CALIB_RESULTS_PATH, SAVE_CALIB_ID + '_snr_area.png')
        plt.savefig(fname=savefigpath, bbox_inches='tight')
        plt.close()
    if SHOW_CALIB_PLOTS:
        plt.show()

    # plot snr and solidity
    calib_col.plot_particle_snr_and(second_plot='solidity')
    plt.suptitle(SAVE_CALIB_ID)
    if SAVE_CALIB_PLOTS is True:
        savefigpath = join(CALIB_RESULTS_PATH, SAVE_CALIB_ID + '_snr_solidity.png')
        plt.savefig(fname=savefigpath, bbox_inches='tight')
        plt.close()
    if SHOW_CALIB_PLOTS:
        plt.show()

    # plot snr and percent of particles assigned valid z-coordinate
    calib_col.plot_particle_snr_and(second_plot='percent_measured')
    plt.suptitle(SAVE_CALIB_ID)
    if SAVE_CALIB_PLOTS is True:
        savefigpath = join(CALIB_RESULTS_PATH, SAVE_CALIB_ID + '_snr_percent_measured.png')
        plt.savefig(fname=savefigpath, bbox_inches='tight')
        plt.close()
    if SHOW_CALIB_PLOTS:
        plt.show()

    # plot calibration images with identified particles
    plot_calib_col_imgs = ['calib_3.tif', 'calib_6.tif', 'calib_9.tif', 'calib_12.tif', 'calib_15.tif', 'calib_18.tif',
                           'calib_21.tif', 'calib_24.tif']
    fig, ax = plt.subplots(ncols=8, figsize=(16, 4))
    for i, img in enumerate(plot_calib_col_imgs):
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
        
    """

    # plot calibration set's stack's self similarity
    fig = calib_set.plot_stacks_self_similarity(min_num_layers=80)
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
    for id in plot_calib_stack_particle_ids:
        save_calib_pid = SAVE_CALIB_ID + 'pid{}'.format(id)
        """
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
        calib_set.calibration_stacks[id].plot_3d_stack(intensity_percentile=(10, 99), stepsize=5, aspect_ratio=2.5)
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
            
        """

else:
    pass

# ------ ------ ------ ------ ------ ------- ------ ------

# ------ ------ CREATE TEST IMAGE COLLECTION ------ ------ ------

# create copy of calibration collection for test collection
test_col = GdpytImageCollection(CALIB_IMG_PATH,
                                filetype,
                                file_basestring=CALIB_IMG_STRING,
                                #subset=[20, 25],
                                processing_specs=CALIB_PROCESSING,
                                thresholding_specs=THRESHOLD,
                                min_particle_size=MIN_P_AREA,
                                max_particle_size=MAX_P_AREA,
                                shape_tol=SHAPE_TOL,
                                same_id_threshold=SAME_ID_THRESH,
                                true_num_particles=TRUE_NUM_PARTICLES_PER_IMAGE,
                                template_padding=6,
                                if_img_stack_take='first',
                                take_subset_mean=None)

same_id_most_correlated = []
for img in test_col.images.values():
    z_true = np.round(float(img.filename.split(CALIB_IMG_STRING)[-1].split('.')[0]) / N_CAL, 2)
    #if z_true > 0.7:
    #    continue
    for p in img.particles:
        p.set_true_z(z_true)
        SAVE_TEST_ID = 'test_calibstacks_z{}_pid{}'.format(z_true, p.id)
        num_stacks = len(calib_set.calibration_stacks.values())
        color = iter(cm.nipy_spectral(np.linspace(0, 1, num_stacks)))

        #fig, ax = plt.subplots(figsize=(8.75, 6))

        stack_cm_zs = []
        stack_interp_zs = []
        cms = []
        cmi = []
        for stack in calib_set.calibration_stacks.values():
            stack.infer_z(particle=p, function='barnkob_ccorr', min_cm=MIN_CM, infer_sub_image=SUB_IMAGE_INTERPOLATION)

            if p.z is not None and np.isnan(p.z) == False:
                sim_z = p.similarity_curve.iloc[:, 0]
                sim_cm = p.similarity_curve.iloc[:, 1]
                stack_cm_zs.append(sim_z[np.argmax(sim_cm)])
                cms.append(np.max(sim_cm))

                #c = next(color)
                #ax.plot(sim_z, sim_cm, color=c, linewidth=0.5, alpha=0.5, label=r'$c_m$ ' + str(stack.id))

                if SUB_IMAGE_INTERPOLATION:
                    interp_z = p.interpolation_curve.iloc[:, 0]
                    interp_cm = p.interpolation_curve.iloc[:, 1]
                    cmi.append(np.max(interp_cm))
                    stack_interp_zs.append(interp_z[np.argmax(interp_cm)])
                    #ax.plot(interp_z, interp_cm, color=c, linewidth=2)  # , label='interp ' + str(stack.id)

                    same_id_most_correlated.append([z_true, p.id, stack.id, sim_z[np.argmax(sim_cm)], np.max(sim_cm),
                                                    interp_z[np.argmax(interp_cm)], np.max(interp_cm)])

        # if every inferred z-coordinate is NaN, must skip the loop
        if len(cms) == 0:
            continue

        """
        # plot the true value
        ax.axvline(x=p.z_true, ymin=0, ymax=0.925, color='black', linestyle='--', alpha=0.85)
        ax.scatter(p.z_true, 1.00625, s=200, marker='*', color='magenta')

        # calculate stats
        best_guess_cm = np.round(stack_cm_zs[np.argmax(cms)], 3)
        best_guess_interp = np.round(stack_interp_zs[np.argmax(cmi)], 2)

        tp3cm = sorted(zip(cms, stack_cm_zs), reverse=True)[:3]
        tp3cm = [p[1] for p in tp3cm]
        top_three_cm = np.round(np.mean(tp3cm), 2)
        tp3ci = sorted(zip(cmi, stack_interp_zs), reverse=True)[:3]
        tp3ci = [p[1] for p in tp3ci]
        top_three_interp = np.round(np.mean(tp3ci), 2)

        mean_cm = np.round(np.mean(stack_cm_zs), 2)
        mean_interp = np.round(np.mean(stack_interp_zs), 2)
        std_cm = np.round(np.std(stack_cm_zs), 2)

        ax.set_title(r'$z_{true}$/h = ' + str(np.round(p.z_true, 2)) + r': $Stack_{ID}, p_{ID}$' + '= {}'.format(p.id), fontsize=13)
        ax.set_xlabel(r'$z$ / h', fontsize=12)
        #ax.set_xlim([mean_cm - 1.5*std_cm, mean_cm + 1.5*std_cm])
        ax.set_xlim([-0.01, 1.01])
        #ax.set_ylim([0.9495, 1.005])
        ax.set_ylim([0.795, 1.0125])
        ax.set_ylabel(r'$c_{m}$', fontsize=12)

        ax.grid(b=True, which='major', alpha=0.25)
        ax.grid(b=True, which='minor', alpha=0.125)

        textstr = '\n'.join((
                r'$z_{true}$ / h = ' + str(np.round(p.z_true, 2)),
                r'Best Guess $(z_{cc}, z_{interp.})$' + '= {}, {}'.format(best_guess_cm, best_guess_interp),
                r'$Mean_{3}$: $(z_{cc}, z_{interp.})$' + '= {}, {}'.format(top_three_cm, top_three_interp),
                r'$Mean_{all}$: $(z_{cc}, z_{interp.})$' + '= {}, {}'.format(mean_cm, mean_interp)
        ))

        if np.abs(p.z_true - top_three_cm) < 0.02:
            boxcolor = 'springgreen'
        else:
            boxcolor = 'lightcoral'
        props = dict(boxstyle='square', facecolor=boxcolor, alpha=0.25)
        ax.text(0.5, 0.19, textstr, verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes, bbox=props, fontsize=10)
        #ax.text(0.5, 0.25, , verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes, color='black', fontsize=12)
        #ax.text(0.5, 0.20, , verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes, color='black', fontsize=12)

        ax.legend(title=r'$CalibStack_{ID}$', loc='lower center', bbox_to_anchor=(0.5, 0.00625), ncol=int(np.round(num_stacks / 2)), fontsize=10)

        plt.tight_layout()
        savefigpath = join(CALIB_RESULTS_PATH, SAVE_TEST_ID + '_similarity.png')
        fig.savefig(fname=savefigpath, bbox_inches='tight')
        plt.close()
        """

stack_results = np.array(same_id_most_correlated)
dfstack = pd.DataFrame(data=stack_results, index=None, columns=['img_id', 'p_id', 'stack_id', 'z_cm', 'max_c_cm', 'z_interp', 'max_z_interp'])
savedata = join(EXPORT_RESULTS_PATH, 'gdpyt_every_stackid_results.xlsx')
dfstack.to_excel(savedata)

"""
# get test collection stats
test_col_stats = test_col.calculate_image_stats()

# get test collection inference local uncertainties
test_col_local_meas_quality = test_col.calculate_measurement_quality_local(true_xy=False)

# get test collection inference global uncertainties
test_col_global_meas_quality = test_col.calculate_measurement_quality_global(local=test_col_local_meas_quality)


# plot measurement accuracy against calibration stack
fig = test_col.plot_particle_coordinate_calibration(measurement_quality=test_col_local_meas_quality,
                                                    measurement_depth=MEASUREMENT_DEPTH,
                                                    true_xy=False,
                                                    measurement_width=MEASUREMENT_WIDTH)
fig.suptitle(SAVE_TEST_ID)
plt.tight_layout()
if SAVE_PLOTS is True:
    savefigpath = join(CALIB_RESULTS_PATH, SAVE_TEST_ID + '_calibration_line.png')
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
    savefigpath = join(CALIB_RESULTS_PATH, SAVE_TEST_ID + '_rmse_depth_uncertainty.png')
    fig.savefig(fname=savefigpath, bbox_inches='tight')
if SHOW_PLOTS:
    fig.show()

# plot local rmse uncertainty in real-coordinates
fig = test_col.plot_local_rmse_uncertainty(measurement_quality=test_col_local_meas_quality)
fig.suptitle(SAVE_TEST_ID)
plt.tight_layout()
if SAVE_PLOTS is True:
    savefigpath = join(CALIB_RESULTS_PATH, SAVE_TEST_ID + '_rmse_uncertainty.png')
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
        savefigpath = join(CALIB_RESULTS_PATH, SAVE_TEST_ID + '_correlation_pid{}.png'.format(P_INSPECT))
        fig.savefig(fname=savefigpath)
    if SHOW_PLOTS:
        fig.show()

# plot interpolation curves for every particle on image_id N_CAL//2
for IMG_INSPECT in [15, 35]:  # np.arange(1, 2, dtype=int):
    fig = test_col.plot_similarity_curve(sub_image=SUB_IMAGE_INTERPOLATION, method=INFER_METHODS, min_cm=MIN_CM,
                                         particle_id=None, image_id=IMG_INSPECT)
    if fig is not None:
        fig.suptitle(SAVE_TEST_ID + ': ' + r'image #/$N_{cal}$' + ' = {}'.format(IMG_INSPECT))
        plt.tight_layout()
        if SAVE_PLOTS is True:
            savefigpath = join(CALIB_RESULTS_PATH,
                               SAVE_TEST_ID + '_correlation_particles_in_img_{}.png'.format(IMG_INSPECT))
            fig.savefig(fname=savefigpath, bbox_inches='tight')
        if SHOW_PLOTS:
            fig.show()
        plt.close(fig)



# ------ ------ ------ ------ ------ ------- ------ ------

# ------------ ------ EXPORT RESULTS ------ ------ ------

# export data to text file
export_data = {
    'date_and_time': datetime.now(),
    'calib_image_path': CALIB_IMG_PATH,
    'inference_stack_id': stack_id,
    'n_cal': N_CAL,
    'n_cal_averaged_per_z': 5,
    #'noise_level': nl,
    'filter': 'median',
    'filter_disk': MEDIAN_DISK,
    'threshold': 'otsu',
    #'threshold_percent': MEDIAN_PERCENT,
    'zero_calib_stacks': ZERO_CALIB_STACKS,
    'infer': INFER_METHODS,
    'sub_image': SUB_IMAGE_INTERPOLATION,
    #'xy_mean_uncertainty': test_col_global_meas_quality['rmse_xy'],
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
    'fiji_img_bkg_mean': BKG_MEAN,
    'fiji_img_bkg_noise': BKG_NOISES,
    'microscope_gain': GAIN,
    'microscope_focal_length': FOCAL_LENGTH,
    'microscope_cyl_focal_length': CYL_FOCAL_LENGTH,
}

export_df = pd.DataFrame.from_dict(data=export_data, orient='index')
savedata = join(EXPORT_RESULTS_PATH, 'gdpyt_characterization_stackid{}_results.xlsx'.format(stack_id))
export_df.to_excel(savedata)

"""

# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# ----- ----- ----- ----- TEST IMAGE COLLECTION ----- ----- ----- ----- ----- ----- -----