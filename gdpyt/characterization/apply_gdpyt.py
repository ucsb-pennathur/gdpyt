from gdpyt import GdpytImageCollection, GdpytCalibrationSet
from os.path import join
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.morphology import disk, square
from sklearn.neighbors import NearestNeighbors


def calc_defocused_p_dia(z, magnification, numerical_aperture, d_p, lamb=600e-9, n_0=1.0003):
    """
    Return the particle image diameter (on the CCD sensor) as a function of z (from the focal plane)

    Parameters
    ----------
    z                   :   distance from focal plane
    magnification       :   magnification of optics
    numerical_aperture  :   numerical aperture of 20X objective lens
    d_p                 :   diameter of particle
    lamb                :   wavelength of light from particle to CCD sensor (lamb ~ 600e-9)
    n_0                 :   medium of refraction (air = 1.0003)

    Returns
    -------

    """
    d_e = magnification * np.sqrt(d_p**2 + 1.49 * lamb**2 * (n_0**2/numerical_aperture**2 - 1) + 4 * z**2 * (n_0**2/numerical_aperture**2 - 1)**-1)

    return d_e

def gdpyt_analyze_synthetics(
        measure_path,
        magnification,
        numerical_aperture,
        bkg_mean,
        bkg_noise,
        gain,
        cyl_focal_length,
        shape_tolerance,
        min_p_size,
        max_p_size,
        same_id_thresh,
        median_disk,
        infer_method='bccorr',
        min_cm=0.8,
        save_id='gdpyt_data',
        save_calib_plot=False,
        show_calib_plot=False,
):
    """
    Notes:
        Ideally, you'd like to:
            1. choose the calibration image set.
            2. choose the test image set.
            3. the function would compare measured and true values on the test image set using the calibration set.
            4. export the data.

    --- This is not implemented as of right now though ---

    """

    # define filepaths
    calib_img_path = join(measure_path, 'calibration_images')
    calib_txt_path = join(measure_path, 'calibration_input')
    calib_results_path = join(measure_path, 'calibration_results')

    # define filetypes
    filetype = '.tif'
    save_img_type = '.png'
    save_txt_type = '.csv'

    # define image processing
    if median_disk == 0 or median_disk is None:
        processing = {'none': []}
    else:
        processing = {'median': {'args': [disk(median_disk)]},
                      'gaussian': {'args': [], 'kwargs': dict(sigma=1, preserve_range=True)}}
    smooth_div_disk = 3 # size of disk for mean filter to smooth mask
    smooth_div = 100 # threshold_image = smoothed_threshold_image > smooth_div
    threshold_mod = 1.5 # initial threshold = mean(img) + mean(img) * threshold_mod
    threshold = {'otsu': []}
    #threshold = {'manual_smoothing': [smooth_div_disk, smooth_div, threshold_mod]} # 'otsu': []
    #threshold = {'manual': [threshold_mod]}

    # create image collection
    calib_col = GdpytImageCollection(calib_img_path,
                                     filetype,
                                     background_subtraction='min',
                                     processing_specs=processing,
                                     thresholding_specs=threshold,
                                     min_particle_size=min_p_size,
                                     max_particle_size=max_p_size,
                                     shape_tol=shape_tolerance,
                                     folder_ground_truth=calib_txt_path)

    # uniformize particle id's
    calib_col.uniformize_particle_ids(threshold=same_id_thresh)

    # Calibration image filename to z position dictionary
    name_to_z = {}
    for image in calib_col.images.values():
        name_to_z.update({image.filename: float(image.filename.split('_')[-1].split('.')[0])})  # 'calib_X.tif' to z = X
    calib_set = calib_col.create_calibration(name_to_z, dilate=True)  # Dilate: dilate images to slide template

    # plot calibration images with identified particles
    plot_calib_col_imgs = ['calib_-12.0.tif', 'calib_0.0.tif', 'calib_12.0.tif']
    fig, ax = plt.subplots(ncols=3, figsize=(12,4))
    for i, img in enumerate(plot_calib_col_imgs):
        num_particles = len(calib_col.images[img].particles)
        ax[i].imshow(calib_col.images[img].draw_particles(raw=False, thickness=1, draw_id=False, draw_bbox=False))
        ax[i].set_title(img + ', {} particles'.format(num_particles))
        ax[i].axis('off')
    plt.tight_layout
    if save_calib_plot is True:
        savepath = join(calib_results_path, save_id + '_calib_col.png')
        plt.savefig(savepath)
    if show_calib_plot is True:
        plt.show()

    # plot calibration stack for a single particle
    plot_calib_stack_particle_ids = [2]
    #fig, ax = plt.subplots(nrows=3, figsize=(12,12))
    for id in plot_calib_stack_particle_ids:
        calib_set.calibration_stacks[id].plot(imgs_per_row=6, fig=None, ax=None)
    plt.tight_layout
    if save_calib_plot is True:
        savepath = join(calib_results_path, save_id + '_calib_stack.png')
        plt.savefig(savepath)
    if show_calib_plot is True:
        plt.show()

    # test images on identical calibration set
    test_col = calib_col

    # Infer position using Barnkob's ('bccorr'), znccorr' (zero-normalized cross-corr.) or 'ncorr' (normalized cross-corr.)
    if infer_method == 'bccorr':
        test_col.infer_z(calib_set).bccorr(min_cm=min_cm)
    elif infer_method == 'znccorr':
        test_col.infer_z(calib_set).znccorr(min_cm=min_cm)
    elif infer_method == 'ncorr':
        test_col.infer_z(calib_set).nccorr(min_cm=min_cm)

    if show_calib_plot is True or save_calib_plot is True:
        sort_imgs = lambda x: int(x.split('calib_')[-1].split('.')[0])
        # Pass ids that should be displayed
        fig = test_col.plot_particle_coordinate_calibration()
        fig.suptitle(save_id)
        if save_calib_plot is True:
            savefigpath = join(calib_results_path, save_id + '_z_truez.png')
            fig.savefig(fname=savefigpath)
        if show_calib_plot:
            fig.show()

        fig = test_col.plot_particle_snr_and(second_plot='area')
        fig.suptitle(save_id)
        if save_calib_plot is True:
            savefigpath = join(calib_results_path, save_id + '_snr_area.png')
            fig.savefig(fname=savefigpath)
        if show_calib_plot:
            fig.show()

        fig = test_col.plot_particle_snr_and(second_plot='solidity')
        fig.suptitle(save_id)
        if save_calib_plot is True:
            savefigpath = join(calib_results_path, save_id + '_snr_solidity.png')
            fig.savefig(fname=savefigpath)
        if show_calib_plot:
            fig.show()

    img_id = []
    p_id = []
    true_x = []
    true_y = []
    true_z = []
    meas_x = []
    meas_y = []
    meas_z = []
    err_x = []
    err_y = []
    err_z = []
    median_size = []
    cm_threshold = []
    p_sim = []
    p_snr = []
    p_dia = []
    p_dia_def = []
    p_dia_meas = []
    p_area = []
    p_sol = []
    bkg_means = []
    bkg_noises = []
    gains = []
    cyl_focal_lengths = []


    for img in test_col.images.values():
        fname = img.filename[:-4]
        ground_truth = np.loadtxt(join(calib_txt_path, fname + '.txt'))  # [x, y, z, diameter] units: microns
        groundtruth_xy = ground_truth[:, 0:2]
        groundtruth_z = ground_truth[:, 2]

        for p in img.particles:
            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(groundtruth_xy)
            result = neigh.kneighbors([p.location])

            # --- particle signal to noise ratio ---
            p_signal_to_noise = p.snr

            # --- particle diameter (ground truth, predicted defocused, measured from area)
            p_diameter = ground_truth[result[1][0][0]][3]
            p_z = groundtruth_z[result[1][0][0]]
            p_def = calc_defocused_p_dia(z=p_z, magnification=magnification, numerical_aperture=numerical_aperture, d_p=p_diameter)
            p_meas = np.sqrt(p.area * 4 / np.pi)

            # --- X-Y uncertainty analysis ---
            uncertainty_x = groundtruth_xy[result[1][0][0]][0] - p.location[0]
            uncertainty_y = groundtruth_xy[result[1][0][0]][1] - p.location[1]

            # --- Z-uncertainty analysis ---
            uncertainty_z = groundtruth_z[result[1][0][0]] - p.z

            img_id.append(fname)
            p_id.append(int(p.id))
            true_x.append(float(groundtruth_xy[result[1][0][0]][0]))
            true_y.append(float(groundtruth_xy[result[1][0][0]][1]))
            true_z.append(float(groundtruth_z[result[1][0][0]]))
            meas_x.append(float(p.location[0]))
            meas_y.append(float(p.location[1]))
            meas_z.append(float(p.z))
            err_x.append(float(uncertainty_x))
            err_y.append(float(uncertainty_y))
            err_z.append(float(uncertainty_z))
            median_size.append(float(median_disk))
            cm_threshold.append(float(min_cm))
            if p.max_sim is not None:
                p_sim.append(float(p.max_sim))
            else:
                p_sim.append(float(0))
            p_snr.append(float(p_signal_to_noise))
            p_dia.append(float(p_diameter))
            p_dia_def.append(float(p_def))
            p_dia_meas.append(float(p_meas))
            p_area.append(float(p.area))
            p_sol.append(float(p.solidity))
            bkg_means.append(float(bkg_mean))
            bkg_noises.append(float(bkg_noise))
            gains.append(float(gain))
            cyl_focal_lengths.append(float(cyl_focal_length))

    gdpyt_data = {
        'img': img_id,
        'p_id': p_id,
        'true_x': true_x,
        'true_y': true_y,
        'true_z': true_z,
        'meas_x': meas_x,
        'meas_y': meas_y,
        'meas_z': meas_z,
        'err_x': err_x,
        'err_y': err_y,
        'err_z': err_z,
        'median_size': median_size,
        'cm_threshold': cm_threshold,
        'p_sim': p_sim,
        'p_snr': p_snr,
        'p_dia': p_dia,
        'p_dia_def': p_dia_def,
        'p_dia_meas': p_dia_meas,
        'p_area': p_area,
        'p_sol': p_sol,
        'bkg_mean': bkg_means,
        'bkg_noise': bkg_noises,
        'gain': gains,
        'cyl_f_l': cyl_focal_lengths
    }

    df = pd.DataFrame(data=gdpyt_data)
    df.sort_values(by=['true_z', 'p_id'])
    savepath = join(calib_results_path, save_id + '.csv')
    df.to_csv(path_or_buf=savepath, index=False)

# ----------------------------------------------------------------------------------------------------------------------

# image path
MEAS_PATHS = [r'/Users/mackenzie/Desktop/gdpyt-characterization/synthetic_no-noise',
                r'/Users/mackenzie/Desktop/gdpyt-characterization/synthetic_0.1-noise',
                r'/Users/mackenzie/Desktop/gdpyt-characterization/synthetic_0.2-noise',
                r'/Users/mackenzie/Desktop/gdpyt-characterization/synthetic_0.3-noise']


# synthetic particle generator data
MAGNIFCATION = 20
NA = 0.45
BKG_MEAN = 115
BKG_NOISES = [0, 11.5, 23, 34.5]
GAIN = 5
CYL_FOCAL_LENGTH = 0

#optics
WAVELENGTH = 600e-9
N_0 = 1.0003
LATERAL_RESOLLUTION = 16E-6
depth_of_focus = WAVELENGTH * N_0 / NA**2 + N_0 / (MAGNIFCATION * NA) * LATERAL_RESOLLUTION

# image pre-processing
SHAPE_TOL = 0.25         # None == take any shape; 1 == take perfectly circular shape only.
MIN_P_SIZE = 3          # minimum particle size (area: units are in pixels) (recommended: 5)
MAX_P_SIZE = 500        # maximum particle size (area: units are in pixels) (recommended: 200)
SAME_ID_THRESH = 7      # maximum tolerance in x- and y-directions for particle to have the same ID between images
MEDIAN_DISK = [2.5]         # size of median disk filter - [1, 1.5, 2, 2.5, 3, 4,]

# similarity
INFER_METHODS = ['bccorr'] # , 'znccorr', 'nccorr']
MIN_CM = 0.8

# display options
SHOW_CALIB_PLOT = True
SAVE_CALIB_PLOT = True

# run process
for index, m_path in enumerate(MEAS_PATHS):
    bkg_noise = BKG_NOISES[index]

    for med in MEDIAN_DISK:
        for inf_method in INFER_METHODS:


            savename = 'gdpyt_data_' + 'noise-' + str(bkg_noise) + '_median-' + str(med) + '_infer-' + str(inf_method)

            gdpyt_analyze_synthetics(
                    measure_path=m_path,
                    magnification=MAGNIFCATION,
                    numerical_aperture=NA,
                    bkg_mean=BKG_MEAN,
                    bkg_noise=bkg_noise,
                    gain=GAIN,
                    cyl_focal_length=CYL_FOCAL_LENGTH,
                    shape_tolerance=SHAPE_TOL,
                    min_p_size=MIN_P_SIZE,
                    max_p_size=MAX_P_SIZE,
                    same_id_thresh=SAME_ID_THRESH,
                    median_disk=med,
                    infer_method=inf_method,
                    min_cm=MIN_CM,
                    save_id=savename,
                    save_calib_plot=SAVE_CALIB_PLOT,
                    show_calib_plot=SHOW_CALIB_PLOT,
            )