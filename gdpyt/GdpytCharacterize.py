# GdpytCharacterization
"""
Test harness for GDPyT characterization
"""

# imports
from gdpyt import GdpytImageCollection
from os.path import join
from datetime import datetime
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.morphology import disk


def test(calib_settings, test_settings, return_variables=None):

    # initial image analysis # TODO: write initial image analysis script
    """
    There should be an initial image analysis function that calculates values like:
        background mean
        background std

    These would be very helpful in deciding the processing filters and thresholding values.
    """

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
                                     hard_baseline=True,
                                     static_templates=calib_settings.inputs.static_templates,
                                     )

    # method for converting filenames to z-coordinates
    name_to_z = {}
    for image in calib_col.images.values():

        # if standard gdpt calibration dataset
        if calib_settings.inputs.ground_truth_file_path == 'standard_gdpt':
            N_CAL = float(len(calib_col.images))
            name_to_z.update(
                {image.filename: float(image.filename.split(calib_settings.inputs.image_base_string)[-1].split('.tif')[0]) *
                                 calib_col.measurement_range / (N_CAL - 1) - calib_col._calibration_stack_z_step})

        # For synthetic image sets, don't convert filename for z-coordinate because this reduces accuracy.
        else:
            # calib_settings.inputs == 'synthetic'
            name_to_z.update({image.filename: float(image.filename.split(calib_settings.inputs.image_base_string)[-1].split('.tif')[0])})

        """# Convert experimental calibration stacks with .tif files from: 'calib_1.tif' to 'calib_80.tif', for example.
        elif calib_settings.inputs == 'experimental':
            name_to_z.update(
                {image.filename: np.round((float(image.filename.split(calib_settings.inputs.image_base_string)[-1].split('.')[0]) - 1)
                                          + 1 / 2, 2)})"""


    # create the calibration set consisting of calibration stacks for each particle
    calib_set = calib_col.create_calibration(name_to_z=name_to_z,
                                             template_padding=calib_settings.processing.template_padding,
                                             min_num_layers=calib_settings.processing.min_layers_per_stack * len(calib_col.images),
                                             self_similarity_method=test_settings.z_assessment.infer_method,
                                             dilate=calib_settings.processing.dilate)

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
    single_particle_test = True
    if calib_settings.inputs.ground_truth_file_path == 'standard_gdpt':
        """ NOTE: I think the below code is wrong due to recent upgrades: 
        standard GDPT dataset with a single calibration image """
        test_collection_baseline = None
    elif single_particle_test is True:
        """ non-standard synthetic particle dataset with a single calibration image """
        test_collection_baseline = test_settings.inputs.baseline_image
        test_particle_id_image = None
    else:
        """ any analysis where the particle distribution in the test collection matches the calibration collection """
        test_collection_baseline = calib_set
        test_particle_id_image = test_settings.inputs.baseline_image

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
                                    measurement_depth=calib_col.measurement_range,
                                    template_padding=test_settings.processing.template_padding,
                                    if_img_stack_take=test_settings.inputs.if_image_stack,
                                    take_subset_mean=test_settings.inputs.take_image_stack_subset_mean_of,
                                    inspect_contours_for_every_image=test_settings.outputs.inspect_contours,
                                    baseline=test_collection_baseline,
                                    hard_baseline=True,
                                    static_templates=test_settings.inputs.static_templates,
                                    particle_id_image=test_particle_id_image,
                                    )

    # if performing characterization on synthetic dataset, set calib_set offset according to z @ minimum particle area.
    if test_settings.inputs.ground_truth_file_path is not None:
        pass
        # offset = calib_col.in_focus_z - test_col.in_focus_z
        # calib_set.zero_stacks(z_zero=test_col.in_focus_z, offset=offset, exclude_ids=None)

    # if performing meta-characterization on an experimental calibration set, set true_z for the test_collection
    else:
        test_col.set_true_z(image_to_z=name_to_z)

    # Infer the z-height of each particle
    test_col.infer_z(calib_set, infer_sub_image=test_settings.z_assessment.sub_image_interpolation).sknccorr(
        min_cm=test_settings.z_assessment.min_cm, use_stack=0)

    # export the particle coordinates
    test_coords = export_particle_coords(test_col, calib_settings, test_settings)

    # get test collection stats
    test_col_stats = test_col.calculate_image_stats()

    # get test collection inference local uncertainties
    test_col_local_meas_quality = test_col.calculate_measurement_quality_local(true_xy=test_settings.inputs.ground_truth_file_path)

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

    # all outputs
    return calib_col, calib_set, calib_col_image_stats, calib_stack_data, test_col, test_col_stats, \
           test_col_local_meas_quality, test_col_global_meas_quality


def plot_test(settings, test_col, test_col_stats, test_col_local_meas_quality, test_col_global_meas_quality):
    if settings.outputs.save_plots or settings.outputs.show_plots:

        # plot for a random selection of particles in the test collection
        if len(test_col.particle_ids) >= 10:
            P_INSPECTS = [int(p) for p in random.sample(set(test_col.particle_ids), 10)]
        else:
            P_INSPECTS = test_col.particle_ids
        for P_INSPECT in P_INSPECTS:

            # plot particle stack with z and true_z as subplot titles
            fig = test_col.plot_single_particle_stack(particle_id=P_INSPECT)
            if fig is not None:
                fig.suptitle(settings.outputs.save_id_string + ': particle id {}'.format(P_INSPECT))
                plt.tight_layout()
                if settings.outputs.save_plots is True:
                    savefigpath = join(settings.outputs.results_path,
                                       settings.outputs.save_id_string + '_correlation_pid{}.png'.format(P_INSPECT))
                    fig.savefig(fname=savefigpath)
                    plt.close()
                if settings.outputs.show_plots:
                    fig.show()

            # plot interpolation curves for every particle
            """
            fig = test_col.plot_similarity_curve(sub_image=settings.z_assessment.sub_image_interpolation, method=settings.z_assessment.infer_method, min_cm=settings.z_assessment.min_cm,
                                                 particle_id=P_INSPECT, image_id=None)
            if fig is not None:
                fig.suptitle(settings.outputs.save_id_string + ': particle id {}'.format(P_INSPECT))
                plt.tight_layout()
                if settings.outputs.save_plots is True:
                    savefigpath = join(settings.outputs.results_path, settings.outputs.save_id_string + '_correlation_pid{}.png'.format(P_INSPECT))
                    fig.savefig(fname=savefigpath)
                    plt.close()
                if settings.outputs.show_plots:
                    fig.show()
            """

        # plot test collection images with particles
        if len(test_col.files) >= 4:
            plot_test_col_imgs = [img_id for img_id in random.sample(set(test_col.files), 4)]
            fig, ax = plt.subplots(ncols=4, figsize=(12, 10))
            for i, img in enumerate(plot_test_col_imgs):
                ax[i].imshow(test_col.images[img].draw_particles(raw=False, thickness=1, draw_id=True, draw_bbox=True))
                img_formatted = np.round(float(img.split(settings.inputs.image_base_string)[-1].split('.tif')[0]), 3)
                ax[i].set_title('{}'.format(img_formatted))
                ax[i].axis('off')
            plt.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savepath = join(settings.outputs.results_path, settings.outputs.save_id_string + '_test_col.png')
                plt.savefig(savepath)
                plt.close()
            if settings.outputs.show_plots is True:
                plt.show()

        # plot normalized local rmse uncertainty
        fig = test_col.plot_local_rmse_uncertainty(measurement_quality=test_col_local_meas_quality,
                                                   measurement_depth=test_col.measurement_depth,
                                                   measurement_width=settings.optics.field_of_view)
        fig.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path, settings.outputs.save_id_string + '_rmse_depth_uncertainty.png')
            fig.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            fig.show()

        # plot normalized local rmse uncertainty and correlation value
        fig = test_col.plot_local_rmse_uncertainty(measurement_quality=test_col_local_meas_quality,
                                                   measurement_depth=test_col.measurement_depth,
                                                   measurement_width=settings.optics.field_of_view,
                                                   second_plot='cm')
        fig.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path, settings.outputs.save_id_string + '_rmse_depth_uncertainty_and_cm.png')
            fig.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            fig.show()

        # plot normalized local rmse uncertainty and valid z-measurements
        fig = test_col.plot_local_rmse_uncertainty(measurement_quality=test_col_local_meas_quality,
                                                   measurement_depth=test_col.measurement_depth,
                                                   measurement_width=settings.optics.field_of_view,
                                                   second_plot='num_valid_z_measure')
        fig.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path, settings.outputs.save_id_string + '_rmse_depth_uncertainty_and_valid_z.png')
            fig.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            fig.show()

        # plot local rmse uncertainty in real-coordinates
        fig = test_col.plot_local_rmse_uncertainty(measurement_quality=test_col_local_meas_quality)
        fig.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path, settings.outputs.save_id_string + '_rmse_uncertainty.png')
            fig.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            fig.show()

        # plot local rmse uncertainty and correlation value
        fig = test_col.plot_local_rmse_uncertainty(measurement_quality=test_col_local_meas_quality, second_plot='cm')
        fig.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path,
                               settings.outputs.save_id_string + '_rmse_uncertainty_and_cm.png')
            fig.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            fig.show()

        # plot normalized local rmse uncertainty and valid z-measurements
        fig = test_col.plot_local_rmse_uncertainty(measurement_quality=test_col_local_meas_quality,
                                                   second_plot='num_valid_z_measure')
        fig.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path,
                               settings.outputs.save_id_string + '_rmse_uncertainty_and_valid_z.png')
            fig.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            fig.show()

        # plot particle area vs z/h for every particle
        fig = test_col.plot_particles_stats()
        plt.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path,
                               settings.outputs.save_id_string + '_particles_area.png')
            plt.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            plt.show()

        # plot particle area vs z/h for every particle
        fig = test_col.plot_particles_stats(stat='img_num_particles')
        plt.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path,
                               settings.outputs.save_id_string + '_number_particles.png')
            plt.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            plt.show()

        # plot snr and area
        test_col.plot_particle_snr_and(second_plot='area')
        plt.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path, settings.outputs.save_id_string + '_snr_area.png')
            plt.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            plt.show()

        # plot snr and percent of particles assigned valid z-coordinate
        test_col.plot_particle_snr_and(second_plot='percent_measured')
        plt.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path, settings.outputs.save_id_string + '_snr_percent_measured.png')
            plt.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            plt.show()

        # plot snr and percent of particles assigned valid z-coordinate
        test_col.plot_particle_snr_and(second_plot='cm')
        plt.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path, settings.outputs.save_id_string + '_snr_cm_max_sim.png')
            plt.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            plt.show()

        # plot interpolation curves for every particle on image_id N_CAL//2
        """
        for IMG_INSPECT in np.arange(start=0, stop=len(test_col.images), step=10):
            fig = test_col.plot_similarity_curve(sub_image=settings.z_assessment.sub_image_interpolation, method=settings.z_assessment.infer_method, min_cm=settings.z_assessment.min_cm,
                                                 particle_id=None, image_id=IMG_INSPECT)
            if fig is not None:
                fig.suptitle(settings.outputs.save_id_string + ': ' + r'image #/$N_{cal}$' + ' = {}'.format(IMG_INSPECT))
                plt.tight_layout()
                if settings.outputs.save_plots is True:
                    savefigpath = join(settings.outputs.results_path,
                                       settings.outputs.save_id_string + '_correlation_particles_in_img_{}.png'.format(IMG_INSPECT))
                    fig.savefig(fname=savefigpath, bbox_inches='tight')
                if settings.outputs.show_plots:
                    fig.show()
                plt.close(fig)
        """

        # plot the particle image PSF-model-based z-calibration function
        """
        test_col.plot_gaussian_ax_ay(plot_type='all')
        plt.suptitle(settings.outputs.save_id_string + '_all_p_ids')
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path,
                               settings.outputs.save_id_string + '_Gaussian_fit_axy_all_pids.png')
            plt.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            plt.show()
        """

def plot_calibration(settings, test_settings, calib_col, calib_set, calib_col_image_stats, calib_col_stats, calib_stack_data):
    # show/save plots
    if settings.outputs.show_plots or settings.outputs.save_plots:

        """
        per-collection plots
        """
        # plot the baseline image with particle ID's
        if calib_col.baseline is not None:
            fig = calib_col.plot_baseline_image_and_particle_ids()
            plt.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, settings.outputs.save_id_string + '_calibration_baseline_image.png')
                plt.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                plt.show()

        # plot number of particles identified in every image: '_num_particles_per_image.png'
        fig = calib_col.plot_num_particles_per_image()
        plt.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path, settings.outputs.save_id_string + '_num_particles_per_image.png')
            plt.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            plt.show()

        # plot particle area vs z or z/h for every particle: '_particles_area.png'
        fig = calib_col.plot_particles_stats()
        plt.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path, settings.outputs.save_id_string + '_particles_area.png')
            plt.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            plt.show()

        # plot snr and area: '_snr_area.png'
        calib_col.plot_particle_snr_and(second_plot='area')
        plt.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path, settings.outputs.save_id_string + '_snr_area.png')
            plt.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            plt.show()

        # plot snr and solidity: '_snr_solidity.png'
        calib_col.plot_particle_snr_and(second_plot='solidity')
        plt.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path, settings.outputs.save_id_string + '_snr_solidity.png')
            plt.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            plt.show()

        # plot snr and percent of particles assigned valid z-coordinate: '_snr_percent_measured.png'
        calib_col.plot_particle_snr_and(second_plot='percent_measured')
        plt.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path, settings.outputs.save_id_string + '_snr_percent_measured.png')
            plt.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            plt.show()

        # plot calibration images with identified particles: '_calib_col.png'
        plot_calib_col_imgs = [index for index in random.sample(set(calib_col.files), 4)]
        fig, ax = plt.subplots(ncols=4, figsize=(12, 6))
        for i, img in enumerate(plot_calib_col_imgs):
            ax[i].imshow(calib_col.images[img].draw_particles(raw=False, thickness=1, draw_id=True, draw_bbox=True))
            ax[i].set_title('z = {}'.format(np.round(calib_col.images[img].z, 2)))
            ax[i].axis('off')
        plt.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savepath = join(settings.outputs.results_path, settings.outputs.save_id_string + '_calib_col{}.png'.format(i))
            plt.savefig(savepath)
            plt.close()
        if settings.outputs.show_plots is True:
            plt.show()

        # plot every calibration image with identified particles: '_calib_col.png'
        plot_calib_col_imgs = calib_col.files
        for i, img in enumerate(plot_calib_col_imgs):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(calib_col.images[img].draw_particles(raw=False, thickness=1, draw_id=True, draw_bbox=True))
            ax.set_title('{}'.format(img))
            plt.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savepath = join(settings.outputs.results_path, settings.outputs.save_id_string + '_calib_col{}.png'.format(i))
                plt.savefig(savepath)
                plt.close()
            if settings.outputs.show_plots is True:
                plt.show()

        # plot calibration set's stack's self similarity: '_calibset_stacks_self_similarity_{}.png'
        """
        fig = calib_set.plot_stacks_self_similarity(min_num_layers=settings.processing.min_layers_per_stack)
        plt.suptitle(settings.outputs.save_id_string + ', infer: {}'.format(test_settings.z_assessment.infer_method))
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savepath = join(settings.outputs.results_path, settings.outputs.save_id_string + '_calibset_stacks_self_similarity_{}.png'.format(test_settings.z_assessment.infer_method))
            plt.savefig(savepath)
            plt.close()
        if settings.outputs.show_plots is True:
            plt.show()
        """

        """
        per-particle (in calibration set) plots:
            1. plot calibration stack and contour outline for a single particle: '_filtered_calib_stack.png'
            2. plot calibration stack and filled contour for a single particle: '_calib_stack_contours.png'
            3. plot calibration stack self similarity: '_calib_stack_self_similarity_{}.png'
            4. plot 3D calibration stack for a single particle: '_calib_stack_3d.png'
            5. plot adjacent similarities with template images: '_calib_stack_index_{}_adjacent_similarity_{}.png'
            6. plot Gaussian fitted ax and ay: '_Gaussian_fit_axy.png'
            7. plot particle SNR and area: '_snr_area.png'
            8. plot theoretical and measured particle signal: '_signal_theory_and_measure.png'
            9. plot theoretical and experimentally-measured particle diameters: '_diameter_theory_and_measure.png'
        """
        # choose particle ID's at random from calibration set
        if len(calib_set.particle_ids) < 10:
            plot_calib_stack_particle_ids = calib_set.particle_ids
        else:
            plot_calib_stack_particle_ids = [pid for pid in random.sample(set(calib_set.particle_ids), 10)]

        for id in plot_calib_stack_particle_ids:

            # save ID for this set of plots
            save_calib_pid = settings.outputs.save_id_string + '_pid{}'.format(id)

            # plot calibration stack and contour outline for a single particle: '_filtered_calib_stack.png'
            calib_set.calibration_stacks[id].plot_calib_stack(imgs_per_row=9, fig=None, ax=None, format_string=False)
            plt.suptitle(save_calib_pid)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savepath = join(settings.outputs.results_path, save_calib_pid + '_filtered_calib_stack.png')
                plt.savefig(savepath)
                plt.close()
            if settings.outputs.show_plots is True:
                plt.show()

            # plot calibration stack and filled contour for a single particle: '_calib_stack_contours.png'
            """
            calib_set.calibration_stacks[id].plot_calib_stack(imgs_per_row=9, fill_contours=True, fig=None, ax=None,
                                                  format_string=False)
            plt.suptitle(save_calib_pid)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savepath = join(settings.outputs.results_path, save_calib_pid + '_calib_stack_contours.png')
                plt.savefig(savepath)
                plt.close()
            if settings.outputs.show_plots is True:
                plt.show()
                """

            # plot calibration stack self similarity: '_calib_stack_self_similarity_{}.png'
            fig = calib_set.calibration_stacks[id].plot_self_similarity()
            plt.suptitle(save_calib_pid + ', infer: {}'.format(test_settings.z_assessment.infer_method))
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savepath = join(settings.outputs.results_path, save_calib_pid + '_calib_stack_no_interp_self_similarity_{}.png'.format(test_settings.z_assessment.infer_method))
                plt.savefig(savepath)
                plt.close()
            if settings.outputs.show_plots is True:
                plt.show()

            # plot particle SNR and area: '_snr_area.png'
            fig = calib_col.plot_particle_snr_and(second_plot='area', particle_id=id)
            plt.suptitle(save_calib_pid)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, save_calib_pid + '_snr_area.png')
                plt.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                plt.show()

            # plot theoretical and measured particle signal: '_signal_theory_and_measure.png'
            fig = calib_col.plot_particle_signal(optics=settings.optics, collection_image_stats=calib_col_image_stats,
                                                 particle_id=id)
            plt.suptitle(save_calib_pid)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, save_calib_pid + '_signal_theory_and_measure.png')
                plt.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                plt.show()

            # plot theoretical and measured particle diameter: '_diameter_theory_and_measure.png'
            """
            fig = calib_col.plot_particle_signal(collection_image_stats=calib_col_image_stats, optics=settings.optics,
                                                   particle_id=id)
            plt.suptitle(save_calib_pid)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, save_calib_pid + '_diameter_theory_and_measure.png')
                plt.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                plt.show()
            """

            # plot 3D calibration stack for a single particle: '_calib_stack_3d.png'
            """calib_set.calibration_stacks[id].plot_3d_stack(intensity_percentile=(20, 99), stepsize=len(calib_col.images) // 10,
                                                           aspect_ratio=2.5)
            plt.suptitle(save_calib_pid)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savepath = join(settings.outputs.results_path, save_calib_pid + '_calib_stack_3d.png')
                plt.savefig(savepath, bbox_inches="tight")
                plt.close()
            if settings.outputs.show_plots is True:
                plt.show()"""

            # plot adjacent similarities with template images: '_calib_stack_index_{}_adjacent_similarity_{}.png'
            """indices = [ind for ind in random.sample(set(np.arange(2, len(calib_col.files))), 4)]
            for indice in indices:
                fig = calib_set.calibration_stacks[id].plot_adjacent_self_similarity(index=[indice])
                plt.suptitle(save_calib_pid + ', infer: {}'.format(test_settings.z_assessment.infer_method))
                plt.tight_layout()
                if settings.outputs.save_plots is True:
                    savepath = join(settings.outputs.results_path, save_calib_pid + '_calib_stack_index_{}_adjacent_similarity_{}.png'.format(indice, test_settings.z_assessment.infer_method))
                    plt.savefig(savepath)
                    plt.close()
                if settings.outputs.show_plots is True:
                    plt.show()"""

            # plot Gaussian fitted ax and ay: '_Gaussian_fit_axy.png'
            """calib_col.plot_gaussian_ax_ay(plot_type='one', p_inspect=[id])
            plt.suptitle(save_calib_pid)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, save_calib_pid + '_Gaussian_fit_axy.png')
                plt.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                plt.show()"""

        """
        Plots in development:
            * plot the particle image PSF-model-based z-calibration function
        """
        """
        # plot the particle image PSF-model-based z-calibration function
        calib_col.plot_gaussian_ax_ay(plot_type='all')
        plt.suptitle(settings.outputs.save_id_string + '_all_p_ids')
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path,
                               settings.outputs.save_id_string + '_Gaussian_fit_axy_all_pids.png')
            plt.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            plt.show()

        calib_col.plot_gaussian_ax_ay(plot_type='mean')
        plt.suptitle(settings.outputs.save_id_string + '_mean_p_ids')
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path,
                               settings.outputs.save_id_string + '_Gaussian_fit_axy_mean_pids.png')
            plt.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            plt.show()
        """

def assess_every_particle_and_stack_id(settings, calib_set, test_col):
        # plot the stack similarity curve for every image, every particle, and every stack
        test_col.plot_every_image_particle_stack_similarity(calib_set=calib_set, plot=True, min_cm=settings.z_assessment.min_cm,
                                                            save_results_path=settings.outputs.results_path,
                                                            infer_sub_image=settings.z_assessment.sub_image_interpolation,
                                                            measurement_depth=test_col.measurement_depth)


def export_results_and_settings(export='test', calib_settings=None, calib_col=None, calib_stack_data=None, calib_col_stats=None,
                                test_settings=None, test_col=None, test_col_global_meas_quality=None, test_col_stats=None):

    # export data to text file
    export_data = {
        'date_and_time': datetime.now(),
    }

    optics_dict = {
        'magnification': calib_settings.optics.magnification,
        'numerical_aperture': calib_settings.optics.numerical_aperture,
        'field_of_view': calib_settings.optics.field_of_view,
        'depth_of_field': calib_settings.optics.depth_of_field,
        'z_range': calib_settings.optics.z_range,
        'wavelength': calib_settings.optics.wavelength,
        'particle_diameter': calib_settings.optics.particle_diameter,
        'pixels_per_particle_in_focus': calib_settings.optics.pixels_per_particle_in_focus,
        'pixel_size': calib_settings.optics.pixel_size,
        'pixel_dim_x': calib_settings.optics.pixel_dim_x,
        'pixel_dim_y': calib_settings.optics.pixel_dim_y,
        'microns_per_pixel': calib_settings.optics.microns_per_pixel,
        'focal_length': calib_settings.optics.focal_length,
        'ref_index_medium': calib_settings.optics.ref_index_medium,
        'ref_index_lens': calib_settings.optics.ref_index_lens,
        'calib_img_bkg_mean': calib_settings.optics.bkg_mean,
        'calib_img_bkg_noise': calib_settings.optics.bkg_noise,
        'test_img_bkg_mean': test_settings.optics.bkg_mean,
        'test_img_bkg_noise': test_settings.optics.bkg_noise,
        'microscope_gain': calib_settings.optics.gain,
        'microscope_cyl_focal_length': calib_settings.optics.cyl_focal_length,
        'points_per_pixel': calib_settings.optics.points_per_pixel,
        'n_rays': calib_settings.optics.n_rays,
    }

    calib_settings_dict = {
        'calib_image_path': calib_settings.inputs.image_path,
        'calib_base_string': calib_settings.inputs.image_base_string,
        'calib_ground_truth_image_path': calib_settings.inputs.ground_truth_file_path,
        'calib_image_subset': calib_settings.inputs.image_subset,
        'calib_z_step_size': calib_settings.inputs.calibration_z_step_size,
        'calib_if_image_stack': calib_settings.inputs.if_image_stack,
        'calib_take_image_stack_mean_of': calib_settings.inputs.take_image_stack_subset_mean_of,
        'calib_results_path': calib_settings.outputs.results_path,
        'calib_save_plots': calib_settings.outputs.save_plots,
        'calib_save_id_string': calib_settings.outputs.save_id_string,
        'calib_template_padding': calib_settings.processing.template_padding,
        'calib_dilate': calib_settings.processing.dilate,
        'calib_shape_tolerance': calib_settings.processing.shape_tolerance,
        'calib_overlap_threshold': calib_settings.processing.overlap_threshold,
        'calib_same_id_threshold': calib_settings.processing.same_id_threshold_distance,
        'calib_min_particle_area': calib_settings.processing.min_particle_area,
        'calib_max_particle_area': calib_settings.processing.max_particle_area,
        'calib_min_layers_per_stack': calib_settings.processing.min_layers_per_stack,
        'calib_zero_calib_stacks': calib_settings.processing.zero_calib_stacks,
        'calib_zero_stacks_offset': calib_settings.processing.zero_stacks_offset,
        'calib_stacks_use_raw': calib_settings.processing.stacks_use_raw,
        'calib_background_subtraction': calib_settings.processing.background_subtraction,
        'calib_cropping_params': calib_settings.processing.cropping_params,
        'calib_processing_method': calib_settings.processing.processing_method,
        'calib_processing_filter_type': calib_settings.processing.processing_filter_type,
        'calib_processing_filter_size': calib_settings.processing.processing_filter_size,
        'calib_threshold_method': calib_settings.processing.threshold_method,
        'calib_threshold_modifier': calib_settings.processing.threshold_modifier,
        'calib_filter_params': calib_settings.processing.processing_params,
        'calib_threshold_params': calib_settings.processing.threshold_params,
    }

    calib_col_dict = {
        'calib_col_number_of_particles': len(calib_col.particle_ids),
        'calib_col_true_number_of_particles': calib_col.true_num_particles,
        'calib_col_particle_ids': calib_col.particle_ids,
        'calib_col_number_of_images': calib_col.num_images,
        'calib_col_number_of_images_averaged_per_z': calib_col.num_images_if_mean,
        'calib_col_measurement_depth': calib_col.measurement_depth,
    }

    calib_col_stats_dict = {
        'calib_col_mean_snr_filtered': calib_col_stats['mean_snr_filtered'],
        'calib_col_mean_signal': calib_col_stats['mean_signal'],
        'calib_col_mean_background': calib_col_stats['mean_background'],
        'calib_col_mean_noise': calib_col_stats['std_background'],
        'calib_col_mean_particle_density': calib_col_stats['mean_particle_density'],
        'calib_col_mean_pixel_density': calib_col_stats['mean_pixel_density'],
        'calib_col_percent_particles_idd': calib_col_stats['percent_particles_idd'],
        'calib_col_true_num_particles': calib_col_stats['true_num_particles'],
    }

    calib_stack_data_dict = {
        'calib_stack_particle_ids': calib_stack_data.particle_id.unique(),
        'calib_stack_mean_number_of_layers': calib_stack_data.layers.mean(),
        'calib_stack_mean_min_particle_area': calib_stack_data.min_particle_area.mean(),
        'calib_stack_mean_min_particle_dia': calib_stack_data.min_particle_dia.mean(),
        'calib_stack_mean_max_particle_area': calib_stack_data.max_particle_area.mean(),
        'calib_stack_mean_max_particle_dia': calib_stack_data.max_particle_dia.mean(),
        'calib_stack_mean_avg_area': calib_stack_data.avg_area.mean(),
        'calib_stack_mean_avg_snr': calib_stack_data.avg_snr.mean(),
    }

    if export == 'calibration':

        dicts = [optics_dict, calib_settings_dict, calib_col_dict, calib_col_stats_dict, calib_stack_data_dict]

        for d in dicts:
            export_data.update(d)

        export_df = pd.DataFrame.from_dict(data=export_data, orient='index')

        savedata = join(calib_settings.outputs.results_path, 'gdpyt_calibration_{}_{}.xlsx'.format(calib_settings.inputs.image_collection_id, export_data['date_and_time']))
        export_df.to_excel(savedata)

        return export_df

    test_settings_dict = {
        'inference_stack_id': 'gdpyt-determined',  # TODO: fix this
        'infer': test_settings.z_assessment.infer_method,
        'sub_image': test_settings.z_assessment.sub_image_interpolation,
        'min_cm': test_settings.z_assessment.min_cm,
        'test_image_path': test_settings.inputs.image_path,
        'test_base_string': test_settings.inputs.image_base_string,
        'test_ground_truth_image_path': test_settings.inputs.ground_truth_file_path,
        'test_image_subset': test_settings.inputs.image_subset,
        'test_if_image_stack': test_settings.inputs.if_image_stack,
        'test_take_image_stack_mean_of': test_settings.inputs.take_image_stack_subset_mean_of,
        'test_results_path': test_settings.outputs.results_path,
        'test_save_plots': test_settings.outputs.save_plots,
        'test_save_id_string': test_settings.outputs.save_id_string,
        'test_template_padding': test_settings.processing.template_padding,
        'test_dilate': test_settings.processing.dilate,
        'test_shape_tolerance': test_settings.processing.shape_tolerance,
        'test_overlap_threshold': test_settings.processing.overlap_threshold,
        'test_same_id_threshold': test_settings.processing.same_id_threshold_distance,
        'test_min_particle_area': test_settings.processing.min_particle_area,
        'test_max_particle_area': test_settings.processing.max_particle_area,
        'test_zero_stacks_offset': test_settings.processing.zero_stacks_offset,
        'test_background_subtraction': test_settings.processing.background_subtraction,
        'test_cropping_params': test_settings.processing.cropping_params,
        'test_processing_method': test_settings.processing.processing_method,
        'test_processing_filter_type': test_settings.processing.processing_filter_type,
        'test_processing_filter_size': test_settings.processing.processing_filter_size,
        'test_threshold_method': test_settings.processing.threshold_method,
        'test_threshold_modifier': test_settings.processing.threshold_modifier,
        'test_filter_params': test_settings.processing.processing_params,
        'test_threshold_params': test_settings.processing.threshold_params,
    }

    test_col_dict = {
        'test_col_number_of_particles': len(test_col.particle_ids),
        'test_col_true_number_of_particles': test_col.true_num_particles,
        'test_col_particle_ids': test_col.particle_ids,
        'test_col_number_of_images': test_col.num_images,
        'test_col_number_of_images_averaged_per_z': test_col.num_images_if_mean,
    }

    test_col_stats_dict = {
        'test_col_percent_particles_idd': test_col_stats['percent_particles_idd'],
        'test_col_mean_pixel_density': test_col_stats['mean_pixel_density'],
        'test_col_mean_particle_density': test_col_stats['mean_particle_density'],
        'test_col_mean_snr_filtered': test_col_stats['mean_snr_filtered'],
        'test_col_mean_signal': test_col_stats['mean_signal'],
        'test_col_mean_background': test_col_stats['mean_background'],
        'test_col_mean_std_background': test_col_stats['std_background'],
    }

    if test_settings.inputs.ground_truth_file_path is not None:
        test_col_global_meas_quality_dict = {
            'mean_error_x': test_col_global_meas_quality['error_x'],
            'mean_error_y': test_col_global_meas_quality['error_y'],
            'mean_error_z': test_col_global_meas_quality['error_z'],
            'mean_rmse_x': test_col_global_meas_quality['rmse_x'],
            'mean_rmse_y': test_col_global_meas_quality['rmse_y'],
            'mean_rmse_xy': test_col_global_meas_quality['rmse_xy'],
            'mean_rmse_z': test_col_global_meas_quality['rmse_z'],
            'mean_std_x': test_col_global_meas_quality['std_x'],
            'mean_std_y': test_col_global_meas_quality['std_y'],
            'mean_std_z': test_col_global_meas_quality['std_z'],
            'number_idd': test_col_global_meas_quality['num_idd'],
            'number_z_meas': test_col_global_meas_quality['num_valid_z_measure'],
            'percent_particles_measured': test_col_global_meas_quality['percent_measure'],
            'mean_cm': test_col_global_meas_quality['cm'],
            'max_sim': test_col_global_meas_quality['max_sim']
        }
    else:
        test_col_global_meas_quality_dict = {
            'mean_error_z': test_col_global_meas_quality['error_z'],
            'mean_rmse_z': test_col_global_meas_quality['rmse_z'],
            'mean_std_z': test_col_global_meas_quality['std_z'],
            'number_idd': test_col_global_meas_quality['num_idd'],
            'number_z_meas': test_col_global_meas_quality['num_valid_z_measure'],
            'percent_particles_measured': test_col_global_meas_quality['percent_measure'],
            'mean_cm': test_col_global_meas_quality['cm'],
            'max_sim': test_col_global_meas_quality['max_sim']
        }

    dicts = [optics_dict, test_col_global_meas_quality_dict, calib_settings_dict, calib_col_dict, calib_col_stats_dict,
             calib_stack_data_dict, test_settings_dict, test_col_dict, test_col_stats_dict]

    for d in dicts:
        export_data.update(d)

    export_df = pd.DataFrame.from_dict(data=export_data, orient='index')

    savedata = join(test_settings.outputs.results_path, 'gdpyt_{}_{}_{}.xlsx'.format(test_settings.outputs.save_id_string, calib_settings.outputs.save_id_string, export_data['date_and_time']))
    export_df.to_excel(savedata)

    return export_df

def export_key_results(calib_settings=None, calib_col=None, calib_stack_data=None, calib_col_stats=None,
                                test_settings=None, test_col=None, test_col_global_meas_quality=None, test_col_stats=None):

    # export data to text file
    export_data = {
        'date_and_time': datetime.now(),
    }

    optics_dict = {
        'field_of_view': calib_settings.optics.field_of_view,
        'depth_of_field': calib_settings.optics.depth_of_field,
        'pixels_per_particle_in_focus': calib_settings.optics.pixels_per_particle_in_focus,
        'img_bkg_mean': calib_settings.optics.bkg_mean,
        'img_bkg_noise': calib_settings.optics.bkg_noise,
    }

    calib_settings_dict = {
        'calib_image_path': calib_settings.inputs.image_path,
    }

    calib_col_dict = {
        'calib_col_number_of_particles': len(calib_col.particle_ids),
        'calib_col_true_number_of_particles': calib_col.true_num_particles,
        'calib_col_particle_ids': calib_col.particle_ids,
        'calib_col_measurement_depth': calib_col.measurement_depth,
    }

    calib_col_stats_dict = {
        'calib_col_mean_snr_filtered': calib_col_stats['mean_snr_filtered'],
        'calib_col_mean_signal': calib_col_stats['mean_signal'],
        'calib_col_mean_background': calib_col_stats['mean_background'],
        'calib_col_mean_noise': calib_col_stats['std_background'],
        'calib_col_mean_particle_density': calib_col_stats['mean_particle_density'],
        'calib_col_mean_pixel_density': calib_col_stats['mean_pixel_density'],
        'calib_col_percent_particles_idd': calib_col_stats['percent_particles_idd'],
        'calib_col_true_num_particles': calib_col_stats['true_num_particles'],
    }

    calib_stack_data_dict = {
        'calib_stack_particle_ids': calib_stack_data.particle_id.unique(),
        'calib_stack_mean_number_of_layers': calib_stack_data.layers.mean(),
        'calib_stack_mean_avg_snr': calib_stack_data.avg_snr.mean(),
    }

    test_settings_dict = {
        'test_image_path': test_settings.inputs.image_path,
    }

    test_col_dict = {
        'test_col_number_of_particles': len(test_col.particle_ids),
        'test_col_true_number_of_particles': test_col.true_num_particles,
        'test_col_particle_ids': test_col.particle_ids,
    }

    test_col_stats_dict = {
        'test_col_percent_particles_idd': test_col_stats['percent_particles_idd'],
        'test_col_mean_pixel_density': test_col_stats['mean_pixel_density'],
        'test_col_mean_particle_density': test_col_stats['mean_particle_density'],
        'test_col_mean_snr_filtered': test_col_stats['mean_snr_filtered'],
        'test_col_mean_signal': test_col_stats['mean_signal'],
        'test_col_mean_background': test_col_stats['mean_background'],
        'test_col_mean_std_background': test_col_stats['std_background'],
    }

    test_col_global_meas_quality_dict = {
        'mean_rmse_z': test_col_global_meas_quality['rmse_z'],
        'number_idd': test_col_global_meas_quality['num_idd'],
        'number_z_meas': test_col_global_meas_quality['num_valid_z_measure'],
        'percent_particles_measured': test_col_global_meas_quality['percent_measure'],
        'mean_cm': test_col_global_meas_quality['cm'],
        'max_sim': test_col_global_meas_quality['max_sim']
    }

    dicts = [optics_dict, test_col_global_meas_quality_dict, test_col_stats_dict, calib_col_stats_dict, test_col_dict,
             calib_stack_data_dict, calib_col_dict, test_settings_dict, calib_settings_dict]

    for d in dicts:
        export_data.update(d)

    export_df = pd.DataFrame.from_dict(data=export_data, orient='index')

    savedata = join(test_settings.outputs.results_path, 'gdpyt_key_results_{}_{}_{}.xlsx'.format(test_settings.outputs.save_id_string, calib_settings.outputs.save_id_string, export_data['date_and_time']))
    export_df.to_excel(savedata)

    return export_df

def export_local_meas_quality(calib_settings, test_settings, test_col_local_meas_quality):
    """
    Export local measurement quality to Excel
    """
    savedata = join(test_settings.outputs.results_path, 'gdpyt_local_data_{}_{}_{}.xlsx'.format(test_settings.outputs.save_id_string, calib_settings.outputs.save_id_string, datetime.now()))
    test_col_local_meas_quality.to_excel(savedata)

def export_particle_coords(collection, calib_settings, test_settings):
    """
    Export particle coordinates to Excel
    """
    if collection.image_collection_type == 'calibration':
        df = collection.get_particles_in_collection_coords(true_xy=calib_settings.inputs.ground_truth_file_path)

        savedata = join(calib_settings.outputs.results_path,
                        'gdpyt_calib_coords_{}_{}.xlsx'.format(calib_settings.outputs.save_id_string, datetime.now()))

    elif collection.image_collection_type == 'test':
        df = collection.get_particles_in_collection_coords(true_xy=test_settings.inputs.ground_truth_file_path)
        df['error'] = df['z_true'] - df['z']
        df.sort_values(by='error')

        savedata = join(test_settings.outputs.results_path,
                        'gdpyt_test_coords_{}_{}_{}.xlsx'.format(test_settings.outputs.save_id_string,
                                                            calib_settings.outputs.save_id_string, datetime.now()))

    df.to_excel(savedata, index=False)

    return df