# GdpytCharacterization
"""
Test harness for GDPyT characterization
"""

# imports
import os

from gdpyt import GdpytImageCollection
from gdpyt.utils import plotting
from gdpyt.utils import get
from os.path import join
from datetime import datetime
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.morphology import disk
from skimage.io import imsave


def test(calib_settings, test_settings=None, calib_col=None, calib_set=None, return_variables=None):
    # initial image analysis # TODO: write initial image analysis script
    """
    There should be an initial image analysis function that calculates values like:
        background mean
        background std

    These would be very helpful in deciding the processing filters and thresholding values.
    """

    if calib_col is None:
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
                                         hard_baseline=calib_settings.inputs.hard_baseline,
                                         static_templates=calib_settings.inputs.static_templates,
                                         overlapping_particles=calib_settings.inputs.overlapping_particles,
                                         single_particle_calibration=calib_settings.inputs.single_particle_calibration,
                                         optics_setup=calib_settings.optics,
                                         )
        # calculate particle similarity per image
        compute_p2p_similarity = False
        if compute_p2p_similarity:
            df_img_average_sim, df_collection_particle_sims = calib_col.calculate_image_particle_similarity()
            save_avg_sims = join(calib_settings.outputs.results_path,
                                 'average_similarity_{}.xlsx'.format(calib_settings.outputs.save_id_string))
            df_img_average_sim.to_excel(save_avg_sims, index=True)
            num_exports = int(np.ceil(len(df_collection_particle_sims) / 1e6))
            if num_exports > 1:
                for i in range(num_exports + 1):
                    row_i = int(i * 1e6)
                    row_f = int((i + 1) * 1e6)
                    export_ptp_similarities = df_collection_particle_sims.iloc[row_i:row_f]
                    export_ptp_similarities.to_excel(join(calib_settings.outputs.results_path,
                                                          'collection_similarities_sheet{}_{}.xlsx'.format(
                                                              i,
                                                              calib_settings.outputs.save_id_string)),
                                                     index=True)
            else:
                save_col_sims = join(calib_settings.outputs.results_path,
                                     'collection_similarities_{}.xlsx'.format(calib_settings.outputs.save_id_string))
                df_collection_particle_sims.to_excel(save_col_sims, index=True)
            compute_p2p_similarity = False

        # plot the baseline image with particle ID's
        if calib_col.baseline is not None:
            fig = calib_col.plot_baseline_image_and_particle_ids()
            plt.show()
            # pass

        # method for converting filenames to z-coordinates
        name_to_z = {}
        for image in calib_col.images.values():

            # if standard gdpt calibration dataset
            if calib_settings.inputs.ground_truth_file_path == 'standard_gdpt':
                N_CAL = float(len(calib_col.images))
                z_val = (float(image.filename.split(calib_settings.inputs.image_base_string)[-1].split('.tif')[0]) *
                         (calib_col.measurement_range / N_CAL)) - (calib_col._calibration_stack_z_step / 2) + \
                        calib_settings.processing.zero_stacks_offset
                name_to_z.update({image.filename: z_val})
            else:
                name_to_z.update({image.filename: float(
                    image.filename.split(calib_settings.inputs.image_base_string)[-1].split('.tif')[0])})  #  * calib_col._calibration_stack_z_step})

    if calib_set is None:
        plot_calibration_set = True

        # create the calibration set consisting of calibration stacks for each particle
        calib_set = calib_col.create_calibration(name_to_z=name_to_z,
                                                 template_padding=calib_settings.processing.template_padding,
                                                 min_num_layers=calib_settings.processing.min_layers_per_stack * len(
                                                     calib_col.images),
                                                 self_similarity_method='sknccorr',
                                                 dilate=calib_settings.processing.dilate)

        # plot the baseline image with particle ID's
        if calib_col.baseline is not None:
            # fig = calib_col.plot_baseline_image_and_particle_ids()
            # plt.show()
            pass

        # calculate particle similarity per image
        compute_p2p_similarity = False
        if compute_p2p_similarity:
            df_img_average_sim, df_collection_particle_sims = calib_col.calculate_image_particle_similarity()
            save_avg_sims = join(calib_settings.outputs.results_path, 'average_similarity_{}.xlsx'.format(calib_settings.outputs.save_id_string))
            df_img_average_sim.to_excel(save_avg_sims, index=True)
            num_exports = int(np.ceil(len(df_collection_particle_sims) / 1e6))
            if num_exports > 1:
                for i in range(num_exports + 1):
                    row_i = int(i * 1e6)
                    row_f = int((i + 1) * 1e6)
                    export_ptp_similarities = df_collection_particle_sims.iloc[row_i:row_f]
                    export_ptp_similarities.to_excel(join(calib_settings.outputs.results_path,
                                                           'collection_similarities_sheet{}_{}.xlsx'.format(
                                                               i,
                                                               calib_settings.outputs.save_id_string)),
                                                      index=True)
            else:
                save_col_sims = join(calib_settings.outputs.results_path,
                                     'collection_similarities_{}.xlsx'.format(calib_settings.outputs.save_id_string))
                df_collection_particle_sims.to_excel(save_col_sims, index=True)

    else:
        plot_calibration_set = False

    # export calibration self-similarity stack
    compute_ss_by_dzc = False
    if compute_ss_by_dzc:
        dzcs = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        center_on_zf = 50

        for dzc in dzcs:
            mdf, fdf = calib_set.calculate_all_stacks_self_similarity(dzc=dzc, center_on_zf=center_on_zf)
            mdf.to_excel(join(calib_settings.outputs.results_path,
                              'calib_stacks_middle_self-similarity_{}_{}.xlsx'.format(
                                  calib_settings.outputs.save_id_string, dzc)),
                         index=False)
            fdf.to_excel(join(calib_settings.outputs.results_path,
                              'calib_stacks_forward_self-similarity_{}_{}.xlsx'.format(
                                  calib_settings.outputs.save_id_string, dzc)),
                         index=False)

    # get calibration correction data
    if plot_calibration_set:

        if calib_settings.inputs.single_particle_calibration is False:
            # NOTE: this works when minimum particle size is 8 but fails when it is 5.
            calc_gauss = True
            if calc_gauss:
                calib_col.calculate_particle_to_particle_spacing(max_n_neighbors=3,
                                                                 theoretical_diameter_params=None,
                                                                 )
                calib_col.structure_spct_stats(idpt=False)
                calib_col.calculate_idpt_stats_gaussian(param_zf='gauss_A',
                                                        filter_percent_frames=calib_settings.processing.min_layers_per_stack)
                calib_col.correct_plane_tilt(zf_from='nsv')
                calib_col.update_particles_in_images()

                spath = calib_settings.outputs.results_path
                sid = calib_settings.outputs.save_id_string

                calib_col.spct_stats.to_excel(join(spath, 'calib_idpt_stats_{}.xlsx'.format(sid)), index=False)
                calib_col.spct_particle_defocus_stats.to_excel(
                    join(spath, 'calib_idpt_pid_defocus_stats_{}.xlsx'.format(sid)), index=False)
                calib_col.spct_population_defocus_stats.to_excel(
                    join(spath, 'calib_idpt_pop_defocus_stats_{}.xlsx'.format(sid)))
                calib_col.fitted_plane_equation.to_excel(
                    join(spath, 'calib_idpt_fitted_plane_params_{}.xlsx'.format(sid)))

                """dfc, dfz = calib_col.correct_calibration()
                savecoords = join(calib_settings.outputs.results_path,
                                  'calib_correction_coords_{}.xlsx'.format(calib_settings.outputs.save_id_string))
                dfc.to_excel(savecoords, index=False)
                savefocus = join(calib_settings.outputs.results_path,
                                 'calib_in-focus_coords_{}.xlsx'.format(calib_settings.outputs.save_id_string))
                dfz.to_excel(savefocus, index=False)"""

        elif calib_settings.inputs.single_particle_calibration is True:
            calib_col.calculate_particle_to_particle_spacing(max_n_neighbors=3, theoretical_diameter_params=None)
            calib_col.structure_spct_stats()
            calib_col.calculate_spct_stats(param_zf='zf_nsv',
                                           filter_percent_frames=calib_settings.processing.min_layers_per_stack)
            calib_col.correct_plane_tilt(zf_from='nsv')
            calib_col.update_particles_in_images()

            spath = calib_settings.outputs.results_path
            sid = calib_settings.outputs.save_id_string

            calib_col.spct_stats.to_excel(join(spath, 'calib_spct_stats_{}.xlsx'.format(sid)), index=False)
            calib_col.spct_particle_defocus_stats.to_excel(
                join(spath, 'calib_spct_pid_defocus_stats_{}.xlsx'.format(sid)), index=False)
            calib_col.spct_population_defocus_stats.to_excel(
                join(spath, 'calib_spct_pop_defocus_stats_{}.xlsx'.format(sid)))
            calib_col.fitted_plane_equation.to_excel(
                join(spath, 'calib_spct_fitted_plane_params_{}.xlsx'.format(sid)))

            dfc, dfz = calib_col.correct_calibration()
            savecoords = join(calib_settings.outputs.results_path,
                              'calib_correction_coords_{}.xlsx'.format(calib_settings.outputs.save_id_string))
            dfc.to_excel(savecoords, index=False)
            savefocus = join(calib_settings.outputs.results_path,
                             'calib_in-focus_coords_{}.xlsx'.format(calib_settings.outputs.save_id_string))
            dfz.to_excel(savefocus, index=False)

    # get calibration collection image stats
    calib_col_image_stats = calib_col.calculate_calibration_image_stats()

    # get calibration collection mean stats
    calib_col_stats = calib_col.calculate_image_stats()

    # get calibration stacks data
    calib_stack_data = calib_set.all_stacks_stats  # calculate_stacks_stats()

    # export calibration images data
    if plot_calibration_set:
        # export calibration self-similarity stack
        mdf, fdf = calib_set.calculate_all_stacks_self_similarity()
        mdf.to_excel(join(calib_settings.outputs.results_path,
                          'calib_stacks_middle_self-similarity_{}.xlsx'.format(calib_settings.outputs.save_id_string)),
                     index=False)
        fdf.to_excel(join(calib_settings.outputs.results_path,
                          'calib_stacks_forward_self-similarity_{}.xlsx'.format(calib_settings.outputs.save_id_string)),
                     index=False)

        # export calibration settings
        export_results_and_settings('calibration', calib_settings, calib_col, calib_stack_data, calib_col_stats)

        export_calib_stats(calib_settings, calib_col_image_stats, calib_stack_data)
        plot_calibration(calib_settings, test_settings, calib_col, calib_set, calib_col_image_stats,
                         calib_col_stats,
                         calib_stack_data)

    # ---

    # ---

    if return_variables == 'calibration':
        return calib_col, calib_set

    # for analyses of the standard GDPT dataset, there must not be a baseline.
    if test_settings.inputs.hard_baseline is False and calib_settings.inputs.single_particle_calibration is False:
        """ we allow movement of the particles in the test images and try to match them to calibration stacks """
        test_collection_baseline = None
        test_particle_id_image = test_settings.inputs.baseline_image
        use_stack_for_inference = None
    elif calib_settings.inputs.ground_truth_file_path is None and calib_settings.inputs.single_particle_calibration is False:
        """ any analysis where the particle distribution in the test collection matches the calibration collection """
        test_collection_baseline = calib_set
        test_particle_id_image = test_settings.inputs.baseline_image
        use_stack_for_inference = None
    elif calib_settings.inputs.ground_truth_file_path is not None and calib_settings.inputs.single_particle_calibration is False:
        """ synthetic particle dataset with identical calibration and test particle distributions """
        test_collection_baseline = calib_set
        test_particle_id_image = test_settings.inputs.baseline_image
        use_stack_for_inference = None
    elif calib_settings.inputs.ground_truth_file_path == 'standard_gdpt':
        """ standard GDPT dataset with a single calibration image """
        assert calib_settings.inputs.single_particle_calibration is True
        test_collection_baseline = test_settings.inputs.baseline_image  # test_collection_baseline = None
        test_particle_id_image = None
        use_stack_for_inference = 0
    elif calib_settings.inputs.single_particle_calibration is True and test_settings.inputs.static_templates is True:
        """ SPC matching with static - any analysis where the particle distribution matches calibration distribution"""
        test_collection_baseline = test_settings.inputs.baseline_image
        test_particle_id_image = None
        use_stack_for_inference = test_settings.z_assessment.use_stack_id
    elif calib_settings.inputs.single_particle_calibration is True:
        """ random synthetic particle dataset with a single calibration image """
        test_collection_baseline = test_settings.inputs.baseline_image
        test_particle_id_image = test_settings.inputs.baseline_image
        use_stack_for_inference = test_settings.z_assessment.use_stack_id
    else:
        raise ValueError("Unknown test situation.")

    # correct variables for meta-assessment
    if test_settings.inputs.image_collection_type == 'meta-test':
        meta_characterization = True
        test_settings.inputs.image_collection_type = 'test'
    else:
        meta_characterization = False

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
                                    hard_baseline=test_settings.inputs.hard_baseline,
                                    static_templates=test_settings.inputs.static_templates,
                                    particle_id_image=test_particle_id_image,
                                    overlapping_particles=test_settings.inputs.overlapping_particles,
                                    xydisplacement=test_settings.processing.xy_displacement,
                                    optics_setup=calib_settings.optics,
                                    )

    # plot the baseline image with particle ID's
    if test_col.baseline is not None:
        fig = test_col.plot_baseline_image_and_particle_ids()
        # plt.show()

    # if performing characterization on synthetic dataset, set calib_set offset according to z @ minimum particle area.
    if test_settings.inputs.ground_truth_file_path is not None:
        pass
    elif test_settings.inputs.known_z is not None:
        name_to_z_test = {}
        for image in test_col.images.values():
            name_to_z_test.update({image.filename: test_settings.inputs.known_z})
        test_col.set_true_z(image_to_z=name_to_z_test)
    elif meta_characterization:
        name_to_z = {}
        for image in test_col.images.values():
            name_to_z.update({image.filename: float(
                image.filename.split(calib_settings.inputs.image_base_string)[-1].split('.tif')[0]) *
                                              calib_col._calibration_stack_z_step})
        test_col.set_true_z(image_to_z=name_to_z)
    else:
        name_to_z_test = {}
        for image in test_col.images.values():
            z_filename = float(image.filename.split(test_settings.inputs.image_base_string)[-1].split('.tif')[0])
            # z_convert = (z_filename - z_filename % 3) / 3 * 5 + 5
            # z_convert_shift_to_focus = z_convert - 68.1
            # z_convert_shift_to_calib_focus = np.round(z_convert_shift_to_focus + 49.6, 2)
            name_to_z_test.update({image.filename: z_filename})
        test_col.set_true_z(image_to_z=name_to_z_test)

    # Infer the z-height of each particle
    test_col.infer_z(calib_set, infer_sub_image=test_settings.z_assessment.sub_image_interpolation).sknccorr(
        min_cm=test_settings.z_assessment.min_cm, use_stack=use_stack_for_inference)

    # export the particle coordinates
    test_coords = export_particle_coords(test_col, calib_settings, test_settings)

    # export particle image stats
    dft = test_col.package_test_particle_image_stats(true_xy=test_settings.inputs.ground_truth_file_path)
    dft.to_excel(join(test_settings.outputs.results_path, 'test_coords_stats.xlsx'), index=False)

    # get test collection stats
    test_col_stats = test_col.calculate_image_stats()

    # get test collection inference local uncertainties
    # test_col_local_meas_quality = test_col.calculate_measurement_quality_local(num_bins=20, min_cm=0.5,
    #                                                                           true_xy=test_settings.inputs.ground_truth_file_path)
    test_col_local_meas_quality = None

    # export local measurement quality
    # export_local_meas_quality(calib_settings, test_settings, test_col_local_meas_quality)

    # get test collection inference global uncertainties
    # test_col_global_meas_quality = test_col.calculate_measurement_quality_global(local=test_col_local_meas_quality)
    test_col_global_meas_quality = None

    # if plot calib was delayed but still desired
    # get calibration correction data
    calib_data = False
    if calib_data:
        if plot_calibration_set:

            if calib_settings.inputs.single_particle_calibration is False:
                # NOTE: this works when minimum particle size is 8 but fails when it is 5.
                calc_gauss = True
                if calc_gauss:
                    calib_col.calculate_particle_to_particle_spacing(max_n_neighbors=3,
                                                                     theoretical_diameter_params=None,
                                                                     )
                    calib_col.structure_spct_stats(idpt=False)
                    calib_col.calculate_idpt_stats_gaussian(param_zf='gauss_A',
                                                            filter_percent_frames=calib_settings.processing.min_layers_per_stack)
                    calib_col.correct_plane_tilt(zf_from='nsv')
                    calib_col.update_particles_in_images()

                    spath = calib_settings.outputs.results_path
                    sid = calib_settings.outputs.save_id_string

                    calib_col.spct_stats.to_excel(join(spath, 'calib_idpt_stats_{}.xlsx'.format(sid)), index=False)
                    calib_col.spct_particle_defocus_stats.to_excel(
                        join(spath, 'calib_idpt_pid_defocus_stats_{}.xlsx'.format(sid)), index=False)
                    calib_col.spct_population_defocus_stats.to_excel(
                        join(spath, 'calib_idpt_pop_defocus_stats_{}.xlsx'.format(sid)))
                    calib_col.fitted_plane_equation.to_excel(
                        join(spath, 'calib_idpt_fitted_plane_params_{}.xlsx'.format(sid)))

                    """dfc, dfz = calib_col.correct_calibration()
                    savecoords = join(calib_settings.outputs.results_path,
                                      'calib_correction_coords_{}.xlsx'.format(calib_settings.outputs.save_id_string))
                    dfc.to_excel(savecoords, index=False)
                    savefocus = join(calib_settings.outputs.results_path,
                                     'calib_in-focus_coords_{}.xlsx'.format(calib_settings.outputs.save_id_string))
                    dfz.to_excel(savefocus, index=False)"""

            elif calib_settings.inputs.single_particle_calibration is True:
                calib_col.calculate_particle_to_particle_spacing(max_n_neighbors=3, theoretical_diameter_params=None)
                calib_col.structure_spct_stats()
                calib_col.calculate_spct_stats(param_zf='zf_nsv',
                                               filter_percent_frames=calib_settings.processing.min_layers_per_stack)
                calib_col.correct_plane_tilt(zf_from='nsv')
                calib_col.update_particles_in_images()

                spath = calib_settings.outputs.results_path
                sid = calib_settings.outputs.save_id_string

                calib_col.spct_stats.to_excel(join(spath, 'calib_spct_stats_{}.xlsx'.format(sid)), index=False)
                calib_col.spct_particle_defocus_stats.to_excel(
                    join(spath, 'calib_spct_pid_defocus_stats_{}.xlsx'.format(sid)), index=False)
                calib_col.spct_population_defocus_stats.to_excel(
                    join(spath, 'calib_spct_pop_defocus_stats_{}.xlsx'.format(sid)))
                calib_col.fitted_plane_equation.to_excel(
                    join(spath, 'calib_spct_fitted_plane_params_{}.xlsx'.format(sid)))

                dfc, dfz = calib_col.correct_calibration()
                savecoords = join(calib_settings.outputs.results_path,
                                  'calib_correction_coords_{}.xlsx'.format(calib_settings.outputs.save_id_string))
                dfc.to_excel(savecoords, index=False)
                savefocus = join(calib_settings.outputs.results_path,
                                 'calib_in-focus_coords_{}.xlsx'.format(calib_settings.outputs.save_id_string))
                dfz.to_excel(savefocus, index=False)

        # get calibration collection image stats
        calib_col_image_stats = calib_col.calculate_calibration_image_stats()

        # get calibration collection mean stats
        calib_col_stats = calib_col.calculate_image_stats()

        # get calibration stacks data
        calib_stack_data = calib_set.all_stacks_stats  # calculate_stacks_stats()

        # export calibration self-similarity stack
        mdf, fdf = calib_set.calculate_all_stacks_self_similarity()
        mdf.to_excel(join(calib_settings.outputs.results_path,
                          'calib_stacks_middle_self-similarity_{}.xlsx'.format(calib_settings.outputs.save_id_string)),
                     index=False)
        fdf.to_excel(join(calib_settings.outputs.results_path,
                          'calib_stacks_forward_self-similarity_{}.xlsx'.format(calib_settings.outputs.save_id_string)),
                     index=False)

        # export calibration images data
        if plot_calibration_set:
            export_calib_stats(calib_settings, calib_col_image_stats, calib_stack_data)
            plot_calibration(calib_settings, test_settings, calib_col, calib_set, calib_col_image_stats,
                             calib_col_stats,
                             calib_stack_data)

    # ---

    # export data to excel
    export_results_and_settings('test', calib_settings, calib_col, calib_stack_data, calib_col_stats, test_settings,
                                test_col, test_col_global_meas_quality, test_col_stats)

    # export key results to excel (redundant data as full excel export but useful for quickly ascertaining results)
    export_key_results(calib_settings, calib_col, calib_stack_data, calib_col_stats, test_settings, test_col, test_col_global_meas_quality, test_col_stats)

    # plot
    plot_test(test_settings, test_col, test_col_stats, test_col_local_meas_quality, test_col_global_meas_quality, calib_set)

    # ---

    # export particle similarity curves
    export_similarity_curves = False
    if export_similarity_curves:
        pid_similarity_curves = test_col.package_particle_similarity_curves()
        num_exports = int(np.ceil(len(pid_similarity_curves) / 1e6))
        if num_exports > 1:
            for i in range(num_exports + 1):
                row_i = int(i * 1e6)
                row_f = int((i + 1) * 1e6)
                export_similarity_curves = pid_similarity_curves.iloc[row_i:row_f]
                export_similarity_curves.to_excel(join(test_settings.outputs.results_path,
                                                       'particle_similarity_curves_sheet{}_t{}_c{}.xlsx'.format(
                                                           i,
                                                           calib_settings.outputs.save_id_string,
                                                           test_settings.outputs.save_id_string)),
                                                  index=False)
        else:
            pid_similarity_curves.to_excel(join(test_settings.outputs.results_path,
                                                'particle_similarity_curves_t{}_c{}.xlsx'.format(
                                                    calib_settings.outputs.save_id_string,
                                                    test_settings.outputs.save_id_string)),
                                           index=False)

    # ---

    # assess every calib stack and particle ID
    assess_every_stack = False
    if assess_every_stack:
        assess_every_particle_and_stack_id(test_settings, calib_set, test_col)

    # all outputs
    return_variables = False
    if return_variables is False:
        del calib_col, calib_set, calib_col_image_stats, calib_stack_data, test_col, test_col_stats, \
            test_col_local_meas_quality, test_col_global_meas_quality
    else:
        return calib_col, calib_set, calib_col_image_stats, calib_stack_data, test_col, test_col_stats, \
               test_col_local_meas_quality, test_col_global_meas_quality

# ---

def plot_test(settings, test_col, test_col_stats, test_col_local_meas_quality, test_col_global_meas_quality, calib_set):

    if settings.outputs.save_plots or settings.outputs.show_plots:

        TEST_RESULTS_PATH = join(settings.outputs.results_path, 'figs')
        if not os.path.exists(TEST_RESULTS_PATH):
            os.makedirs(TEST_RESULTS_PATH)

        # ---

        # Scatter plot: frame vs. z-coord

        # calib
        arr_cal = calib_set.stack_locations

        # test
        df = test_col.get_particles_in_collection_coords()
        df_baseline = df[df['frame'] == df['frame'].min()]

        # plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(arr_cal[:, 1], arr_cal[:, 2], s=4, color='b', label='c')
        ax.scatter(df_baseline.x, df_baseline.y, s=1, color='r', label='t')
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        ax.legend()
        ax.set_title('[[X, Y]] does not shift actual positions. Only for linking purposes', fontsize=8)
        plt.tight_layout()
        savefigpath = join(settings.outputs.results_path, 'figs',
                           settings.outputs.save_id_string + '_baseline_test_on_calib.png')
        plt.savefig(fname=savefigpath, bbox_inches='tight')

        """
        Save particle images for specific particles
        """
        save_particle_images = True
        save_particle_arrays = True
        num_particles_to_save_images = 5

        # print("Test collection particle ID's: {}".format(test_col.particle_ids))

        # define particles of interest
        # radial_disp_pids = [11, 19, 38, 41, 50, 52, 53, 55, 62, 66, 72, 89, 94]  # for IDPT
        # radial_disp_pids = [17, 25, 33, 37, 39, 45, 47, 49, 46, 55, 50, 48]  # for SPCT

        if save_particle_images or save_particle_arrays:

            # select "good" particles
            # plot_calib_stack_particle_ids = test_col.particle_ids[:num_particles_to_save_images]  # radial_disp_pids
            plot_calib_stack_particle_ids = [pid for pid in random.sample(set(test_col.particle_ids), num_particles_to_save_images)]

            print("Save images for particle ID's: {}".format(plot_calib_stack_particle_ids))

            # image folder
            dir_png = os.path.join(settings.outputs.results_path, 'test_pngs')
            if not os.path.exists(dir_png):
                os.makedirs(dir_png)

            # tiff array folder
            dir_tif = os.path.join(settings.outputs.results_path, 'test_tiffs')
            if not os.path.exists(dir_tif):
                os.makedirs(dir_tif)

            for id in plot_calib_stack_particle_ids:

                # save ID for this set of plots
                save_calib_pid = settings.outputs.save_id_string + '_pid{}'.format(id)

                # save every (particle template, z) as a separate file
                for img in test_col.images.values():
                    for p in img.particles:
                        if p.id == id:

                            if save_particle_arrays:
                                savearrpath = join(dir_tif,
                                                   'template_pid{}_z{}_ztrue{}.tiff'.format(id,
                                                                                            np.round(p.z, 2),
                                                                                            np.round(p.z_true, 2)))
                                imsave(savearrpath, p.template, check_contrast=False)

                            if save_particle_images:
                                fig, ax = plt.subplots()
                                ax.imshow(p.template, cmap='viridis')
                                ax.axis('off')
                                savefigpath = join(dir_png,
                                                   'template_pid{}_z{}_ztrue{}.png'.format(id,
                                                                                           np.round(p.z, 2),
                                                                                           np.round(p.z_true, 2)))
                                plt.savefig(fname=savefigpath, dpi=100, bbox_inches='tight', pad_inches=0.0)
                                plt.close()

        # ---

        # Scatter plot: frame vs. z-coord
        df = test_col.get_particles_in_collection_coords()
        fig, ax = plt.subplots(figsize=(12, 8))
        for num_pid_id, pid_id in enumerate(df.id.unique()):
            dfpid_id = df[df['id'] == pid_id]
            ax.plot(dfpid_id.frame, dfpid_id.z, '-o', ms=5, label=dfpid_id)
        ax.set_ylabel('z')
        ax.set_xlabel('frame')
        plt.tight_layout()
        savefigpath = join(settings.outputs.results_path, 'figs',
                           settings.outputs.save_id_string + '_line-plot_z_by_frame.png')
        plt.savefig(fname=savefigpath, bbox_inches='tight')
        # plt.show()

        # ---

        if test_col_local_meas_quality is not None:
            # plot NEW normalized local rmse uncertainty
            fig = test_col.plot_bin_local_rmse_z(measurement_quality=test_col_local_meas_quality,
                                                 measurement_depth=test_col.measurement_depth,
                                                 second_plot='cm')
            fig.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, 'figs',
                                   settings.outputs.save_id_string + '_new_rmse_depth_uncertainty.png')
                fig.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                fig.show()

            # plot normalized local rmse uncertainty
            """fig = test_col.plot_local_rmse_uncertainty(measurement_quality=test_col_local_meas_quality,
                                                       measurement_depth=test_col.measurement_depth,
                                                       measurement_width=settings.optics.field_of_view)
            fig.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, 'figs',
                                   settings.outputs.save_id_string + '_rmse_depth_uncertainty.png')
                fig.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                fig.show()"""

            # plot normalized local rmse uncertainty and correlation value
            """fig = test_col.plot_local_rmse_uncertainty(measurement_quality=test_col_local_meas_quality,
                                                       measurement_depth=test_col.measurement_depth,
                                                       measurement_width=settings.optics.field_of_view,
                                                       second_plot='cm')
            fig.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, 'figs',
                                   settings.outputs.save_id_string + '_rmse_depth_uncertainty_and_cm.png')
                fig.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                fig.show()"""

            # plot normalized local rmse uncertainty and valid z-measurements
            """fig = test_col.plot_local_rmse_uncertainty(measurement_quality=test_col_local_meas_quality,
                                                       measurement_depth=test_col.measurement_depth,
                                                       measurement_width=settings.optics.field_of_view,
                                                       second_plot='num_valid_z_measure')
            fig.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, 'figs',
                                   settings.outputs.save_id_string + '_rmse_depth_uncertainty_and_valid_z.png')
                fig.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                fig.show()"""

            # plot local rmse uncertainty in real-coordinates
            """fig = test_col.plot_local_rmse_uncertainty(measurement_quality=test_col_local_meas_quality)
            fig.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, 'figs',
                                   settings.outputs.save_id_string + '_rmse_uncertainty.png')
                fig.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                fig.show()"""

            # plot local rmse uncertainty and correlation value
            """fig = test_col.plot_local_rmse_uncertainty(measurement_quality=test_col_local_meas_quality, second_plot='cm')
            fig.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, 'figs',
                                   settings.outputs.save_id_string + '_rmse_uncertainty_and_cm.png')
                fig.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                fig.show()"""

            # plot normalized local rmse uncertainty and valid z-measurements
            """fig = test_col.plot_local_rmse_uncertainty(measurement_quality=test_col_local_meas_quality,
                                                       second_plot='num_valid_z_measure')
            fig.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, 'figs',
                                   settings.outputs.save_id_string + '_rmse_uncertainty_and_valid_z.png')
                fig.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                fig.show()"""

        # plot particle image stats

        plot_particle_image_stats = False

        if plot_particle_image_stats:
            # plot particle area vs z/h for every particle
            fig = test_col.plot_particles_stats()
            plt.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, 'figs',
                                   settings.outputs.save_id_string + '_particles_area.png')
                plt.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                plt.show()

            # plot particle area vs z/h for every particle
            """fig = test_col.plot_particles_stats(stat='img_num_particles')
            plt.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, 'figs',
                                   settings.outputs.save_id_string + '_number_particles.png')
                plt.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                plt.show()"""

            # plot snr and area
            """test_col.plot_particle_snr_and(second_plot='area')
            plt.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, 'figs',
                                   settings.outputs.save_id_string + '_snr_area.png')
                plt.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                plt.show()"""

            # plot snr and percent of particles assigned valid z-coordinate
            """
            test_col.plot_particle_snr_and(second_plot='percent_measured')
            plt.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, 'figs',
                                   settings.outputs.save_id_string + '_snr_percent_measured.png')
                plt.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                plt.show()
            """

            # plot snr and Cm
            test_col.plot_particle_snr_and(second_plot='cm')
            plt.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, 'figs',
                                   settings.outputs.save_id_string + '_snr_cm_max_sim.png')
                plt.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                plt.show()

        # plot for a random selection of particles in the test collection
        if len(test_col.particle_ids) >= 10:
            P_INSPECTS = [int(p) for p in random.sample(set(test_col.particle_ids), 10)]
        else:
            P_INSPECTS = test_col.particle_ids

        for P_INSPECT in P_INSPECTS:  # radial_disp_pids:  # P_INSPECTS:

            P_INSPECT = int(P_INSPECT)

            """
            # plot particle stack with z and true_z as subplot titles
            fig = test_col.plot_single_particle_stack(particle_id=P_INSPECT)
            if fig is not None:
                fig.suptitle(settings.outputs.save_id_string + ': particle id {}'.format(P_INSPECT))
                plt.tight_layout()
                if settings.outputs.save_plots is True:
                    savefigpath = join(settings.outputs.results_path, 'figs',
                                       settings.outputs.save_id_string + '_particle_stack_pid{}.png'.format(P_INSPECT))
                    fig.savefig(fname=savefigpath)
                    plt.close()
                if settings.outputs.show_plots:
                    fig.show()
            """

            """
            # plot the mean peak intensity per particle:
            fig = test_col.plot_particle_peak_intensity(particle_id=P_INSPECT)
            if fig:
                plt.suptitle(settings.outputs.save_id_string)
                plt.tight_layout()
                if settings.outputs.save_plots is True:
                    savefigpath = join(settings.outputs.results_path, 'figs',
                                       settings.outputs.save_id_string + '_peak_intensity_pid{}.png'.format(P_INSPECT))
                    plt.savefig(fname=savefigpath, bbox_inches='tight')
                    plt.close()
                if settings.outputs.show_plots:
                    plt.show()
            """

            # plot interpolation curves for every particle

            """
            fig = test_col.plot_similarity_curve(sub_image=settings.z_assessment.sub_image_interpolation,
                                                 method=settings.z_assessment.infer_method,
                                                 min_cm=settings.z_assessment.min_cm,
                                                 particle_id=P_INSPECT, image_id=None)
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
            """

            """
            # plot particle template at z_guess and z_calib_nearest_true
            if settings.outputs.save_plots is True:
                sup_title = settings.outputs.save_id_string + ': particle id {}'.format(P_INSPECT)
                savefigpath = join(settings.outputs.results_path, 'figs',
                                   settings.outputs.save_id_string + '_z_error_pid{}.png'.format(P_INSPECT))
            error = 5
            particles_errors = get.particles_with_large_z_error(error, test_col, particle_id=P_INSPECT, image_id=None)
            plotting.plot_images_and_similarity_curve(calib_set, test_col, particle_id=P_INSPECT, image_id=None,
                                                      min_cm=0.5, sup_title=sup_title, save_path=savefigpath)
            """

        # plot test collection images with particles
        if len(test_col.files) >= 4:
            plot_test_col_imgs = [img_id for img_id in random.sample(set(test_col.files), 4)]
            fig, ax = plt.subplots(ncols=4, figsize=(12, 3))
            for i, img in enumerate(plot_test_col_imgs):
                ax[i].imshow(
                    test_col.images[img].draw_particles(raw=False, thickness=1, draw_id=True, draw_bbox=False))
                img_formatted = np.round(float(img.split(settings.inputs.image_base_string)[-1].split('.tif')[0]),
                                         3)
                ax[i].set_title('{}'.format(img_formatted))
                ax[i].axis('off')
            plt.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savepath = join(settings.outputs.results_path, 'figs',
                                settings.outputs.save_id_string + '_test_col.png')
                plt.savefig(savepath)
                plt.close()
            if settings.outputs.show_plots is True:
                plt.show()

        # plot every test image with identified particles: '_test_col{}.png'
        """
        plot_test_col_imgs = test_col.files[2:4]
        for i, img in enumerate(plot_test_col_imgs):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(test_col.images[img].draw_particles(raw=False, thickness=1, draw_id=False, draw_bbox=False))
            ax.set_title('{}'.format(img))
            plt.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savepath = join(settings.outputs.results_path, 'figs',
                                settings.outputs.save_id_string + '_test_col{}.png'.format(i))
                plt.savefig(savepath)
                plt.close()
            if settings.outputs.show_plots is True:
                plt.show()
        """

        # plot the mean peak intensity per particle:
        """
        fig = test_col.plot_particle_peak_intensity()
        plt.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savefigpath = join(settings.outputs.results_path, 'figs',
                               settings.outputs.save_id_string + '_peak_intensity.png')
            plt.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()
        if settings.outputs.show_plots:
            plt.show()
        """

        # plot interpolation curves for every particle on image_id N_CAL//2
        """
        for IMG_INSPECT in np.arange(start=0, stop=len(test_col.images), step=50):
            fig = test_col.plot_similarity_curve(sub_image=settings.z_assessment.sub_image_interpolation,
                                                 method=settings.z_assessment.infer_method,
                                                 min_cm=settings.z_assessment.min_cm,
                                                 particle_id=None, image_id=IMG_INSPECT)
            if fig is not None:
                fig.suptitle(
                    settings.outputs.save_id_string + ': ' + r'image #/$N_{cal}$' + ' = {}'.format(IMG_INSPECT))
                plt.tight_layout()
                if settings.outputs.save_plots is True:
                    savefigpath = join(settings.outputs.results_path, 'figs',
                                       settings.outputs.save_id_string + '_correlation_particles_in_img_{}.png'.format(
                                           IMG_INSPECT))
                    fig.savefig(fname=savefigpath, bbox_inches='tight')
                if settings.outputs.show_plots:
                    fig.show()
                plt.close(fig)
        """



def plot_calibration(settings, test_settings, calib_col, calib_set, calib_col_image_stats, calib_col_stats,
                     calib_stack_data):
    # show/save plots
    if settings.outputs.show_plots or settings.outputs.save_plots:

        CALIB_RESULTS_PATH = join(settings.outputs.results_path, 'figs')
        if not os.path.exists(CALIB_RESULTS_PATH):
            os.makedirs(CALIB_RESULTS_PATH)

        """
        per-collection plots
        """

        # plot the baseline image with particle ID's
        if calib_col.baseline is not None:
            fig = calib_col.plot_baseline_image_and_particle_ids()
            plt.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, 'figs',
                                   settings.outputs.save_id_string + '_calibration_baseline_image.png')
                plt.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                plt.show()

        # ---

        # plots that are only useful for SPCT
        if settings.inputs.single_particle_calibration is True:

            # plot number of particles identified in every image: '_num_particles_per_image.png'
            fig = calib_col.plot_num_particles_per_image()
            plt.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, 'figs',
                                   settings.outputs.save_id_string + '_num_particles_per_image.png')
                plt.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                plt.show()

            # plot particle area vs z or z/h for every particle: '_particles_area.png'
            fig = calib_col.plot_particles_stats()
            plt.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, 'figs',
                                   settings.outputs.save_id_string + '_particles_area.png')
                plt.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                plt.show()

            # plot snr and area: '_snr_area.png'
            calib_col.plot_particle_snr_and(second_plot='area')
            plt.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, 'figs',
                                   settings.outputs.save_id_string + '_snr_area.png')
                plt.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                plt.show()

            # plot the mean peak intensity per particle:
            fig = calib_col.plot_particle_peak_intensity()
            plt.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, 'figs',
                                   settings.outputs.save_id_string + '_peak_intensity.png')
                plt.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                plt.show()

            # plot snr and solidity: '_snr_solidity.png'
            """calib_col.plot_particle_snr_and(second_plot='solidity')
            plt.suptitle(settings.outputs.save_id_string)
            plt.tight_layout()
            if settings.outputs.save_plots is True:
                savefigpath = join(settings.outputs.results_path, 'figs',
                settings.outputs.save_id_string + '_snr_solidity.png')
                plt.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()
            if settings.outputs.show_plots:
                plt.show()"""

        # plot calibration images with identified particles: '_calib_col.png'
        plot_calib_col_imgs = [index for index in random.sample(set(calib_col.files), 4)]
        fig, ax = plt.subplots(ncols=4, figsize=(12, 6))
        for i, img in enumerate(plot_calib_col_imgs):
            ax[i].imshow(calib_col.images[img].draw_particles(raw=True, thickness=1, draw_id=True, draw_bbox=True))
            ax[i].set_title('z = {}'.format(np.round(calib_col.images[img].z, 2)))
            ax[i].axis('off')
        plt.suptitle(settings.outputs.save_id_string)
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savepath = join(settings.outputs.results_path, 'figs',
                            settings.outputs.save_id_string + '_multi_calib_col{}.png'.format(i))
            plt.savefig(savepath)
            plt.close()
        if settings.outputs.show_plots is True:
            plt.show()

        # plot every calibration image with identified particles: '_calib_col.png'
        plot_every_frame = False
        plot_every_z_frames = 5

        if plot_every_frame:
            plot_calib_col_imgs = calib_col.files
            for i, img in enumerate(plot_calib_col_imgs[::plot_every_z_frames]):
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'aspect': 'equal'}, frameon=False)
                ax.imshow(calib_col.images[img].draw_particles(raw=True, thickness=1,
                                                               draw_id=False, draw_bbox=True, draw_contour=False),
                          cmap='binary')

                ax.axis('off')
                # ax.set_title('{}'.format(img))
                # plt.suptitle(settings.outputs.save_id_string)
                # plt.tight_layout()
                if settings.outputs.save_plots is True:
                    savepath = join(settings.outputs.results_path, 'figs',
                                    settings.outputs.save_id_string + '_calib_col{}.png'.format(i))
                    plt.savefig(savepath, bbox_inches='tight')
                    plt.close()
                if settings.outputs.show_plots is True:
                    plt.show()

                # ---

                # plot with particle ID drawn
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'aspect': 'equal'}, frameon=False)
                ax.imshow(calib_col.images[img].draw_particles(raw=True, thickness=1,
                                                               draw_id=True, draw_bbox=True, draw_contour=False),
                          cmap='binary')

                ax.axis('off')
                if settings.outputs.save_plots is True:
                    savepath = join(settings.outputs.results_path, 'figs',
                                    settings.outputs.save_id_string + '_pIDs_calib_col{}.png'.format(i))
                    plt.savefig(savepath, bbox_inches='tight')
                    plt.close()
                if settings.outputs.show_plots is True:
                    plt.show()

                # ---

        # ---

        # plot calibration set's stack's self similarity: '_calibset_stacks_self_similarity_{}.png'
        """
        fig = calib_set.plot_stacks_self_similarity(min_num_layers=settings.processing.min_layers_per_stack,
                                                    save_string=settings.outputs.save_id_string)
        plt.suptitle(settings.outputs.save_id_string + ', infer: sknccorr')
        plt.tight_layout()
        if settings.outputs.save_plots is True:
            savepath = join(settings.outputs.results_path, 'figs',
            settings.outputs.save_id_string + '_calibset_stacks_self_similarity_sknccorr.png')
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

        """
        Save particle images for specific particles
        """
        save_particle_images = True
        save_particle_arrays = True
        num_particles_to_save_images = 5
        dz_calib_step_size = 5

        # define particles of interest
        radial_disp_pids = [44, 84, 39, 50, 46, 85]  # [5, 15, 22, 30, 34, 36, 45, 42, 43, 44, 46, 45, 52, 53]

        # choose particle ID's at random from calibration set
        if len(calib_set.best_stack_ids) < 5:
            plot_calib_stack_particle_ids = calib_set.best_stack_ids
        else:
            plot_calib_stack_particle_ids = [pid for pid in random.sample(set(calib_set.best_stack_ids), num_particles_to_save_images)]
        radial_disp_pids = plot_calib_stack_particle_ids

        if save_particle_images or save_particle_arrays:

            # select "good" particles
            # plot_calib_stack_particle_ids = calib_set.best_stack_ids[:num_particles_to_save_images]

            # make sure 'calib' particle is included
            if settings.inputs.use_stack_id not in plot_calib_stack_particle_ids:
                if isinstance(settings.inputs.use_stack_id, (int, float)):
                    plot_calib_stack_particle_ids = np.append(plot_calib_stack_particle_ids, settings.inputs.use_stack_id)

            # plot_calib_stack_particle_ids = radial_disp_pids  # [10, 11, 41, 53, 50, 52, 55, 69, 72, 89, 94]

            # get 'z' values that we want to plot
            plot_zs = calib_col_image_stats.z.to_numpy()
            plot_zs = np.arange(np.min(plot_zs), np.max(plot_zs) + 1, dz_calib_step_size)
            print(plot_zs)

            # Gaussian fit image folder
            """
            dir_gauss = os.path.join(settings.outputs.results_path, 'gaussians')
            if not os.path.exists(dir_gauss):
                os.makedirs(dir_gauss)

            # plot Gaussian fit on image
            if settings.inputs.single_particle_calibration is True:
                plotting.plot_gaussian_fit_on_image_for_particle(collection=calib_col,
                                                                 particle_ids=plot_calib_stack_particle_ids,
                                                                 frame_step_size=int(
                                                                     np.floor(len(calib_col.images.values()) / 11)),
                                                                 path_figs=dir_gauss)
            """

            # image folder
            dir_png = os.path.join(settings.outputs.results_path, 'pngs')
            if not os.path.exists(dir_png):
                os.makedirs(dir_png)

            # tiff array folder
            dir_tif = os.path.join(settings.outputs.results_path, 'tiffs')
            if not os.path.exists(dir_tif):
                os.makedirs(dir_tif)

            print(plot_calib_stack_particle_ids)

            for id in plot_calib_stack_particle_ids:

                # save ID for this set of plots
                save_calib_pid = settings.outputs.save_id_string + '_pid{}'.format(id)

                # save every (particle template, z) as a separate file
                for img in calib_col.images.values():
                    if img.z in plot_zs:
                        for p in img.particles:
                            if p.id == id:

                                if save_particle_arrays:
                                    savearrpath = join(dir_tif, 'template_pid{}_z_true{}.tiff'.format(id, p.z_true))
                                    imsave(savearrpath, p.template, check_contrast=False)

                                if save_particle_images:
                                    fig, ax = plt.subplots()
                                    ax.imshow(p.template, cmap='binary')
                                    ax.axis('off')
                                    savefigpath = join(dir_png, 'template_pid{}_z_true{}.png'.format(id, p.z_true))
                                    plt.savefig(savefigpath, dpi=300, bbox_inches='tight', pad_inches=0.0)
                                    plt.close()

            # ---

        # ---

        """
        Particles at random
        """
        plot_figs_for_single_pids = True
        radial_disp_pids = plot_calib_stack_particle_ids

        if plot_figs_for_single_pids:

            # plot_calib_stack_particle_ids = [40, 42, 44, 35, 0, 78]
            #if settings.inputs.use_stack_id not in plot_calib_stack_particle_ids:
                #if isinstance(settings.inputs.use_stack_id, int):
                #plot_calib_stack_particle_ids = np.append(plot_calib_stack_particle_ids, settings.inputs.use_stack_id)

            for id in radial_disp_pids:  # plot_calib_stack_particle_ids:

                if id is None:
                    continue

                # save ID for this set of plots
                save_calib_pid = settings.outputs.save_id_string + '_pid{}'.format(id)

                # --------------------------

                # plot the particle image + fitted Gaussian at 3 different z-heights
                """
                fig = calib_col.plot_particle_peak_intensity(particle_id=id)
                plt.suptitle(settings.outputs.save_id_string)
                plt.tight_layout()
                if settings.outputs.save_plots is True:
                    savefigpath = join(settings.outputs.results_path, 'figs',
                                       settings.outputs.save_id_string + '_peak_intensity_pid{}.png'.format(id))
                    plt.savefig(fname=savefigpath, bbox_inches='tight')
                    plt.close()
                if settings.outputs.show_plots:
                    plt.show()
                """

                # --------------------------

                # plot the mean peak intensity per particle:
                fig = calib_col.plot_particle_peak_intensity(particle_id=id)
                plt.suptitle(settings.outputs.save_id_string)
                plt.tight_layout()
                if settings.outputs.save_plots is True:
                    savefigpath = join(settings.outputs.results_path, 'figs',
                                       settings.outputs.save_id_string + '_peak_intensity_pid{}.png'.format(id))
                    plt.savefig(fname=savefigpath, bbox_inches='tight')
                if settings.outputs.show_plots:
                    plt.show()
                plt.close()

                # plot calibration stack and contour outline for a single particle: '_filtered_calib_stack.png'
                calib_set.calibration_stacks[id].plot_calib_stack(imgs_per_row=9, fig=None, ax=None, format_string=False)
                plt.suptitle(save_calib_pid)
                plt.tight_layout()
                if settings.outputs.save_plots is True:
                    savepath = join(settings.outputs.results_path, 'figs',
                                    save_calib_pid + '_filtered_calib_stack.png')
                    plt.savefig(savepath)
                if settings.outputs.show_plots is True:
                    plt.show()
                plt.close()

                # plot calibration stack and filled contour for a single particle: '_calib_stack_contours.png'
                """
                calib_set.calibration_stacks[id].plot_calib_stack(imgs_per_row=9, fill_contours=True, fig=None, ax=None,
                                                      format_string=False)
                plt.suptitle(save_calib_pid)
                plt.tight_layout()
                if settings.outputs.save_plots is True:
                    savepath = join(settings.outputs.results_path, 'figs',
                     save_calib_pid + '_calib_stack_contours.png')
                    plt.savefig(savepath)
                    plt.close()
                if settings.outputs.show_plots is True:
                    plt.show()
                """

                # plot calibration stack self similarity: '_calib_stack_self_similarity_{}.png'
                """fig = calib_set.calibration_stacks[id].plot_self_similarity()
                plt.suptitle(save_calib_pid + ', infer: sknccorr')
                plt.tight_layout()
                if settings.outputs.save_plots is True:
                    savepath = join(settings.outputs.results_path, 'figs',
                                    save_calib_pid + '_calib_stack_no_interp_self_similarity_sknccorr.png')
                    plt.savefig(savepath)
                    plt.close()
                if settings.outputs.show_plots is True:
                    plt.show()"""

                # plot particle SNR and area: '_snr_area.png'

                fig = calib_col.plot_particle_snr_and(second_plot='area', particle_id=id)
                plt.suptitle(save_calib_pid)
                plt.tight_layout()
                if settings.outputs.save_plots is True:
                    savefigpath = join(settings.outputs.results_path, 'figs', save_calib_pid + '_snr_area.png')
                    plt.savefig(fname=savefigpath, bbox_inches='tight')
                if settings.outputs.show_plots:
                    plt.show()
                plt.close()

                # plot theoretical and measured particle signal: '_signal_theory_and_measure.png'
                """
                fig = calib_col.plot_particle_signal(optics=settings.optics, collection_image_stats=calib_col_image_stats,
                                                     particle_id=id, intensity_max_or_mean='max')
                plt.suptitle(save_calib_pid)
                plt.tight_layout()
                if settings.outputs.save_plots is True:
                    savefigpath = join(settings.outputs.results_path, 'figs', save_calib_pid + '_signal_theory_and_measure.png')
                    plt.savefig(fname=savefigpath, bbox_inches='tight')
                    plt.close()
                if settings.outputs.show_plots:
                    plt.show()
                """

                # plot theoretical and measured particle diameter: '_diameter_theory_and_measure.png'
                """
                fig = calib_col.plot_particle_diameter(collection_image_stats=calib_col_image_stats, optics=settings.optics,
                                                       particle_id=id)
                plt.suptitle(save_calib_pid)
                plt.tight_layout()
                if settings.outputs.save_plots is True:
                    savefigpath = join(settings.outputs.results_path, 'figs',
                                       save_calib_pid + '_diameter_theory_and_measure.png')
                    plt.savefig(fname=savefigpath, bbox_inches='tight')
                if settings.outputs.show_plots:
                    plt.show()
                plt.close()
                """

                # plot 3D calibration stack for a single particle: '_calib_stack_3d.png'
                """if len(calib_col.images) > 80:
                    len(calib_col.images) // 10
                    stepsize = 10
                elif len(calib_col.images) > 30:
                    stepsize= 5
                else:
                    stepsize = 1
                calib_set.calibration_stacks[id].plot_3d_stack(intensity_percentile=(20, 99), stepsize=stepsize,
                                                               aspect_ratio=2.5)
                plt.suptitle(save_calib_pid)
                plt.tight_layout()
                if settings.outputs.save_plots is True:
                    savepath = join(settings.outputs.results_path, 'figs', save_calib_pid + '_calib_stack_3d.png')
                    plt.savefig(savepath, bbox_inches="tight")
                    plt.close()
                if settings.outputs.show_plots is True:
                    plt.show()"""

                # plot adjacent similarities with template images: '_calib_stack_index_{}_adjacent_similarity_{}.png'

                """indices = [ind for ind in random.sample(set(np.arange(2, len(calib_col.files)-3)), 4)]
                for indice in indices:
                    fig = calib_set.calibration_stacks[id].plot_adjacent_self_similarity(index=[indice])
                    plt.suptitle(save_calib_pid + ', infer: {sknccorr}')
                    plt.tight_layout()
                    if settings.outputs.save_plots is True:
                        savepath = join(settings.outputs.results_path, 'figs',
                                        save_calib_pid +
                                        '_calib_stack_index_{}_adjacent_similarity_{}.png'.format(indice, 'sknccorr'))
                        plt.savefig(savepath)
                        plt.close()
                    if settings.outputs.show_plots is True:
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
    test_col.plot_every_image_particle_stack_similarity(calib_set=calib_set, plot=True,
                                                        min_cm=settings.z_assessment.min_cm,
                                                        save_results_path=settings.outputs.results_path,
                                                        infer_sub_image=settings.z_assessment.sub_image_interpolation,
                                                        measurement_depth=test_col.measurement_depth)


def export_results_and_settings(export='test', calib_settings=None, calib_col=None, calib_stack_data=None,
                                calib_col_stats=None,
                                test_settings=None, test_col=None, test_col_global_meas_quality=None,
                                test_col_stats=None):
    # export data to text file
    export_data = {
        'date_and_time': datetime.now(),
    }

    if test_settings is not None:
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
    else:
        optics_dict = None

    calib_settings_dict = {
        'calib_image_path': calib_settings.inputs.image_path,
        'calib_base_string': calib_settings.inputs.image_base_string,
        'calib_ground_truth_image_path': calib_settings.inputs.ground_truth_file_path,
        'calib_baseline_image': calib_settings.inputs.baseline_image,
        'calib_hard_baseline': calib_settings.inputs.hard_baseline,
        'calib_static_templates': calib_settings.inputs.static_templates,
        'calib_ovrlp_prtcl_segmentation': calib_settings.inputs.overlapping_particles,
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
        'calib_stack_mean_max_particle_area': calib_stack_data.max_particle_area.mean(),
        'calib_stack_mean_avg_area': calib_stack_data.avg_area.mean(),
        'calib_stack_mean_avg_snr': calib_stack_data.avg_snr.mean(),

    }

    if export == 'calibration':

        dicts = [optics_dict, calib_settings_dict, calib_col_dict, calib_col_stats_dict, calib_stack_data_dict]

        for d in dicts:
            if d is not None:
                export_data.update(d)

        export_df = pd.DataFrame.from_dict(data=export_data, orient='index')

        savedata = join(calib_settings.outputs.results_path,
                        'gdpyt_calibration_{}_{}.xlsx'.format(calib_settings.outputs.save_id_string,
                                                              export_data['date_and_time']))
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
        'test_baseline_image': test_settings.inputs.baseline_image,
        'test_hard_baseline': test_settings.inputs.hard_baseline,
        'test_static_templates': test_settings.inputs.static_templates,
        'test_ovrlp_prtcl_segmentation': test_settings.inputs.overlapping_particles,
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

    if test_col_stats is not None:
        test_col_stats_dict = {
            'test_col_percent_particles_idd': test_col_stats['percent_particles_idd'],
            'test_col_mean_pixel_density': test_col_stats['mean_pixel_density'],
            'test_col_mean_particle_density': test_col_stats['mean_particle_density'],
            'test_col_mean_snr_filtered': test_col_stats['mean_snr_filtered'],
            'test_col_mean_signal': test_col_stats['mean_signal'],
            'test_col_mean_background': test_col_stats['mean_background'],
            'test_col_mean_std_background': test_col_stats['std_background'],
        }
    else:
        test_col_stats_dict = {}

    if test_settings.inputs.ground_truth_file_path is not None and test_col_global_meas_quality is not None:
        test_col_global_meas_quality_dict = {
            # 'mean_error_x': test_col_global_meas_quality['error_x'],
            # 'mean_error_y': test_col_global_meas_quality['error_y'],
            # 'mean_error_z': test_col_global_meas_quality['error_z'],
            # 'mean_rmse_x': test_col_global_meas_quality['rmse_x'],
            # 'mean_rmse_y': test_col_global_meas_quality['rmse_y'],
            # 'mean_rmse_xy': test_col_global_meas_quality['rmse_xy'],
            'mean_rmse_z': test_col_global_meas_quality['rmse_z'],
            # 'mean_std_x': test_col_global_meas_quality['std_x'],
            # 'mean_std_y': test_col_global_meas_quality['std_y'],
            # 'mean_std_z': test_col_global_meas_quality['std_z'],
            # 'number_idd': test_col_global_meas_quality['num_idd'],
            'number_z_meas': test_col_global_meas_quality['num_meas'],
            'percent_particles_measured': test_col_global_meas_quality['percent_meas'],
            'mean_cm': test_col_global_meas_quality['cm'],
            'max_sim': test_col_global_meas_quality['max_sim']
        }
    elif test_col_global_meas_quality is not None:
        test_col_global_meas_quality_dict = {
            # 'mean_error_z': test_col_global_meas_quality['error_z'],
            'mean_rmse_z': test_col_global_meas_quality['rmse_z'],
            # 'mean_std_z': test_col_global_meas_quality['std_z'],
            # 'number_idd': test_col_global_meas_quality['num_idd'],
            'number_z_meas': test_col_global_meas_quality['num_meas'],
            'percent_particles_measured': test_col_global_meas_quality['percent_meas'],
            'mean_cm': test_col_global_meas_quality['cm'],
            'max_sim': test_col_global_meas_quality['max_sim']
        }
    else:
        test_col_global_meas_quality_dict = {}

    dicts = [optics_dict, test_col_global_meas_quality_dict, calib_settings_dict, calib_col_dict, calib_col_stats_dict,
             calib_stack_data_dict, test_settings_dict, test_col_dict, test_col_stats_dict]

    for d in dicts:
        export_data.update(d)

    export_df = pd.DataFrame.from_dict(data=export_data, orient='index')

    savedata = join(test_settings.outputs.results_path,
                    'gdpyt_{}_{}_{}.xlsx'.format(test_settings.outputs.save_id_string,
                                                 calib_settings.outputs.save_id_string, export_data['date_and_time']))
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

    if test_col_global_meas_quality is not None:
        test_col_global_meas_quality_dict = {
            'mean_rmse_z': test_col_global_meas_quality['rmse_z'],
            # 'number_idd': test_col_global_meas_quality['num_idd'],
            'number_z_meas': test_col_global_meas_quality['num_meas'],
            'percent_particles_measured': test_col_global_meas_quality['percent_meas'],
            'mean_cm': test_col_global_meas_quality['cm'],
            'max_sim': test_col_global_meas_quality['max_sim']
        }
    else:
        test_col_global_meas_quality_dict = {}

    dicts = [optics_dict, test_col_global_meas_quality_dict, test_col_stats_dict, calib_col_stats_dict, test_col_dict,
             calib_stack_data_dict, calib_col_dict, test_settings_dict, calib_settings_dict]

    for d in dicts:
        export_data.update(d)

    export_df = pd.DataFrame.from_dict(data=export_data, orient='index')

    savedata = join(test_settings.outputs.results_path,
                    'gdpyt_key_results_{}_{}_{}.xlsx'.format(test_settings.outputs.save_id_string,
                                                             calib_settings.outputs.save_id_string,
                                                             export_data['date_and_time']))
    export_df.to_excel(savedata)

    return export_df


def export_local_meas_quality(calib_settings, test_settings, test_col_local_meas_quality):
    """
    Export local measurement quality to Excel
    """
    calib_string = calib_settings.outputs.save_id_string
    if len(calib_string) > 20:
        calib_string = calib_string[-20:]

    test_string = test_settings.outputs.save_id_string
    if len(test_string) > 20:
        test_string = test_string[-20:]

    savedata = join(test_settings.outputs.results_path,
                    'test_local_t{}_c{}_{}.xlsx'.format(calib_string, test_string, datetime.now()))

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
        df['error'] = df['z'] - df['z_true']
        df['z_error'] = df['error']
        df.sort_values(by='error')

        calib_string = calib_settings.outputs.save_id_string
        if len(calib_string) > 20:
            calib_string = calib_string[-20:]

        test_string = test_settings.outputs.save_id_string
        if len(test_string) > 20:
            test_string = test_string[-20:]

        savedata = join(test_settings.outputs.results_path,
                        'test_coords_t{}_c{}_{}.xlsx'.format(calib_string, test_string, datetime.now()))

    df.to_excel(savedata, index=False)

    return df


def export_calib_stats(calib_settings, calib_col_image_stats, calib_stack_data):
    """
    Export calibration stats to Excel
    """

    savedata = join(calib_settings.outputs.results_path,
                    'calib_col_image_stats_{}_{}.xlsx'.format(calib_settings.outputs.save_id_string, datetime.now()))

    calib_col_image_stats.to_excel(savedata, index=False)

    savedata = join(calib_settings.outputs.results_path,
                    'calib_stack_data_{}_{}.xlsx'.format(calib_settings.outputs.save_id_string, datetime.now()))

    calib_stack_data = calib_stack_data.sort_values('layers', ascending=False)
    calib_stack_data.to_excel(savedata, index=False)