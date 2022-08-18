# GdpytSetup
"""
This script:
    1. creates GDPyT classes which hold input/output variables for GDPyT functions
    2. creates a test harness to simplify GDPyT characterization across multiple parameters

    The Goal:
        Simplify GDPyT testing to a single API.
"""

# imports
from os.path import join
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.morphology import disk, square


class GdpytSetup(object):

    def __init__(self, inputs, outputs, processing, z_assessment, optics):
        """

        Parameters
        ----------
        inputs
        outputs
        processing
        z_assessment
        optics
        """
        self.inputs = inputs
        self.outputs = outputs
        self.processing = processing
        self.z_assessment = z_assessment
        self.optics = optics


class inputs(object):
    def __init__(self, dataset, image_collection_type=None, image_path=None, image_file_type=None,
                 image_base_string=None,
                 image_subset=None, calibration_z_step_size=1, baseline_image=None, static_templates=False,
                 if_image_stack='first', take_image_stack_subset_mean_of=[], single_particle_calibration=False,
                 ground_truth_file_path=None, ground_truth_file_type=None, known_z=None, true_number_of_particles=None,
                 hard_baseline=False, overlapping_particles=True, use_stack_id=None):
        """

        Parameters
        ----------
        dataset
        image_collection_type
        image_path
        image_file_type
        image_base_string
        image_subset
        calibration_z_step_size
        baseline_image
        if_image_stack
        take_image_stack_subset_mean_of
        ground_truth_file_path
        ground_truth_file_type
        true_number_of_particles
        use_stack_id
        """
        # image collection ID and type
        self.dataset = dataset
        self.image_collection_type = image_collection_type

        # file paths
        self.image_path = image_path
        self.image_file_type = image_file_type
        self.ground_truth_file_path = ground_truth_file_path
        self.ground_truth_file_type = ground_truth_file_type

        # file information
        self.image_base_string = image_base_string

        # dataset
        self.single_particle_calibration = single_particle_calibration
        self.image_subset = image_subset
        self.calibration_z_step_size = calibration_z_step_size
        self.baseline_image = baseline_image
        self.hard_baseline = hard_baseline
        self.static_templates = static_templates
        self.overlapping_particles = overlapping_particles
        self.if_image_stack = if_image_stack
        self.take_image_stack_subset_mean_of = take_image_stack_subset_mean_of
        self.true_number_of_particles = true_number_of_particles
        self.known_z = known_z
        self.use_stack_id = use_stack_id


class outputs(object):
    def __init__(self, results_path=None, save_id_string=None, show_plots=False, save_plots=False,
                 inspect_contours=False, assess_similarity_for_all_stacks=False):
        """

        Parameters
        ----------
        results_path
        show_plots
        save_plots
        inspect_contours
        """
        self.results_path = results_path
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.save_id_string = save_id_string
        self.inspect_contours = inspect_contours
        self.assess_similarity_for_all_stacks = assess_similarity_for_all_stacks


class processing(object):
    def __init__(self, min_layers_per_stack=None,
                 cropping_params=None,
                 background_subtraction=None,
                 filter_params=None,
                 threshold_params=None,
                 processing_method=None, processing_filter_type=None, processing_filter_size=None,
                 threshold_method=None, threshold_modifier=None,
                 min_particle_area=5, max_particle_area=1000, template_padding=3, dilate=True,
                 shape_tolerance=0.75, same_id_threshold_distance=5, overlap_threshold=0.3,
                 stacks_use_raw=False, zero_calib_stacks=False, zero_stacks_offset=0, xy_displacement=[[0, 0]]):
        """
        Parameters
        ----------
        min_layers_per_stack
        threshold_params
        background_subtraction
        threshold_method
        threshold_modifier
        shape_tolerance
        min_particle_area
        max_particle_area
        template_padding
        dilate
        same_id_threshold_distance
        stacks_use_raw
        zero_calib_stacks
        zero_stacks_offset
        """

        self.template_padding = template_padding
        self.dilate = dilate
        self.shape_tolerance = shape_tolerance
        self.overlap_threshold = overlap_threshold
        self.min_particle_area = min_particle_area
        self.max_particle_area = max_particle_area
        self.same_id_threshold_distance = same_id_threshold_distance
        self.min_layers_per_stack = min_layers_per_stack
        self.zero_calib_stacks = zero_calib_stacks
        self.zero_stacks_offset = zero_stacks_offset
        self.xy_displacement = xy_displacement
        self.stacks_use_raw = stacks_use_raw
        self.background_subtraction = background_subtraction
        self.cropping_params = cropping_params
        self.processing_method = processing_method
        self.processing_filter_type = processing_filter_type
        self.processing_filter_size = processing_filter_size
        self.threshold_method = threshold_method
        self.threshold_modifier = threshold_modifier

        self.processing_params = filter_params
        self.threshold_params = threshold_params


class z_assessment(object):
    def __init__(self, infer_method=None, min_cm=0.5, sub_image_interpolation=True, use_stack_id=None):
        """

        Parameters
        ----------
        infer_method
        min_cm
        sub_image_interpolation
        use_stack_id
        """
        self.infer_method = infer_method
        self.min_cm = min_cm
        self.sub_image_interpolation = sub_image_interpolation

        if use_stack_id is None:
            use_stack_id = 0

        self.use_stack_id = use_stack_id


class optics(object):
    def __init__(self, particle_diameter=None, demag=None, magnification=None, numerical_aperture=None,
                 focal_length=None,
                 ref_index_medium=None, ref_index_lens=None, pixel_size=None,
                 pixel_dim_x=None, pixel_dim_y=None, bkg_mean=None, bkg_noise=None, points_per_pixel=None, n_rays=None,
                 gain=None, cyl_focal_length=None, wavelength=None, overlap_scaling=None, z_range=20):
        """

        Parameters
        ----------
        particle_diameter
        demag
        magnification
        numerical_aperture
        focal_length
        ref_index_medium
        ref_index_lens
        pixel_size
        pixel_dim_x
        pixel_dim_y
        bkg_mean
        bkg_noise
        points_per_pixel
        n_rays
        gain
        cyl_focal_length
        wavelength

        Notes:
            20X - LCPlanFL N 20X LCD        [LCPLFLN20xLCD]
                magnification:              20
                numerical_aperture:         0.45
                field_number:               26.5
                working distance:           7.4 - 8.3 mm
                transmittance:              90% @ 425 - 670 nm
                correction collar:          0 - 1.2 mm
                objective lens diameter:    15 mm

                microns per pixel:          1.55 (careful with using this value)
        """
        self.particle_diameter = particle_diameter
        self.demag = demag
        self.magnification = magnification
        self.numerical_aperture = numerical_aperture
        self.focal_length = focal_length
        self.ref_index_medium = ref_index_medium
        self.ref_index_lens = ref_index_lens
        self.pixel_size = pixel_size
        self.pixel_dim_x = pixel_dim_x
        self.pixel_dim_y = pixel_dim_y
        self.bkg_mean = bkg_mean
        self.bkg_noise = bkg_noise
        self.points_per_pixel = points_per_pixel
        self.n_rays = n_rays
        self.gain = gain
        self.cyl_focal_length = cyl_focal_length
        self.wavelength = wavelength
        self.z_range = z_range
        self.overlap_scaling = overlap_scaling

        # effective magnification
        self.effective_magnification = demag * magnification

        # pixels per particle
        self.pixels_per_particle_in_focus = particle_diameter * self.effective_magnification / pixel_size

        # field of view
        self.field_of_view = pixel_size * pixel_dim_x / self.effective_magnification

        # microns per pixel scaling factor
        self.microns_per_pixel = pixel_size / self.effective_magnification

        # Rayleigh Criterion (maximum lateral resolution)
        """
        The Rayleigh Criterion is the maximum lateral resolution (distance) that two point sources can be discerned. 
        It's defined as the distance between the distance where the principal diffraction maximum (central spot of the 
        Airy disk) from one point source overlaps with the first minimum (dark region surrounding the central spot) from 
        the Airy disk of the other point source.
        """
        self.Rayleigh_criterion = 0.61 * wavelength / numerical_aperture

        # depth of field
        """
        The depth of field is made up of two terms:
            1. at higher numerical apertures, the wave optics dominates the depth of field (left term)
            2. at lower numerical apertures, the circle of confusion dominates; which is dependent on the CCD pixel size
        """
        self.depth_of_field = ref_index_medium * wavelength / numerical_aperture ** 2 + ref_index_medium * pixel_size / \
                              (self.effective_magnification * numerical_aperture)

        # constants for stigmatic/astigmatic imaging systems (ref: Rossi & Kahler 2014, DOI 10.1007/s00348-014-1809-2)
        self.c1 = 2 * (ref_index_medium ** 2 / numerical_aperture ** 2 - 1) ** -0.5
        self.c2 = (particle_diameter ** 2 + 1.49 * wavelength ** 2 * (
                    ref_index_medium ** 2 / numerical_aperture ** 2 - 1)) ** 0.5

        # create a measurement depth
        z_space = np.linspace(start=-z_range, stop=z_range, num=250)

        # particle image diameter with distance from focal plane (stigmatic system)
        # (ref 1: Rossi & Kahler 2014, DOI 10.1007 / s00348-014-1809-2)
        self.particle_diameter_z1 = self.effective_magnification * (self.c1 ** 2 * z_space ** 2 + self.c2 ** 2) ** 0.5
        # (ref 2: Rossi & Kahler 2014, )
        self.particle_diameter_z2 = self.effective_magnification * self.particle_diameter

    def stigmatic_diameter_z(self, z_space, z_zero):
        """
        particle diameter with distance from the focal plane (stigmatic system)

        z_space:
        z_zero:
        """
        # if units of z are not in microns, adjust prior to calculating the intensity profile
        mod = False
        if np.max(np.abs(z_space)) > 1:
            z_space = z_space * 1e-6
            mod = True

        # create dense z-space for smooth plotting
        z_space = np.linspace(start=np.min(z_space), stop=np.max(z_space), num=250)

        # particle image diameter with distance from focal plane (stigmatic system)
        # (ref 1: Rossi & Kahler 2014, DOI 10.1007 / s00348-014-1809-2)
        particle_diameter_z1 = self.effective_magnification * (
                    self.c1 ** 2 * (z_space - z_zero * 1e-6) ** 2 + self.c2 ** 2) ** 0.5

        # convert units to microns
        particle_diameter_z1 = particle_diameter_z1 * 1e6

        if mod is True:
            z_space = z_space * 1e6

        return z_space, particle_diameter_z1

    def stigmatic_maximum_intensity_z(self, z_space, max_intensity_in_focus, z_zero=0, background_intensity=0,
                                      num=250):
        """
        maximum intensity with distance from the focal plane (stigmatic system)

        z_space: list or array like; containing all the non-normalized z-coordinates in the collection
        max_intensity_in_focus: float; maximum mean particle intensity in the collection
            * Note: this is the maximum of the mean of the contour areas (region of signal) in the collection.
        """
        # if units of z are not in microns, adjust prior to calculating the intensity profile
        mod = False
        if np.max(np.abs(z_space)) > 1:
            z_space = z_space * 1e-6
            mod = True

        # create dense z-space for smooth plotting
        z_space = np.linspace(start=np.min(z_space), stop=np.max(z_space), num=num)

        # calculate the intensity profile as a function of z
        stigmatic_intensity_profile = self.c2 ** 2 / ((self.c1 ** 2 * (z_space - z_zero * 1e-6) ** 2 + self.c2 ** 2) **
                                                      0.5 * (self.c1 ** 2 * (z_space - z_zero * 1e-6) ** 2 +
                                                             self.c2 ** 2) ** 0.5)

        # normalize the intensity profile so it's maximum value is equal to the max_intensity_in_focus
        stigmatic_intensity_profile = np.round(
            (max_intensity_in_focus - background_intensity) * stigmatic_intensity_profile / \
            np.max(stigmatic_intensity_profile) + background_intensity,
            1)

        if mod is True:
            z_space = z_space * 1e6

        return z_space, stigmatic_intensity_profile

    # maximum intensity with distance from the focal plane (astigmatic system)
    # (ref: Rossi & Kahler 2014, DOI 10.1007 / s00348-014-1809-2)