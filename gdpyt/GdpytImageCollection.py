import matplotlib.pyplot as plt

from .GdpytImage import GdpytImage
from .GdpytCalibrationSet import GdpytCalibrationSet
from gdpyt.utils.plotting import *
from gdpyt.similarity.correlation import parabola
from gdpyt.utils import binning, fit
from gdpyt.subpixel_localization import gaussian
from gdpyt.subpixel_localization.gaussian import fit as fit_gaussian_subpixel

from os.path import join, isdir
from os import listdir
import re
from collections import OrderedDict

from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import curve_fit
from gdpyt.subpixel_localization.gaussian import gaussian1D
from sklearn.neighbors import NearestNeighbors

from skimage import filters
from skimage.morphology import disk, square, opening

import pandas as pd
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class GdpytImageCollection(object):

    def __init__(self, folder, filetype,
                 file_basestring=None, image_collection_type=None,
                 crop_specs=None, processing_specs=None, thresholding_specs=None, background_subtraction=None,
                 overlapping_particles=True, baseline=None, hard_baseline=False, particle_id_image=None,
                 static_templates=False, single_particle_calibration=False, calibration_stack_z_step=None,
                 stacks_use_raw=False, infer_sub_image=True, template_padding=1,
                 subset=None, if_img_stack_take='mean', take_subset_mean=None,
                 min_particle_size=None, max_particle_size=None,
                 shape_tol=0.2, overlap_threshold=0.3, same_id_threshold=10,
                 folder_ground_truth=None, true_num_particles=None, xydisplacement=None, measurement_depth=None,
                 optics_setup=None,
                 inspect_contours_for_every_image=False,
                 exclude=[], ):
        """

        Parameters
        ----------
        folder
        filetype
        crop_specs
        processing_specs
        thresholding_specs
        background_subtraction
        min_particle_size
        max_particle_size
        shape_tol
        overlap_threshold
        exclude
        subset
        folder_ground_truth
        stacks_use_raw
        infer_sub_image
        measurement_depth
        true_num_particles
        if_img_stack_take
        take_subset_mean
        inspect_contours_for_every_image
        template_padding
        file_basestring
        same_id_threshold
        image_collection_type
        calibration_stack_z_step
        """
        super(GdpytImageCollection, self).__init__()

        if not isdir(folder):
            raise ValueError("Specified folder {} does not exist".format(folder))

        # experimental details
        self.optics = optics_setup

        # properties of the image collection
        self._image_collection_type = image_collection_type
        self._folder = folder
        self._filetype = filetype
        self._file_basestring = file_basestring
        self._folder_ground_truth = folder_ground_truth

        # image collection data
        exclude = []
        self.in_focus_z = None
        self._in_focus_area = None
        self.in_focus_coords = None
        self._particle_ids = None
        self._true_num_particles = true_num_particles
        self._calibration_stack_z_step = calibration_stack_z_step

        # image processing filters
        self._min_particle_size = min_particle_size
        self._max_particle_size = max_particle_size
        self._same_id_threshold = same_id_threshold
        self._overlap_threshold = overlap_threshold
        self._shape_tol = shape_tol

        # toggles for loading images
        self._if_img_stack_take = if_img_stack_take
        self._take_subset_mean = take_subset_mean
        if if_img_stack_take == 'first':
            self._num_images_if_mean = 1
        else:
            self._num_images_if_mean = take_subset_mean[1] - take_subset_mean[0]

        # toggles for inspection
        self._inspect_contours_for_every_image = inspect_contours_for_every_image

        # toggles for calibration stacks and inference
        self.baseline = baseline
        self._static_templates = static_templates
        self._single_particle_calibration = single_particle_calibration
        self.particle_id_image = particle_id_image
        self._overlapping_particles = overlapping_particles
        self._template_padding = template_padding
        self._stacks_use_raw = stacks_use_raw
        self._infer_sub_image = infer_sub_image

        # if a subset is supplied, add all files not in the subset to exclude
        self._get_exclusion_subset(exclude=exclude, subset=subset)

        # find all files in the directory; store the total number of files (which is important in calculating the
        # measurment depth), and only save the files not in exclude.
        self._find_files(exclude=exclude)

        # find all the files in the ground truth folder
        self._find_files_ground_truth(exclude=exclude)

        # set the measurement depth (based on the total number of files from _find_files() )
        # set the measurement range (based on the number of files actually used in the collection)
        self._set_measurement_depth(measurement_depth)

        # add images
        self._add_images()

        # Define the cropping done on all the images in this collection
        if crop_specs is None:
            self._crop_specs = {}
        else:
            self._crop_specs = crop_specs
            self.crop_images()

        # Define the background image subtraction method
        if background_subtraction is None:
            self._background_subtraction = {}
        else:
            self._background_subtraction = background_subtraction
            self._background_subtract()

        # Define the processing done on all the images in this collection
        if processing_specs is None:
            self._processing_specs = {}
        else:
            self._processing_specs = processing_specs

        # Define the thresholding done on all the images in this collection
        self._thresholding_specs = thresholding_specs

        # calculate the average of all images in the collection
        # self._calculate_mean_image()

        # filter and identify contours
        self.filter_images()
        self.identify_particles()

        # uniformize particle ids across all images in the collection
        self._hard_baseline = hard_baseline
        self.uniformize_particle_ids(baseline=baseline, uv=xydisplacement)

        self.identify_particles_ground_truth()
        self.refine_particles_ground_truth()

        # initialize image particle dataframe
        self.spct_stats = None
        self.spct_particle_defocus_stats = None
        self.spct_population_defocus_stats = None
        self.fitted_plane_equation = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        key = list(self.images.keys())[item]
        return self.images[key]

    def __repr__(self):
        class_ = 'GdpytImageCollection'
        repr_dict = {'Folder': self.folder,
                     'Filetype': self.filetype,
                     'Number of images': len(self),
                     'Min. and max. particle sizes': [self._min_particle_size, self._max_particle_size],
                     'Shape tolerance': self._shape_tol,
                     'Preprocessing': self._processing_specs}
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def _add_images(self):
        images = OrderedDict()
        frame = 0
        for file in self._files:
            img = GdpytImage(join(self.folder, file), frame=frame, if_img_stack_take=self._if_img_stack_take,
                             take_subset_mean=self._take_subset_mean, true_num_particles=self._true_num_particles)
            images.update({img.filename: img})
            logger.warning('Loaded image {}'.format(img.filename))
            frame += 1
        self._images = images

    def _get_exclusion_subset(self, exclude, subset):
        """
        modify the variable [exclude] to include all files not in the [subset]
        """
        if subset is None:
            pass
        else:
            base_string = self._file_basestring
            all_files = listdir(self._folder)
            save_files = []

            for file in all_files:
                if file.endswith(self._filetype):
                    if file in exclude:
                        continue
                    save_files.append(file)

            # if subset is an integer, this indicates the total number of files to include.
            #   Note: the files are randomly selected from the collection.
            if len(subset) == 1:
                random_files = [rf for rf in random.sample(set(save_files), subset[0])]
                for f in save_files:
                    if f not in random_files:
                        exclude.append(f)

            # if subset is a two element list, this indicates a start and stop range for z-values.
            elif len(subset) == 2:
                start = subset[0]
                stop = subset[1]
                for f in save_files:
                    search_string = base_string + '(.*)' + self._filetype
                    file_index = float(re.search(search_string, f).group(1))
                    if file_index < start or file_index > stop:
                        exclude.append(f)

            # if subset is a three element list, this indicates a start and stop z-range and sampling rate.
            elif len(subset) == 3:

                protected_files = []
                # we always want to include the baseline or particle_id_image in the image collection.
                if self.baseline is not None:
                    if isinstance(self.baseline, GdpytCalibrationSet):
                        pass
                    else:
                        protected_files.append(self.baseline)
                if self.particle_id_image is not None:
                    protected_files.append(self.particle_id_image)

                start = subset[0]
                stop = subset[1]
                subset_files = []
                subset_index = []

                # get files within subset z-coordinates
                for f in save_files:
                    search_string = base_string + '(.*)' + self._filetype
                    file_index = float(re.search(search_string, f).group(1))
                    if file_index >= start and file_index <= stop:
                        subset_files.append(f)
                        subset_index.append(file_index)

                # sort the zipped list of files and indices (floats for the file's z-coordinate)
                sorted_subset = sorted(list(zip(subset_files, subset_index)), key=lambda x: x[1])

                # sample the list according to the third element in subset
                sorted_files, sorted_indices = map(list, zip(*sorted_subset))
                n_sampling = subset[2]
                sampled_sorted_subset_files = sorted_files[::n_sampling]

                # append files not sampled to exclude
                for f in save_files:
                    if f not in sampled_sorted_subset_files + protected_files:
                        exclude.append(f)

            else:
                raise ValueError("Collecting multiple subsets is not implemented at the moment.")

    def _find_files(self, exclude=[]):
        """
        Identifies all files of filetype filetype in folder
        :return:
        """
        all_files = listdir(self._folder)
        number_of_files = 0
        save_files = []
        for file in all_files:
            if file.endswith(self._filetype):
                if file in exclude:
                    number_of_files += 1
                    continue
                save_files.append(file)
                number_of_files += 1

        save_files = sorted(save_files,
                            key=lambda filename: float(filename.split(self.file_basestring)[-1].split('.')[0]))

        logger.warning(
            "Found {} files with filetype {} in folder {}".format(len(save_files), self.filetype, self.folder))
        # Save all the files of the right filetype in this attribute
        self._files = save_files
        self._total_number_of_files = number_of_files

    def _find_files_ground_truth(self, exclude=[]):
        if self._folder_ground_truth is None:
            logger.warning("No ground truth")
            self._files_ground_truth = None
        elif self._folder_ground_truth == 'standard_gdpt':
            logger.warning("Using standard GDPT particle locations: [29, 29]")
            self._files_ground_truth = None
        else:
            all_files = listdir(self._folder_ground_truth)
            save_files = []
            for file in all_files:
                if file.endswith('.txt'):
                    if file in exclude:
                        continue
                    save_files.append(file)

            logger.warning(
                "Found {} files with filetype {} in ground truth folder {}".format(len(save_files), '.txt',
                                                                                   self._folder_ground_truth))
            # Save all the files of the right filetype in this attribute
            self._files_ground_truth = save_files

    def create_calibration(self, name_to_z, dilate=False, template_padding=0, min_num_layers=None,
                           self_similarity_method='sknccorr', exclude=[]):
        """
        This method:
            1. creates a calibration set from the image collection.
            2. finds the in-focus z-coordinate for every particle
            3. stores the in-focus x,y-coordinates and interpolated z and area values for every particle in a dataframe.
            4. sets the in_focus_z attribute.

        :param name_to_z: dictionary, maps each filename to a height. e.g {'run5_0.tif': 0.0, 'run5_1.tif': 1.0 ...}
                        This could also be done differently
        :return: A list of GdptCalibrationStacks. One for each particle in the images
        """

        # create the calibration set
        calibration_set = GdpytCalibrationSet(self, name_to_z, dilate=dilate, template_padding=template_padding,
                                              min_num_layers=min_num_layers,
                                              self_similarity_method=self_similarity_method,
                                              exclude=exclude)

        # find the in-focus coordinates of all particles and set the particle's in-focus z-coordinate and area
        # self.find_particle_in_focus_z(use_true_z=False, use_peak_int=True)

        # determine the z-coordinate where most particles are in focus and set this as the collection in_focus_z
        # self.find_collection_z_of_min_area()
        # self.find_collection_in_focus_z() --> This function doesn't seem to work and is a duplicate of ^

        return calibration_set

    def create_model_based_z_calibration(self):
        # code for model based z-assessment
        pass

    def crop_images(self):
        for image in self.images.values():
            image.crop_image(self._crop_specs)
            logger.warning("Cropped image {}".format(image.filename))

    def _background_subtract(self):

        if isinstance(self._background_subtraction, str):
            background_subtraction_method = self._background_subtraction
            background_subtraction_param = 125
        elif isinstance(self._background_subtraction, dict):
            background_subtraction_method = self._background_subtraction['method']
            background_subtraction_param = self._background_subtraction['param']
        else:
            raise ValueError('background subtraction variable must be either a string or dict.')

        sizey, sizex = np.shape((list(self.images.values())[0]._raw))

        if background_subtraction_method == 'manual':
            background_img = np.ones((sizey, sizex), dtype=np.uint16) * background_subtraction_param

        elif background_subtraction_method == 'median':
            background_imgs = [i.raw for i in list(self.images.values())]
            background_img = np.median(background_imgs, axis=0)

        elif background_subtraction_method == 'baseline_image_subtraction':
            baseline_image = self.images[self.baseline].raw

            footprint = disk(self._min_particle_size * 3)
            res = opening(baseline_image, footprint)
            res = filters.gaussian(res, sigma=2, preserve_range=True)
            background_img = np.rint(res).astype(np.uint16)

        elif background_subtraction_method in ['min', 'min_limit_max', 'min_value']:

            # --- compute the mean image intensity percentile across all images ---
            background_add = np.zeros((sizey, sizex), dtype=np.uint16)

            for i in self.images.values():
                image = i._raw.copy()
                background_add = background_add + image  # add images

            # take mean
            background_mean = np.divide(background_add, len(self.images))

            # --- compute the minimum pixel intensity across all images ---
            if background_subtraction_method == 'min_limit_max':
                background_img = np.ones((sizey, sizex), dtype=np.uint16) * background_subtraction_param
                background_value = background_subtraction_param
            else:
                background_img = np.ones((sizey, sizex), dtype=np.uint16) * 2 ** 16
                background_value = 2 ** 16

            # loop through images
            for i in self.images.values():
                image = i._raw.copy()
                if background_subtraction_method in ['min', 'min_limit_max']:
                    background_img = np.where(image < background_img, image, background_img)  # take min value
                elif background_subtraction_method == 'min_value':
                    background_value = np.where(np.min(image) < background_value, np.min(image), background_value)
                    background_img = np.ones((sizey, sizex), dtype=np.uint16) * background_value
                else:
                    raise ValueError(
                        "{} background subtraction method is not yet implemented".format(background_subtraction_method))

        # store background image
        self._background_img = background_img

        # perform background subtraction
        for image in self.images.values():
            image.subtract_background(background_subtraction_method, self._background_img)
            logger.warning("Background subtraction image {}".format(image.filename))

    def _calculate_mean_image(self):
        """ Calculate the mean image by summing across all images and dividing by the number of images """

        background_add = np.zeros_like(self.images[self.baseline].raw, dtype=np.uint16)
        for i in self.images.values():
            background_add = background_add + i.raw.copy()

        background_mean_float = np.divide(background_add, len(self.images))

        background_mean = np.rint(background_mean_float).astype(np.uint16)

        fig, ax = plt.subplots()
        ax.imshow(background_mean)
        ax.set_title('Max, Mean = {}, {}'.format(np.max(background_mean), np.mean(background_mean)))
        plt.show()

        self.mean_image = background_mean

    def filter_images(self):
        for image in self.images.values():
            image.filter_image(self._processing_specs)
            logger.warning("{} filtered image {}".format(self._processing_specs.keys(), image.filename))

    def identify_particles(self):
        """
        Options for identifying particles:

        Calibration Collection:
            1. if a baseline is passed:
                1.1 if static_templates is True:
                    assign particle ID's based on baseline image (where baseline image is static)
                1.2 if static templates is False:
                    assign particle ID's based on baseline image (where baseline image is dynamic)
            2. if a baseline is not passed:
                assign particle ID's randomly (the templates are dynamic)

        Test Collection:
            1. if baseline is a CalibrationSet:
                1.1 if static_templates is True:
                    assign particle ID's based on the baseline image (where baseline image is static)
                1.2 if static templates is False:
                    assign particle ID's randomly (the templates are dynamic)
            2. if baseline is an image:
                2.1 if static_templates is True:
                    assign particle ID's based on baseline image (where baseline image is static)
                2.2 if static templates is False:
                    assign particle ID's randomly (the templates are dynamic)
            3. if baseline is None:
                3.1 if static_templates is True:
                    assign particle ID's based on baseline image (where baseline image is static)
                3.2 if static templates is False:
                    assign particle ID's randomly (the templates are dynamic)
        """

        if self._image_collection_type == 'calibration':
            if self._static_templates is True:
                particle_identification_image = self.images[self.baseline].filtered
            else:
                particle_identification_image = None
        elif self._image_collection_type == 'test':
            if isinstance(self.baseline, GdpytCalibrationSet):
                if self._static_templates is True:
                    particle_identification_image = self.images[self.particle_id_image].filtered
                else:
                    particle_identification_image = None
            elif self._static_templates is True:
                particle_identification_image = self.images[self.baseline].filtered
            else:
                particle_identification_image = None
        else:
            raise ValueError("Image collection must either be 'calibration' or 'test'.")

        for image in self.images.values():
            image.identify_particles_sk(self._thresholding_specs,
                                        min_size=self._min_particle_size, max_size=self._max_particle_size,
                                        shape_tol=self._shape_tol, overlap_threshold=self._overlap_threshold,
                                        same_id_threshold=self._same_id_threshold,
                                        inspect_contours_for_every_image=self._inspect_contours_for_every_image,
                                        padding=self._template_padding,
                                        image_collection_type=self._image_collection_type,
                                        particle_id_image=particle_identification_image,
                                        overlapping_particles=self._overlapping_particles,
                                        template_use_raw=self._stacks_use_raw)
            logger.info("Identified {} particles on image {}".format(len(image.particles), image.filename))

    def _set_measurement_depth(self, measurement_depth):
        """
        Sets the measurement depth which is important in defining the true_z value of calibration stacks
        """
        if measurement_depth and self._image_collection_type == 'test':
            self._measurement_depth = measurement_depth

        elif self._folder_ground_truth == 'standard_gdpt' and self._image_collection_type == 'calibration':
            # in this case, the calibration images at the measurment depth boundaries are centered at z = 0 and z = 86.
            self._measurement_depth = self._total_number_of_files * self._calibration_stack_z_step
            self._measurement_range = len(self._files) * self._calibration_stack_z_step

        elif self._calibration_stack_z_step and self._image_collection_type == 'calibration':
            self._measurement_depth = self._total_number_of_files * self._calibration_stack_z_step
            self._measurement_range = len(self._files) * self._calibration_stack_z_step

        else:
            raise ValueError("If this is a calibration collection, then the measurement depth should be set by the "
                             "total number of calibration images * z-step per calibration image. If this is a test"
                             "collection, then the measurement depth needs to be directly inputted into"
                             "GdpytImageCollection class.")

    def identify_particles_ground_truth(self):

        if self._folder_ground_truth == 'standard_gdpt':
            for image in self.images.values():
                for p in image.particles:
                    x_true = 29
                    y_true = 29
                    p._set_location_true(x=x_true, y=y_true)

        elif self._folder_ground_truth is not None:
            for image in self.images.values():
                filename = image.filename[:-4]
                ground_truth = np.loadtxt(join(self._folder_ground_truth, filename + '.txt'))
                if len(ground_truth.shape) < 2:
                    ground_truth_xyz = ground_truth[np.newaxis, 0:3]
                else:
                    ground_truth_xyz = ground_truth[:, 0:3]
                image.set_true_num_particles(data=ground_truth_xyz)
                image._update_particle_density_stats()

                for p in image.particles:
                    neigh = NearestNeighbors(n_neighbors=1)
                    neigh.fit(ground_truth_xyz[:, 0:2])
                    result = neigh.kneighbors([p.location])

                    # TODO: add a logger.Warning if the nearest particle is > same_id_threshold (likely something wrong)
                    x_true = ground_truth_xyz[result[1][0][0]][0]
                    y_true = ground_truth_xyz[result[1][0][0]][1]
                    z_true = ground_truth_xyz[result[1][0][0]][2]
                    if z_true == 0.0:
                        z_true = float(0.000001)
                    p._set_location_true(x=x_true, y=y_true, z=z_true)

        else:
            # there is no reason to set the ground truth on an image collection that doesn't have ground truth
            # coordinates. The calibration stack/set class has it's own name_to_z mapping function and this should be
            # the only place where such a mapper occurs. Also, the calibration stack/set will assign a z-value to
            # the images in the calibration stack/set and this will automatically update all the particles in those
            # images
            pass

    def refine_particles_ground_truth(self):

        if self._folder_ground_truth == 'standard_gdpt':
            pass

        elif self._folder_ground_truth is not None:
            pass

            # find the in-focus coordinates of all particles and set the particle's in-focus z-coordinate and area
            # self.find_particle_in_focus_z(use_true_z=True)

            # determine the z-coordinate where most particles are in focus and set this as the collection in_focus_z
            # self.find_collection_z_of_min_area()

            """if self.image_collection_type == 'test':
                df_min_area_z_offset = 0.49125 - df_min_area_z  # TODO: fix this, it should be taken from the calibration stack
                logger.warning("USING Z-OFFSET VALUE OF 0.49125!!")

                for img in self.images.values():
                    for p in img.particles:
                        new_z_true = p.z_true + df_min_area_z_offset
                        p._set_true_z(new_z_true)"""

    def uniformize_particle_ids_and_groups(self, baseline=None, baseline_img=None, hard_baseline=False, uv=None):
        """
        Parameters
        ----------
        baseline: only use in RARE cases where you already have the location of the particles and their IDs.
        threshold: maximum x-y displacement between images to be assigned the same ID.
        uv: a simulated displacement between images
        baseline_img: should be the filename (e.g. calib_23.tif) of the image file to use as the baseline for ID assignment
        """
        if uv is None:
            uv = [[0, 0]]

        baseline_locations = []
        # If a calibration set is given as the baseline, the particle IDs in this collection are assigned based on
        # the location and ID of the calibration set. This should always be done when the collection contains target images
        if baseline is not None:
            if isinstance(baseline, GdpytCalibrationSet):
                for stack in baseline.calibration_stacks.values():
                    baseline_locations.append(pd.DataFrame({'x': stack.location[0], 'y': stack.location[1]},
                                                           index=[stack.id]))
                skip_first_img = False

            elif isinstance(baseline, GdpytImage):
                baseline_img = baseline

                for particle in baseline_img.particles:
                    baseline_locations.append(pd.DataFrame({'x': particle.location[0], 'y': particle.location[1]},
                                                           index=[particle.id]))
                skip_first_img = False
            elif isinstance(baseline, str):
                baseline_img = self.images[baseline]
                for particle in baseline_img.particles:
                    baseline_locations.append(pd.DataFrame({'x': particle.location[0], 'y': particle.location[1]},
                                                           index=[particle.id]))
                skip_first_img = False
            else:
                raise TypeError("Invalid type for baseline")

        # If no baseline is given, the particle IDs are assigned based on the IDs and location of the particles in the
        # baseline_img or else the first image
        else:
            index = 0
            if baseline_img is not None:
                index = self._files.index(baseline_img)
            baseline_img = self._files[index]
            baseline_img = self.images[baseline_img]

            for particle in baseline_img.particles:
                baseline_locations.append(pd.DataFrame({'x': particle.location[0], 'y': particle.location[1]},
                                                       index=[particle.id]))

            skip_first_img = True

        if len(baseline_locations) == 0:
            baseline_locations = pd.DataFrame()
            next_id = None
        else:
            baseline_locations = pd.concat(baseline_locations).sort_index()
            # The next particle that can't be matched to a particle in the baseline gets this id
            next_id = len(baseline_locations)

        for i, file in enumerate(self._files):
            if (i == 0) and skip_first_img:
                continue
            image = self.images[file]
            # Convert to list because ordering is important
            particles = [particle for particle in image.particles]
            locations = [list(p.location) for p in particles]

            if len(locations) == 0:
                continue
            if baseline_locations.empty:
                dfs = [pd.DataFrame({'x': p.location[0], 'y': p.location[1]}, index=[p.id]) for p in particles]
                baseline_locations = pd.concat(dfs)
                next_id = len(baseline_locations)
                continue

            # TODO: set NearestNeighbors(n_neighbors=2) and if a particle center is within an "overlap_range" then,
            # merge those contours and maintain that particle's ID. Do this same function for the other merged particle
            # (i.e. maintain it's ID but merge the contours).

            # NearestNeighbors(x+u,y+v): previous location (x,y) + displacement guess (u,v)
            nneigh = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(baseline_locations.values + uv)
            # NOTE: the displacement guess (u,v) could incorporate a "time" variable (image number or time data)
            # such that [u_i,v_i] = [u(t), v(t)] in order to better match time-dependent or periodic displacements.

            distances, indices = nneigh.kneighbors(np.array(locations))

            remove_p_not_in_calib = []
            for distance, idx, particle in zip(distances, indices, particles):
                # If the particle is close enough to a particle of the baseline, give that particle the same ID as the
                # particle in the baseline
                """
                Maintaining single particle identification throughout contour merging due to defocused image overlap:
                    1. if the identified contour center is close enough to a single baseline particle:
                        1.1 assign it the same ID and,
                        1.2 update the baseline location.
                    2. if the identified contour center is close enough to two (or more) baseline particles:
                        2.1 create two (or more) new particles at the location of the previous baseline with the
                        previous bounding box size.
                        2.2 do not update the baseline location.
                """
                # get the number of nearby particles
                dist_close = [d for d in distance if d < self._same_id_threshold]
                num_close = len(dist_close)

                indic = 0
                for dist_i, idx_i in zip(distance, idx):

                    if dist_i < 15:

                        indic += 1

                        particle.set_id(baseline_locations.index.values[idx_i.squeeze()])

                        if num_close == 1:
                            baseline_locations.loc[particle.id, ('x', 'y')] = (
                                particle.location[0], particle.location[1])
                        else:
                            baseline_locations.loc[particle.id, ('x', 'y')] = (
                                particle.location[0], particle.location[1])


                    elif indic == 0:
                        # If the particle is not in the baseline, we may remove it via two methods:
                        #   1. if the baseline is a CalibrationSet, as including will significantly reduce accuracy.
                        #   2. if we designate a "hard baseline" where we don't want to add new particles.

                        # filter if not in CalibrationSet baseline:
                        if isinstance(baseline, GdpytCalibrationSet):
                            remove_p_not_in_calib.append(particle)
                            logger.warning(
                                "Removed particle {} at location {} b/c not in Calibration Set baseline".format(
                                    particle.id,
                                    particle.location))
                            continue

                        # filter if not in "hard baseline":
                        if self._hard_baseline is True or hard_baseline is True:
                            remove_p_not_in_calib.append(particle)
                            logger.warning(
                                "Removed particle {} at location {} b/c not in hard baseline".format(particle.id,
                                                                                                     particle.location))
                            continue

                        # else, assign it a new, non-existent id and add it to the baseline for subsequent images
                        logger.warning("File {}: New IDs: {}".format(file, next_id))
                        particle.set_id(next_id)
                        assert (next_id not in baseline_locations.index)
                        baseline_locations = baseline_locations.append(
                            pd.DataFrame({'x': particle.location[0], 'y': particle.location[1]},
                                         index=[particle.id]))
                        next_id += 1

            for p in remove_p_not_in_calib:
                logger.warning("Removing particle particle {}".format(p.id))
                # TODO: note this is changed
                if p in image.particles:
                    image.particles.remove(p)
            # The nearest neighbor mapping creates particles with duplicate IDs under some circumstances
            # These need to be merged to one giant particle
            image.merge_duplicate_particles()

    def uniformize_particle_ids(self, baseline=None, baseline_img=None, uv=None):
        """
        Parameters
        ----------
        baseline: only use in RARE cases where you already have the location of the particles and their IDs.
        threshold: maximum x-y displacement between images to be assigned the same ID.
        uv: a simulated displacement between images
        baseline_img: should be the filename (e.g. calib_23.tif) of the image file to use as the baseline for ID assignment
        """

        if uv is None:
            uv = [[0, 0]]

        baseline_locations = []
        # If a calibration set is given as the baseline, the particle IDs in this collection are assigned based on
        # the location and ID of the calibration set. This should always be done when the collection contains target images
        if baseline is not None:
            if isinstance(baseline, GdpytCalibrationSet):
                for stack in baseline.calibration_stacks.values():
                    baseline_locations.append(pd.DataFrame({'x': stack.location[0], 'y': stack.location[1]},
                                                           index=[stack.id]))
                skip_first_img = False

            elif isinstance(baseline, GdpytImage):
                baseline_img = baseline

                for particle in baseline_img.particles:
                    baseline_locations.append(pd.DataFrame({'x': particle.location[0], 'y': particle.location[1]},
                                                           index=[particle.id]))
                skip_first_img = False
            elif isinstance(baseline, str):
                baseline_img = self.images[baseline]
                for particle in baseline_img.particles:
                    baseline_locations.append(pd.DataFrame({'x': particle.location[0], 'y': particle.location[1]},
                                                           index=[particle.id]))
                skip_first_img = False
            else:
                raise TypeError("Invalid type for baseline")

        # If no baseline is given, the particle IDs are assigned based on the IDs and location of the particles in the
        # baseline_img or else the first image
        else:
            index = 0
            if baseline_img is not None:
                index = self._files.index(baseline_img)
            baseline_img = self._files[index]
            baseline_img = self.images[baseline_img]

            for particle in baseline_img.particles:
                baseline_locations.append(pd.DataFrame({'x': particle.location[0], 'y': particle.location[1]},
                                                       index=[particle.id]))

            skip_first_img = True

        if len(baseline_locations) == 0:
            baseline_locations = pd.DataFrame()
            next_id = None
        else:
            baseline_locations = pd.concat(baseline_locations).sort_index()
            baseline_locations['x'] = baseline_locations['x'] + uv[0][0]
            baseline_locations['y'] = baseline_locations['y'] + uv[0][1]
            # The next particle that can't be matched to a particle in the baseline gets this id
            next_id = len(baseline_locations)

        for i, file in enumerate(self._files):
            if (i == 0) and skip_first_img:
                continue
            image = self.images[file]
            # Convert to list because ordering is important
            particles = [particle for particle in image.particles]
            locations = [list(p.location) for p in particles]

            if len(locations) == 0:
                continue
            if baseline_locations.empty:
                dfs = [pd.DataFrame({'x': p.location[0], 'y': p.location[1]}, index=[p.id]) for p in particles]
                baseline_locations = pd.concat(dfs)
                next_id = len(baseline_locations)
                continue

            # TODO: set NearestNeighbors(n_neighbors=2) and if a particle center is within an "overlap_range" then,
            # merge those contours and maintain that particle's ID. Do this same function for the other merged particle
            # (i.e. maintain it's ID but merge the contours).

            # NearestNeighbors(x+u,y+v): previous location (x,y) + displacement guess (u,v)
            nneigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(baseline_locations.values)
            # NOTE: the displacement guess (u,v) could incorporate a "time" variable (image number or time data)
            # such that [u_i,v_i] = [u(t), v(t)] in order to better match time-dependent or periodic displacements.

            distances, indices = nneigh.kneighbors(np.array(locations))

            remove_p_not_in_calib = []
            for distance, idx, particle in zip(distances, indices, particles):
                # If the particle is close enough to a particle of the baseline, give that particle the same ID as the
                # particle in the baseline
                if distance < self._same_id_threshold:

                    particle.set_id(baseline_locations.index.values[idx.squeeze()])

                    # assign the baseline coordinates (x,y) to the matched particle coordinates (x,y)
                    baseline_locations.loc[particle.id, ('x', 'y')] = (particle.location[0], particle.location[1])
                    # the baseline is essentially the "current" location for that particle ID and after each
                    # successful identification, we update the "current" location of that particle ID.

                else:
                    # If the particle is not in the baseline, we may remove it via two methods:
                    #   1. if the baseline is a CalibrationSet, as including will significantly reduce accuracy.
                    #   2. if we designate a "hard baseline" where we don't want to add new particles.

                    # filter if not in CalibrationSet baseline:
                    if isinstance(baseline, GdpytCalibrationSet):
                        remove_p_not_in_calib.append(particle)
                        logger.warning(
                            "Removed particle {} at location {} b/c not in Calibration Set baseline".format(particle.id,
                                                                                                            particle.location))
                        continue

                    # filter if not in "hard baseline":
                    elif self._hard_baseline is True:
                        remove_p_not_in_calib.append(particle)
                        logger.warning(
                            "Removed particle {} at location {} b/c not in hard baseline".format(particle.id,
                                                                                                 particle.location))
                        continue

                    # else, assign it a new, non-existent id and add it to the baseline for subsequent images
                    logger.warning("File {}: New IDs: {}".format(file, next_id))
                    particle.set_id(next_id)
                    assert (next_id not in baseline_locations.index)
                    baseline_locations = baseline_locations.append(
                        pd.DataFrame({'x': particle.location[0], 'y': particle.location[1]},
                                     index=[particle.id]))
                    next_id += 1
            for p in remove_p_not_in_calib:
                logger.warning("Removing particle particle {}".format(p.id))
                image.particles.remove(p)
            # The nearest neighbor mapping creates particles with duplicate IDs under some circumstances
            # These need to be merged to one giant particle
            image.merge_duplicate_particles()

    def calculate_particle_to_particle_spacing(self, max_n_neighbors=10, theoretical_diameter_params=None):
        """
        """
        if theoretical_diameter_params is not None:
            mag_eff = theoretical_diameter_params[0]
            zf = theoretical_diameter_params[1]
            c1 = theoretical_diameter_params[2]
            c2 = theoretical_diameter_params[3]

            def theoretical_diameter_function(z):
                return mag_eff * np.sqrt(c1 ** 2 * (z - zf) ** 2 + c2 ** 2)

        for img in self.images.values():

            particles = [particle for particle in img.particles]
            locations = np.array([list(p.location) for p in particles])

            if len(locations) < 2:
                continue
            elif len(locations) < max_n_neighbors + 1:
                temp_max_n_neighbors = len(locations)
            else:
                temp_max_n_neighbors = max_n_neighbors + 1

            nneigh = NearestNeighbors(n_neighbors=temp_max_n_neighbors, algorithm='ball_tree').fit(locations)
            distances, indices = nneigh.kneighbors(locations)

            distance_to_others = distances[:, 1:]

            for distance, p in zip(distance_to_others, particles):

                if p.gauss_dia_x is None:
                    continue

                if theoretical_diameter_params is not None:
                    mean_diameter = theoretical_diameter_function(p.z)
                else:
                    mean_diameter = np.mean([p.gauss_dia_x, p.gauss_dia_y])

                mean_dx_all = np.mean(distance)
                min_dx_all = np.min(distance)
                percent_dx_diameter = mean_diameter / min_dx_all
                num_dx_all = temp_max_n_neighbors - 1

                overlapping_dists = distance[distance < mean_diameter]
                if len(overlapping_dists) == 0:
                    mean_dxo = min_dx_all
                    num_dxo = 0
                elif len(overlapping_dists) == 1:
                    mean_dxo = overlapping_dists[0]
                    num_dxo = 1
                else:
                    mean_dxo = np.mean(overlapping_dists)
                    num_dxo = len(overlapping_dists)

                pid_to_particle_spacing = [mean_dx_all, min_dx_all, num_dx_all, mean_dxo, num_dxo, percent_dx_diameter]
                p.set_particle_to_particle_spacing(ptps=pid_to_particle_spacing)

    def set_true_z(self, image_to_z):
        """
        Defines the true_z value for every image in the collection (which then sets the true z value for every
        particle in that image). This is necessary when performing a meta-analysis on a test collection.

        Parameters
        ----------
        image_to_z: a dictionary containing the filename and z-height of every image in the calibration stack. Note,
        that this function should only be called when performing a meta-analysis on a calibration stack (or in other
        cases where the the true z value is known for all images).
        """
        for image in self.images.values():
            if image.filename not in image_to_z.keys():
                raise ValueError("No z coordinate specified for image {}")
            else:
                image.set_true_z(image_to_z[image.filename])

    def is_infered(self):
        """ Checks if the z coordinate has been inferred for the images in this collection. Only returns true if that's
        true for all the images. """
        return all([image.is_infered() for image in self.images.values()])

    def infer_z(self, calib_set, infer_sub_image=True):
        """ Returns an object whose methods implement a different function to infer the z coordinate of each image.
        For example: collection.infer_z.znccorr() assigns z coordinates based on zero-normalized cross correlation
        similarity
        :param calib_set:   GdpytCalibrationSet object
        """
        assert isinstance(calib_set, GdpytCalibrationSet)

        return calib_set.infer_z(self, infer_sub_image=infer_sub_image)

    def update_particles_in_images(self):
        # for each particle ID, attach a list of all frames that the particle appears in
        for pid in self.particle_ids:
            pid_frames = self.spct_stats[self.spct_stats['id'] == pid].frame.values

            for img in self.images.values():
                [p.add_particle_in_image(pid_frames) for p in img.particles if p.id == pid]

    def get_particles_in_images(self, particle_id=None):
        # get particles in every image
        coords = []
        for img in self.images.values():
            [coords.append([p.id, p.z, p.z_true]) for p in img.particles]

        df = pd.DataFrame(data=coords, columns=['id', 'z', 'true_z'])

        if particle_id is None:
            return df
        else:
            return df.loc[df['id'].isin(particle_id)]

    def get_unique_particle_ids(self):
        df = self.get_particles_in_images()
        unique_ids = df.id.unique()
        self._particle_ids = unique_ids
        return unique_ids

    def update_processing(self, processing_specs, erase_old=False):
        if not isinstance(processing_specs, dict):
            raise TypeError("Processing specifications must be specified as a dictionary")
        if not erase_old:
            self._processing_specs.update(processing_specs)
        else:
            self._processing_specs = processing_specs

        self.filter_images()
        self.identify_particles()

    def update_particle_size_range(self, min=None, max=None):
        if min is not None:
            self._min_particle_size = min
        if max is not None:
            self._max_particle_size = max
        self.identify_particles()

    def find_particle_in_focus_z(self, use_true_z=False, use_peak_int=True):
        """

        !!! DEPRECATED DEPRECATED DEPRECATED!!!

        Find the z-coordinate of minimum area for each particle in the collection and set the particle's in_focus_plane
        attribute.

        Steps:
            1. get image collection particle ID's.
            2. for each particle ID, loop through all of the images that the particle appears and append its
            z-coordinate and area.
                NOTE: for test collections with ground_truth, the true_z-coordinate is used because the particle's
                 z-coordinate hasn't been inferred and using the inferred z-coordinate would also introduce errors.
            3. get the index where the minimum particle area occurs.
            4.

            # loop through particle ID list
        for pid in particle_ids:
            for img in self.images.values():
                for particle in img.particles:
                    if particle.id != pid:
                        continue
        """
        # get list of all particle ID's in collection
        particle_ids = self.particle_ids.tolist()

        # loop through particle ID list
        for pid in particle_ids:

            # initialize lists for particle z-coordinate and area
            zs = []
            areas = []
            peak_ints = []

            for img in self.images.values():  # for every image
                for particle in img.particles:  # for every particle

                    # append the in-focus data to lists
                    if particle.id == pid:

                        # get peak intensity
                        peak_ints.append(particle.peak_intensity)

                        # get particle z-coordinate
                        if use_true_z:
                            zs.append(particle.z_true)
                        else:
                            zs.append(particle.z)

                        # get particle area
                        areas.append(particle.area)

            # get min/max z-coordinates
            zl, zh = (min(zs), max(zs))

            # get index of maximum intensity and minimum area
            int_max_index = int(np.argmax(peak_ints))
            amin_index = int(np.argmin(areas))

            # create lower and upper bounds (+/- 5 images) - Note, 5 images were chosen to smooth out the contour noise
            # UPDATED 9/25: to 1 images
            if self.image_collection_type == 'calibration':
                fit_pad = 1
            else:
                fit_pad = 2

            # get bounds for max intensity and minimum area
            lower_intdex = int(np.round(int_max_index - fit_pad, 0))
            upper_intdex = int(np.round(int_max_index + fit_pad, 0))
            lower_index = int(np.round(amin_index - fit_pad, 0))
            upper_index = int(np.round(amin_index + fit_pad, 0))

            # ensure lower and upper bounds don't exceed data space
            if lower_intdex < 0:
                lower_intdex = 0
                upper_intdex = 1 + fit_pad
            if upper_intdex > len(peak_ints) - 1:
                upper_intdex = len(peak_ints) - 1
                lower_intdex = len(peak_ints) - 2 - fit_pad
            if lower_index < 0:
                lower_index = 0
                upper_index = 1 + fit_pad
            if upper_index > len(areas) - 1:
                upper_index = len(areas) - 1
                lower_index = len(areas) - 2 - fit_pad

            # if there are at least three points, perform parabolic interpolation
            if len(zs) > 3 and len(areas) > 3 and len(peak_ints) > 3:

                # fit a parabola
                popt_int, pcov_int = curve_fit(parabola, zs[lower_intdex:upper_intdex + 1],
                                               peak_ints[lower_intdex:upper_intdex + 1])
                popt, pcov = curve_fit(parabola, zs[lower_index:upper_index + 1], areas[lower_index:upper_index + 1])

                # create interpolation space and get resulting parabolic curve
                z_local_int = np.linspace(zs[lower_intdex], zs[upper_intdex], 50)
                peak_ints_interp = parabola(z_local_int, *popt_int)
                z_local = np.linspace(zs[lower_index], zs[upper_index], 50)
                areas_interp = parabola(z_local, *popt)

                # find the z-coordinate where the interpolated area is minimized
                z_zero_peak_int = z_local_int[np.argmax(peak_ints_interp)]
                peak_ints_zero = np.max(peak_ints_interp)

                # find the z-coordinate where the interpolated area is minimized
                z_zero = z_local[np.argmin(areas_interp)]
                areas_zero = np.min(areas_interp)

                """if self.image_collection_type == 'calibration' and len(particle_ids) == 1:
                    fig, ax = plt.subplots()
                    ax.scatter(zs[lower_index:upper_index + 1], areas[lower_index:upper_index + 1], color='tab:blue',
                               alpha=0.75, label='particle area')
                    ax.plot(z_local, areas_interp, color='black', alpha=0.5, label='interpolated')
                    ax.scatter(z_zero, areas_zero, s=50, color='red', marker='.', label='min')
                    ax.set_xlabel('z')
                    ax.set_ylabel('area')
                    ax.grid(alpha=0.25)
                    ax.set_title('Find particles in focus z')
                    savedir = '/Users/mackenzie/Desktop/dumpfigures'
                    savename = 'particle_interpolated_z_of_min_area.png'
                    plt.savefig(join(savedir, savename))"""

            # if less than three points, get the minimum of the areas
            else:
                z_zero_peak_int = zs[np.argmax(peak_ints)]
                peak_ints_zero = np.max(peak_ints)
                z_zero = zs[np.argmin(areas)]
                areas_zero = np.min(areas)

            # round the z-coordinate and area to a reasonable value
            z_zero_peak_int = np.round(z_zero_peak_int, 3)
            peak_ints_zero = np.round(peak_ints_zero, 1)
            z_zero = np.round(z_zero, 3)
            areas_zero = np.round(areas_zero, 1)

            # Set in-focus plane z-coordinate for all particles:
            for imgg in self.images.values():  # for every image
                for p_set in imgg.particles:
                    if p_set.id == pid:
                        if use_peak_int:
                            p_set.set_in_focus_z(z_zero_peak_int)
                        else:
                            p_set.set_in_focus_z(z_zero)
                        p_set.set_in_focus_area(areas_zero)
                        p_set.set_in_focus_intensity(peak_ints_zero)

    def find_collection_in_focus_z(self):
        """
        !!! DEPRECATED DEPRECATED DEPRECATED!!!

        Find the mean particle.in_focus_z value for the collection and store the in-focus coordinates for all particles.
        """

        # get list of all particle ID's in collection
        particle_ids = self.particle_ids.tolist()

        # initialize lists for particle: ID, x,y-coordinates, and interpolated in_focus z-coordinate and area.
        ids = []
        xs = []
        ys = []
        zs = []
        areas = []

        # loop through particle ID list
        while len(particle_ids) > 0:
            for img in self.images.values():
                for particle in img.particles:

                    # append the in-focus data to lists
                    if particle.id in particle_ids:
                        ids.append(particle.id)
                        xs.append(particle.location[0])
                        ys.append(particle.location[1])
                        zs.append(particle.in_focus_z)
                        areas.append(particle.in_focus_area)

                        # remove the particle ID from the list and continue seeking particles in particle list
                        particle_ids.remove(particle.id)

        # stack lists into array
        in_focus_coords = np.stack((ids, xs, ys, zs, areas)).T

        # store in-focus data as DataFrame
        df_in_focus = pd.DataFrame(data=in_focus_coords, columns=['id', 'x', 'y', 'z', 'area'], dtype=float)
        df_in_focus = df_in_focus.sort_values(by='z')
        self.in_focus_coords = df_in_focus

        if len(df_in_focus.id.unique()) < 15:
            # get the minimum area and in_focus_z by taking the median of the filtered dataset
            in_focus_zs = df_in_focus.z
            in_focus_areas = df_in_focus.area
        else:
            # remove outliers by removing the outer 20th percentile in the z-coordinate array
            in_focus_zs = df_in_focus.z[(df_in_focus['z'] > df_in_focus.z.quantile(0.20)) &
                                        (df_in_focus['z'] < df_in_focus.z.quantile(0.80))]
            in_focus_areas = df_in_focus.area[(df_in_focus['z'] > df_in_focus.z.quantile(0.20)) &
                                              (df_in_focus['z'] < df_in_focus.z.quantile(0.80))]

        # sort the filtered data
        in_focus_zs = in_focus_zs.sort_values()
        in_focus_areas = in_focus_areas.sort_values()

        # get the minimum area and in_focus_z by taking the median of the filtered dataset
        in_focus_z = np.median(in_focus_zs)
        in_focus_area = np.median(in_focus_areas)

        # store the mean interpolated in_focus_z coordinate as the collection's in_focus_z coordinate.
        self.in_focus_z = in_focus_z
        self._in_focus_area = in_focus_area

    def find_collection_z_of_min_area(self):
        """
        !!! DEPRECATED DEPRECATED DEPRECATED!!!

        Find the z-coordinate of the minimum area for all particles in a collection

        NOTE:
            * uses the particle's true_z coordinate and thus is currently only applicable to ground_truth datasets.

        Usages:
            * To determine the "focal plane" for collections with unknown or random particle z-coordiantes.
        """

        # get list of all particle ID's in collection
        particle_ids = self.particle_ids.tolist()

        # initialize lists for particle: ID, x,y-coordinates, and interpolated in_focus z-coordinate and area.
        img_ids = []
        ids = []
        xs = []
        ys = []
        zss = []
        areass = []

        # loop through particle ID list
        for img in self.images.values():
            img_id = float(img.filename.split(self.file_basestring)[-1].split('.tif')[0])
            for particle in img.particles:
                img_ids.append(img_id)
                ids.append(particle.id)
                xs.append(particle.location[0])
                ys.append(particle.location[1])
                zss.append(particle.z_true)
                areass.append(particle.area)

        # stack lists into array
        coords = np.stack((img_ids, ids, xs, ys, zss, areass)).T

        # store data as DataFrame
        df = pd.DataFrame(data=coords, columns=['img_id', 'id', 'x', 'y', 'z', 'area'], dtype=float)
        df = df.groupby(by='z').mean().reset_index()

        # remove outliers by removing the outer 2 percentile in the z-coordinate array
        # df = df[(df['area'] > df.area.quantile(0.00025))]

        zs = df.z.to_numpy()
        areas = df.area.to_numpy()

        # get index of minimum area
        amin_index = int(np.argmin(areas))

        # create lower and upper bounds (+/- 2 images)
        fit_pad = np.max([1, int(len(zs) * 0.0125)])

        lower_index = int(np.round(amin_index - fit_pad, 0))
        upper_index = int(np.round(amin_index + fit_pad, 0))

        # ensure lower and upper bounds don't exceed data space
        if lower_index < 0:
            lower_index = 0
            upper_index = 1 + fit_pad
        if upper_index > len(areas) - 1:
            upper_index = len(areas) - 1
            lower_index = len(areas) - 2 - fit_pad

        # if there are at least three points, perform parabolic interpolation
        if len(zs) > 3 and len(areas) > 3:

            # fit a parabola
            popt, pcov = curve_fit(parabola, zs[lower_index:upper_index + 1], areas[lower_index:upper_index + 1])

            # create interpolation space and get resulting parabolic curve
            z_local = np.linspace(zs[lower_index], zs[upper_index], 50)
            areas_interp = parabola(z_local, *popt)

            # find the z-coordinate where the interpolated area is minimized
            z_zero = z_local[np.argmin(areas_interp)]
            areas_zero = np.min(areas_interp)

        # if less than three points, get the minimum of the areas
        else:
            z_zero = zs[np.argmin(areas)]
            areas_zero = np.min(areas)

        # round the z-coordinate and area to a reasonable value
        in_focus_z = np.round(z_zero, 3)
        in_focus_area = np.round(areas_zero, 1)

        # store the mean interpolated in_focus_z coordinate as the collection's in_focus_z coordinate.
        self.in_focus_z = in_focus_z
        self._in_focus_area = in_focus_area

    def structure_spct_stats(self, idpt=False):
        """
        Structure all particle data for all images
        """
        data = []

        for img in self.images.values():

            img_size_y, img_size_x = np.shape(img.raw)
            img_xc, img_yc = img_size_x / 2, img_size_y / 2

            for p in img.particles:

                if p.gauss_dia_x is None:
                    continue

                # unique identifier
                frame = p.frame
                id = p.id

                # location
                locz_true = p.z_true
                locz = p.z
                locx = p.location[0]
                locy = p.location[1]
                dist_r = np.sqrt((locx - img_xc) ** 2 + (locy - img_yc) ** 2)

                # image
                peak_int = p.peak_intensity
                mean_int = p.mean_signal
                bkg_mean = p.mean_background
                bkg_noise = p.std_background
                snr = p.snr
                nsv = p.int_var_sq_norm
                nsv_signal = p.int_var_sq_norm_signal

                # shape
                area_contour = p.area
                diameter_contour = p.diameter
                solidity = p.solidity
                thinness_ratio = p.thinness_ratio

                # spacing
                mean_dx = p.mean_dx
                min_dx = p.min_dx
                num_dx = p.num_dx

                # overlap
                mean_dxo = p.mean_dxo
                num_dxo = p.num_dxo
                percent_dx_diameter = p.percent_dx_diameter

                if idpt is True:
                    datum = [
                        frame, id,
                        locz_true, locz, locx, locy, dist_r,
                        peak_int, mean_int, bkg_mean, bkg_noise, snr, nsv, nsv_signal,
                        area_contour, diameter_contour, solidity, thinness_ratio,
                        mean_dx, min_dx, num_dx,
                        mean_dxo, num_dxo, percent_dx_diameter,
                    ]

                    data.append(datum)

                else:
                    # Gaussian image
                    gauss_A = p.gauss_A
                    gauss_xc = p.gauss_xc
                    gauss_yc = p.gauss_yc

                    gauss_dia_x = p.gauss_dia_x
                    gauss_dia_y = p.gauss_dia_y
                    diameter = np.mean([gauss_dia_x, gauss_dia_y])
                    gauss_sigma_x = p.gauss_sigma_x
                    gauss_sigma_y = p.gauss_sigma_y

                    # astigmatism
                    diax_diay = gauss_dia_x / gauss_dia_y
                    ax_ay = gauss_sigma_x / gauss_sigma_y

                    datum = [
                        frame, id,
                        locz_true, locz, locx, locy, dist_r,
                        peak_int, mean_int, bkg_mean, bkg_noise, snr, nsv, nsv_signal,
                        area_contour, diameter_contour, solidity, thinness_ratio,
                        gauss_A, gauss_xc, gauss_yc,
                        gauss_dia_x, gauss_dia_y, diameter, diax_diay,
                        gauss_sigma_x, gauss_sigma_y, ax_ay,
                        mean_dx, min_dx, num_dx,
                        mean_dxo, num_dxo, percent_dx_diameter,
                    ]

                    data.append(datum)

        if idpt is True:
            columns = ['frame', 'id', 'z_true', 'z', 'x', 'y', 'r', 'peak_int', 'mean_int', 'bkg_mean', 'bkg_noise',
                       'snr',
                       'nsv', 'nsv_signal', 'contour_area', 'contour_diameter', 'solidity', 'thinness_ratio',
                       'mean_dx', 'min_dx', 'num_dx',
                       'mean_dxo', 'num_dxo', 'percent_dx_diameter']
        else:

            columns = ['frame', 'id', 'z_true', 'z', 'x', 'y', 'r', 'peak_int', 'mean_int', 'bkg_mean', 'bkg_noise',
                       'snr',
                       'nsv', 'nsv_signal', 'contour_area', 'contour_diameter', 'solidity', 'thinness_ratio',
                       'gauss_A', 'gauss_xc', 'gauss_yc', 'gauss_dia_x', 'gauss_dia_y', 'gauss_diameter',
                       'gauss_dia_x_y',
                       'gauss_sigma_x', 'gauss_sigma_y', 'gauss_sigma_x_y', 'mean_dx', 'min_dx', 'num_dx',
                       'mean_dxo', 'num_dxo', 'percent_dx_diameter']

        df = pd.DataFrame(data, columns=columns)

        self.spct_stats = df

    def calculate_spct_stats(self, param_zf, filter_percent_frames=0.5):
        """
        Processing Pipeline:
            1.
            2. for each particle:
                2.1 Fit intensity profile
                    2.1.1 in-focus via intensity profile maximum
                    2.1.2 effective numerical aperture (intensity)
                2.2 Fit NSV + NSV (signal)
                    2.2.1. in-focus via NSV
                    2.2.2 in-focus via NSV signal
                2.3 Fit diameter profile
                    2.3.1 in-focus via diameter profile minimum
                    2.3.2 effective numerical aperture (area)
            3. average per-particle analyses
                3.1 average peak intensity profile
                    3.1.1. average peak intensity +/- std
                    3.1.2 average background mean
                    3.1.3 average background std
                3.2 average diameter profile
                    3.2.1 average diameter +/- std
            4. particle distribution analyses
                4.1 particle-to-particle spacing
                    4.1.1 per-p2p spacing (<max diameter)
                    4.1.2 mean p2p spacing
                4.2 percent diameter overlap
                    4.2.1 per-particle percent diameter overlap

        Returns
        -------

        """

        mag_eff = self.optics.effective_magnification

        def particle_diameter_function(z, zf, c1, c2):
            return mag_eff * np.sqrt(c1 ** 2 * (z - zf) ** 2 + c2 ** 2)

        df = self.spct_stats
        total_num_frames = len(df.frame.unique())
        pids = df.id.unique()

        data = []
        for pid in pids:
            # 0.0 DATAFRAME OF PARTICLE ID
            dfpid = df[df['id'] == pid]

            dfpid = dfpid.dropna()

            dfpid = dfpid.sort_values('z_true').reset_index()
            num_frames = len(dfpid.z_true)

            if num_frames < filter_percent_frames * total_num_frames:
                continue

            # 0.0 x-y location
            x = dfpid.x.mean()
            y = dfpid.y.mean()
            r = dfpid.r.mean()

            # 1.0 IN-FOCUS Z
            z = dfpid.z_true.to_numpy()

            # in-focus z (peak intensity, normalized squared variance (nsv) of template pixels, nsv of signal pixels)
            intensity = dfpid.peak_int.to_numpy()
            nsv = dfpid.nsv.to_numpy()
            nsv_signal = dfpid.nsv_signal.to_numpy()
            ylabels = ['intensity', 'nsv', 'nsv signal']
            z_maxs = gaussian.calculate_maximum_of_fitted_gaussian_1d(x=z,
                                                                      y=[intensity, nsv, nsv_signal],
                                                                      normalize=True,
                                                                      show_plot=False,
                                                                      ylabels=ylabels)
            zf_peak_int, zf_nsv, zf_nsv_signal = z_maxs[0], z_maxs[1], z_maxs[2]

            # in-focus z (diameter)
            dia_x = dfpid.gauss_dia_x
            dia_y = dfpid.gauss_dia_y

            zf_c1_c2s, dia_zfs, dia_zmins, dia_zmaxs = \
                gaussian.calculate_minimum_of_fitted_gaussian_diameter(x=z,
                                                                       y=[dia_x, dia_y],
                                                                       fit_function=particle_diameter_function,
                                                                       guess_x0=zf_nsv_signal,
                                                                       show_plot=False)

            zf_from_dia_x, zf_from_dia_y = zf_c1_c2s[0, 0], zf_c1_c2s[1, 0]
            zf_from_dia_mean = np.mean([zf_from_dia_x, zf_from_dia_y])
            c1 = np.mean(zf_c1_c2s[:, 1])
            c2 = np.mean(zf_c1_c2s[:, 2])

            zf_dia_x, zf_dia_y = dia_zfs[0], dia_zfs[1]
            zmin_dia_x, zmin_dia_y = dia_zmins[0], dia_zmins[1]
            zmax_dia_x, zmax_dia_y = dia_zmaxs[0], dia_zmaxs[1]

            # analyze in-focus stats
            zf_index = np.argmin(np.abs(z - zf_nsv))
            zf_nearest_calib = dfpid.iloc[zf_index].z_true
            at_zf_peak_int = dfpid.iloc[zf_index].peak_int
            at_zf_snr = dfpid.iloc[zf_index].snr
            at_zf_min_dx = dfpid.iloc[zf_index].min_dx
            at_zf_percent_dx_diameter = dfpid.iloc[zf_index].percent_dx_diameter
            at_zf_dia_x_y = zf_dia_x / zf_dia_y
            mean_min_dia = np.mean([zf_dia_x, zf_dia_y])

            # analyze z-min stats
            at_zmin_peak_int = dfpid.iloc[0].peak_int
            at_zmin_snr = dfpid.iloc[0].snr
            at_zmin_min_dx = dfpid.iloc[0].min_dx
            at_zmin_percent_dx_diameter = dfpid.iloc[0].percent_dx_diameter
            at_zmin_dia_x_y = zmin_dia_x / zmin_dia_y
            mean_zmin_dia = np.mean([zmin_dia_x, zmin_dia_y])

            # analyze z-max stats
            at_zmax_peak_int = dfpid.iloc[-1].peak_int
            at_zmax_snr = dfpid.iloc[-1].snr
            at_zmax_min_dx = dfpid.iloc[-1].min_dx
            at_zmax_percent_dx_diameter = dfpid.iloc[-1].percent_dx_diameter
            at_zmax_dia_x_y = zmax_dia_x / zmax_dia_y
            mean_zmax_dia = np.mean([zmax_dia_x, zmax_dia_y])

            # analyze defocus stats (from in-focus, to z-min/z-max defocus)
            k3_zmin = mean_min_dia / mean_zmin_dia
            k3_zmax = mean_min_dia / mean_zmax_dia

            datum = [pid, x, y, r, num_frames,
                     zf_nearest_calib, zf_peak_int, zf_nsv, zf_nsv_signal, zf_from_dia_mean,
                     at_zf_peak_int, at_zmin_peak_int, at_zmax_peak_int,
                     at_zf_snr, at_zmin_snr, at_zmax_snr,
                     mean_min_dia, mean_zmin_dia, mean_zmax_dia, k3_zmin, k3_zmax,
                     at_zf_dia_x_y, at_zmin_dia_x_y, at_zmax_dia_x_y,
                     at_zf_min_dx, at_zmin_min_dx, at_zmax_min_dx,
                     at_zf_percent_dx_diameter, at_zmin_percent_dx_diameter, at_zmax_percent_dx_diameter,
                     c1, c2,
                     ]

            data.append(datum)

            # lastly, set particle values
            if param_zf == 'zf_nsv':
                pid_zf = zf_nsv
            elif param_zf == 'zf_peak_int':
                pid_zf = zf_peak_int

            # lastly, set particle values
            for img in self.images.values():
                for p in img.particles:
                    if p.id == pid:
                        p.set_in_focus_z(pid_zf)
                        p.set_in_focus_area(np.pi * mean_min_dia ** 2 / 4)
                        p.set_in_focus_intensity(at_zf_peak_int)

        columns = ['id', 'x', 'y', 'r', 'num_frames',
                   'zf_nearest_calib', 'zf_from_peak_int', 'zf_from_nsv', 'zf_from_nsv_signal', 'zf_from_dia',
                   'zf_peak_int', 'zmin_peak_int', 'zmax_peak_int',
                   'zf_snr', 'zmin_snr', 'zmax_snr',
                   'zf_dia', 'zmin_dia', 'zmax_dia', 'k3_zmin', 'k3_zmax',
                   'zf_dia_x_y', 'zmin_dia_x_y', 'zmax_dia_x_y',
                   'zf_min_dx', 'zmin_min_dx', 'zmax_min_dx',
                   'zf_percent_dx_diameter', 'zmin_percent_dx_diameter', 'zmax_percent_dx_diameter',
                   'c1', 'c2',
                   ]
        df = pd.DataFrame(data, columns=columns)

        self.spct_particle_defocus_stats = df
        self.population_statistics_spct_stats()

    def calculate_idpt_stats_gaussian(self, param_zf, filter_percent_frames=0.5):

        df = self.spct_stats
        total_num_frames = len(df.frame.unique())
        pids = df.id.unique()

        data = []
        for pid in pids:
            # 0.0 DATAFRAME OF PARTICLE ID
            dfpid = df[df['id'] == pid]

            dfpid = dfpid.dropna()

            dfpid = dfpid.sort_values('z_true').reset_index()
            num_frames = len(dfpid.z_true)

            if num_frames < filter_percent_frames * total_num_frames:
                continue

            # 0.0 x-y location
            x = dfpid.x.mean()
            y = dfpid.y.mean()
            r = dfpid.r.mean()

            # 1.0 IN-FOCUS Z
            z = dfpid.z_true.to_numpy()

            # in-focus z (peak intensity, normalized squared variance (nsv) of template pixels, nsv of signal pixels)
            intensity = dfpid.peak_int.to_numpy()
            amplitude = dfpid.gauss_A.to_numpy()
            nsv = dfpid.nsv.to_numpy()
            nsv_signal = dfpid.nsv_signal.to_numpy()
            ylabels = ['intensity', 'amplitude', 'nsv', 'nsv signal']
            z_maxs = gaussian.calculate_maximum_of_fitted_gaussian_1d(x=z,
                                                                      y=[intensity, amplitude, nsv, nsv_signal],
                                                                      normalize=True,
                                                                      show_plot=False,
                                                                      ylabels=ylabels)
            print("pid {}: {}".format(pid, z_maxs))
            zf_peak_int, zf_gauss_A, zf_nsv, zf_nsv_signal = z_maxs[0], z_maxs[1], z_maxs[2], z_maxs[3]

            if zf_peak_int == np.nan:
                zf_peak_int = np.mean(z_maxs)
            if zf_gauss_A == np.nan:
                zf_gauss_A = np.mean(z_maxs)
            if zf_nsv == np.nan:
                zf_nsv = np.mean(z_maxs)
            if zf_nsv_signal == np.nan:
                zf_nsv_signal = np.mean(z_maxs)

            # analyze in-focus stats
            zf_index = np.argmin(np.abs(z - zf_nsv))
            zf_nearest_calib = dfpid.iloc[zf_index].z_true
            at_zf_peak_int = dfpid.iloc[zf_index].peak_int
            at_zf_gauss_A = dfpid.iloc[zf_index].gauss_A
            at_zf_snr = dfpid.iloc[zf_index].snr
            at_zf_min_dx = dfpid.iloc[zf_index].min_dx
            at_zf_percent_dx_diameter = dfpid.iloc[zf_index].percent_dx_diameter

            # analyze z-min stats
            at_zmin_peak_int = dfpid.iloc[0].peak_int
            at_zmin_snr = dfpid.iloc[0].snr
            at_zmin_min_dx = dfpid.iloc[0].min_dx
            at_zmin_percent_dx_diameter = dfpid.iloc[0].percent_dx_diameter

            # analyze z-max stats
            at_zmax_peak_int = dfpid.iloc[-1].peak_int
            at_zmax_snr = dfpid.iloc[-1].snr
            at_zmax_min_dx = dfpid.iloc[-1].min_dx
            at_zmax_percent_dx_diameter = dfpid.iloc[-1].percent_dx_diameter

            datum = [pid, x, y, r, num_frames,
                     zf_nearest_calib, zf_peak_int, zf_gauss_A, zf_nsv, zf_nsv_signal,
                     at_zf_peak_int, at_zf_gauss_A, at_zmin_peak_int, at_zmax_peak_int,
                     at_zf_snr, at_zmin_snr, at_zmax_snr,
                     at_zf_min_dx, at_zmin_min_dx, at_zmax_min_dx,
                     at_zf_percent_dx_diameter, at_zmin_percent_dx_diameter, at_zmax_percent_dx_diameter,
                     ]

            data.append(datum)

            # lastly, set particle values
            if param_zf == 'gauss_A':
                pid_zf = zf_gauss_A
                pid_zf_intensity = at_zf_gauss_A

            for img in self.images.values():
                for p in img.particles:
                    if p.id == pid:
                        p.set_in_focus_z(pid_zf)
                        p.set_in_focus_intensity(pid_zf_intensity)

        columns = ['id', 'x', 'y', 'r', 'num_frames',
                   'zf_nearest_calib', 'zf_from_peak_int', 'zf_from_gauss_A', 'zf_from_nsv', 'zf_from_nsv_signal',
                   'zf_peak_int', 'zf_gauss_A', 'zmin_peak_int', 'zmax_peak_int',
                   'zf_snr', 'zmin_snr', 'zmax_snr',
                   'zf_min_dx', 'zmin_min_dx', 'zmax_min_dx',
                   'zf_percent_dx_diameter', 'zmin_percent_dx_diameter', 'zmax_percent_dx_diameter',
                   ]
        df = pd.DataFrame(data, columns=columns)

        self.spct_particle_defocus_stats = df
        self.population_statistics_spct_stats()

    def calculate_idpt_stats(self, param_zf, filter_percent_frames=0.5):

        df = self.spct_stats
        total_num_frames = len(df.frame.unique())
        pids = df.id.unique()

        data = []
        for pid in pids:
            # 0.0 DATAFRAME OF PARTICLE ID
            dfpid = df[df['id'] == pid]
            dfpid = dfpid.sort_values('z_true').reset_index()
            num_frames = len(dfpid.z_true)

            if num_frames < filter_percent_frames * total_num_frames:
                continue

            # 0.0 x-y location
            x = dfpid.x.mean()
            y = dfpid.y.mean()
            r = dfpid.r.mean()

            # 1.0 IN-FOCUS Z
            z = dfpid.z_true.to_numpy()

            # in-focus z (peak intensity, normalized squared variance (nsv) of template pixels, nsv of signal pixels)
            intensity = dfpid.peak_int.to_numpy()
            nsv = dfpid.nsv.to_numpy()
            nsv_signal = dfpid.nsv_signal.to_numpy()
            z_maxs = gaussian.calculate_maximum_of_fitted_gaussian_1d(x=z,
                                                                      y=[intensity, nsv, nsv_signal],
                                                                      normalize=True,
                                                                      show_plot=False)
            zf_peak_int, zf_nsv, zf_nsv_signal = z_maxs[0], z_maxs[1], z_maxs[2]

            # analyze in-focus stats
            zf_index = np.argmin(np.abs(z - zf_nsv))
            zf_nearest_calib = dfpid.iloc[zf_index].z_true
            at_zf_peak_int = dfpid.iloc[zf_index].peak_int
            at_zf_snr = dfpid.iloc[zf_index].snr
            at_zf_min_dx = dfpid.iloc[zf_index].min_dx
            at_zf_percent_dx_diameter = dfpid.iloc[zf_index].percent_dx_diameter

            # analyze z-min stats
            at_zmin_peak_int = dfpid.iloc[0].peak_int
            at_zmin_snr = dfpid.iloc[0].snr
            at_zmin_min_dx = dfpid.iloc[0].min_dx
            at_zmin_percent_dx_diameter = dfpid.iloc[0].percent_dx_diameter

            # analyze z-max stats
            at_zmax_peak_int = dfpid.iloc[-1].peak_int
            at_zmax_snr = dfpid.iloc[-1].snr
            at_zmax_min_dx = dfpid.iloc[-1].min_dx
            at_zmax_percent_dx_diameter = dfpid.iloc[-1].percent_dx_diameter

            datum = [pid, x, y, r, num_frames,
                     zf_nearest_calib, zf_peak_int, zf_nsv, zf_nsv_signal,
                     at_zf_peak_int, at_zmin_peak_int, at_zmax_peak_int,
                     at_zf_snr, at_zmin_snr, at_zmax_snr,
                     at_zf_min_dx, at_zmin_min_dx, at_zmax_min_dx,
                     at_zf_percent_dx_diameter, at_zmin_percent_dx_diameter, at_zmax_percent_dx_diameter,
                     ]

            data.append(datum)

            # lastly, set particle values
            for img in self.images.values():
                for p in img.particles:
                    if p.id == pid:
                        p.set_in_focus_z(param_zf)
                        p.set_in_focus_area(np.pi * dfpid.contour_diameter.min() ** 2 / 4)
                        p.set_in_focus_intensity(at_zf_peak_int)

        columns = ['id', 'x', 'y', 'r', 'num_frames',
                   'zf_nearest_calib', 'zf_from_peak_int', 'zf_from_nsv', 'zf_from_nsv_signal',
                   'zf_peak_int', 'zmin_peak_int', 'zmax_peak_int',
                   'zf_snr', 'zmin_snr', 'zmax_snr',
                   'zf_min_dx', 'zmin_min_dx', 'zmax_min_dx',
                   'zf_percent_dx_diameter', 'zmin_percent_dx_diameter', 'zmax_percent_dx_diameter',
                   ]
        df = pd.DataFrame(data, columns=columns)

        self.spct_particle_defocus_stats = df
        self.population_statistics_spct_stats()

    def population_statistics_spct_stats(self):

        df = self.spct_particle_defocus_stats

        dfm = df.mean()
        dfstd = df.std()

        dfp = pd.concat([dfm, dfstd], axis=1)
        dfp = dfp.rename(columns={0: 'mean', 1: 'std'})

        if 'gauss_dia_x' in self.spct_stats.columns:
            x0, c1, c2 = self.population_gaussian_diameter_fit()
            mag_eff = self.optics.effective_magnification
            df_diameter = pd.DataFrame([x0, c1, c2, mag_eff],
                                       columns=['mean'],
                                       index=['pop_x0_dia', 'pop_c1', 'pop_c2', 'mag_eff'])
            dft = pd.concat([dfp, df_diameter])
        else:
            dft = dfp

        self.spct_population_defocus_stats = dft

    def population_gaussian_diameter_fit(self):

        mag_eff = self.optics.effective_magnification

        def particle_diameter_function(z, zf, c1, c2):
            return mag_eff * np.sqrt(c1 ** 2 * (z - zf) ** 2 + c2 ** 2)

        coords = []
        for img in self.images.values():
            [coords.append([img.frame, p.id, p.z_true, p.in_focus_z, p.gauss_dia_x, p.gauss_dia_y]) for p in
             img.particles]

        df = pd.DataFrame(data=coords, columns=['frame', 'id', 'z_true', 'zf', 'dia_x', 'dia_y'])

        df = df.dropna()

        df['z_corr'] = df['z_true'] - df['zf']
        df['mean_dia'] = (df['dia_x'] + df['dia_y']) / 2
        df = df.sort_values('z_corr')

        # sometimes only a narrower range will fit without errors
        dfz_narrow = df[(df.z_corr > -30) & (df.z_corr < 30)]

        # in-focus z (diameter)
        z = dfz_narrow.z_corr
        dia = dfz_narrow.mean_dia

        zf_c1_c2s, dia_zfs, dia_zmins, dia_zmaxs = \
            gaussian.calculate_minimum_of_fitted_gaussian_diameter(x=z,
                                                                   y=[dia],
                                                                   fit_function=particle_diameter_function,
                                                                   guess_x0=0,
                                                                   show_plot=True)

        x0, c1, c2 = zf_c1_c2s[0][0], zf_c1_c2s[0][1], zf_c1_c2s[0][2]

        return x0, c1, c2

    # ----------------------------------------- CALIBRATION CORRECTIONS ------------------------------------------------

    def correct_plane_tilt(self, zf_from='nsv'):

        # in-focus method + dataframe
        zf_from = 'zf_from_' + zf_from
        dff = self.spct_particle_defocus_stats[['id', zf_from]]

        # particle locations
        dfxy = self.spct_stats[['id', 'x', 'y']]
        dfxy = dfxy.groupby('id').mean()

        # merge df into dff (removing particles not in dff)
        df = pd.concat([dff, dfxy], axis=1, join="inner")

        # fit plane - this is used for converting x-y pixel locations into z-coords for correction.
        # units: (x, y units: pixels; z units: microns)
        points = np.stack((df.x, df.y, df[zf_from])).T
        px, py, pz, popt = fit.fit_3d_plane(points)

        # store fitted plane parameters
        plane_params = np.array([popt[0], popt[1], popt[2], popt[3], popt[4]]).T
        dfpp = pd.DataFrame(plane_params, index=['a', 'b', 'c', 'd', 'normal'], columns=['params'])

        # calculate z on fitted 3D plane for all particle locations
        df['z_plane'] = fit.calculate_z_of_3d_plane(df.x, df.y, popt=popt)

        # create dictionary to map new z values for spct stats
        df_plane_map = df[['id', 'z_plane']].sort_values('id')
        df_plane_map = df_plane_map.set_index('id')
        mapping_dict = df_plane_map.to_dict()
        mapper = mapping_dict['z_plane']

        # map value to spct stats
        dfxy = self.spct_stats
        dfxy['z_plane'] = dfxy['z_true']
        dfxy['z_plane'] = dfxy['id'].map(mapper)
        dfxy['z_corr'] = dfxy['z_true'] - dfxy['z_plane']

        self.spct_stats = dfxy
        self.fitted_plane_equation = dfpp

    # # --------  (OLD) -----------
    def get_calibration_correction_data(self):
        """
        Export the particle coordinates: frame, id, x, y, z, x_true, y_true, z_true, cm, max_sim, inference_stack_id
        """
        # get the percent of identified particles that were successfully measured (assigned a z-coordinate)
        coords = []

        for img in self.images.values():
            frame_id = float(img.filename.split(self.file_basestring)[-1].split(self.filetype)[0])
            [coords.append([frame_id, p.id, p.z_true, p.z, p.location[0], p.location[1], p.in_focus_z, p.in_focus_area,
                            p.snr, p.peak_intensity, p.mean_signal, p.mean_background, p.std_background]) for p in
             img.particles]

        # read coords into dataframe
        df = pd.DataFrame(data=coords, columns=['frame', 'id', 'z_true', 'z', 'x', 'y', 'z_f', 'area_f', 'snr',
                                                'peak_int', 'mean_int', 'mean_bkg', 'std_bkg'])

        # sort the dataframe by true_z
        df = df.sort_values(by='id', inplace=False)

        # return the particle coordinates
        return df

    def correct_calibration(self):
        """
        # perform calibration correction
            #   1. get peak intensity for every particle and z-position
            #   2. get z-coord of maximum intensity for every particle
            #   3. perform 3-point interpolation to get sub-image resolution.
            #   4. find mean in-focus z-coordinate for all particles
        Returns
        -------
        """
        # get peak intensity and it's z-coordinate for every particle
        # self.find_particle_in_focus_z(use_true_z=False, use_peak_int=True)

        coords = []
        for img in self.images.values():
            frame_id = float(img.filename.split(self.file_basestring)[-1].split(self.filetype)[0])
            [coords.append([frame_id, p.id, p.location[0], p.location[1], p.z, p.in_focus_z,
                            p.in_focus_intensity, p.peak_intensity,
                            p.int_var, p.int_var_sq, p.int_var_sq_norm, p.int_var_sq_norm_signal,
                            p.in_focus_area, p.area,
                            p.snr, p.mean_signal, p.mean_background, p.std_background])
             for p in img.particles if p.in_focus_z is not None]  # len(p.in_images) > 5 and

        df = pd.DataFrame(data=coords, columns=['frame', 'id', 'x', 'y', 'z', 'z_f',
                                                'peak_int_f', 'peak_int',
                                                'int_var', 'int_var_sq', 'int_var_sq_norm', 'int_var_sq_norm_signal',
                                                'area_f', 'area',
                                                'snr', 'mean_int', 'mean_bkg', 'std_bkg'])

        df = df.dropna()

        dfg = df.groupby(by='id').mean()
        z_in_focus_mean = dfg.z_f.mean()

        # get dataframe of each particle at its 'in-focus' frame
        dff = df.loc[(df['z'] == df['z_f'].round(0).astype(int))].copy()
        dff['z_f_avg'] = np.round(z_in_focus_mean, 3)

        return df, dff

    # ----------------------------------------- GET A SUBSET OF DATA (OLD) ---------------------------------------------

    def get_particles_in_collection_coords(self, true_xy=False):
        """
        Export the particle coordinates: frame, id, x, y, z, x_true, y_true, z_true, cm, max_sim, inference_stack_id
        """
        # get the percent of identified particles that were successfully measured (assigned a z-coordinate)
        coords = []

        if true_xy:
            for img in self.images.values():
                frame_id = float(img.filename.split(self.file_basestring)[-1].split(self.filetype)[0])
                [coords.append(
                    [frame_id, p.id, p.inference_stack_id, p.z_true, p.z, p.location[0], p.location[1], p.x_true,
                     p.y_true, p.cm, p.max_sim, p.match_location[0], p.match_location[1],
                     p.match_localization[0], p.match_localization[1]]) for p in img.particles]

            # read coords into dataframe
            df = pd.DataFrame(data=coords, columns=['frame', 'id', 'stack_id', 'z_true', 'z', 'x', 'y', 'x_true',
                                                    'y_true', 'cm', 'max_sim', 'xm', 'ym', 'xg', 'yg'])
        else:
            for img in self.images.values():
                frame_id = float(img.filename.split(self.file_basestring)[-1].split(self.filetype)[0])

                if self._static_templates is False:

                    [coords.append([frame_id, p.id, p.inference_stack_id,
                                    p.z_true, p.z,
                                    p.location[0], p.location[1],
                                    p.cm, p.max_sim,
                                    p.match_location[0], p.match_location[1],
                                    p.match_localization[0], p.match_localization[1],
                                    p.location_on_template[0], p.location_on_template[1],
                                    p.gauss_xc, p.gauss_yc,
                                    p.gauss_A, p.gauss_sigma_x, p.gauss_sigma_y]) for p in img.particles]

                    df = pd.DataFrame(data=coords,
                                      columns=['frame', 'id', 'stack_id',
                                               'z_true', 'z',
                                               'x', 'y',
                                               'cm', 'max_sim',
                                               'xm', 'ym',
                                               'xg', 'yg',
                                               'x_on_template', 'y_on_template',
                                               'gauss_xc', 'gauss_yc',
                                               'gauss_A', 'gauss_sigma_x', 'gauss_sigma_y'])
                else:
                    [coords.append([frame_id, p.id, p.inference_stack_id,
                                    p.z_true, p.z,
                                    p.location[0], p.location[1],
                                    p.cm, p.max_sim,
                                    p.match_location[0], p.match_location[1],
                                    p.match_localization[0], p.match_localization[1],
                                    p.location_on_template[0], p.location_on_template[1]]) for p in img.particles]

                    df = pd.DataFrame(data=coords,
                                      columns=['frame', 'id', 'stack_id',
                                               'z_true', 'z',
                                               'x', 'y',
                                               'cm', 'max_sim',
                                               'xm', 'ym',
                                               'xg', 'yg',
                                               'x_on_template', 'y_on_template'])

        # sort the dataframe by true_z
        df = df.sort_values(by='z_true', inplace=False)

        # return the particle coordinates
        return df

    # ------------------------------ CALCULATE IMAGE STATS (OLD) -------------------------------------------------------

    def calculate_calibration_image_stats(self):
        """
        img_keys = list(calib_col.images.keys())
        img_keys = sorted(img_keys, key=lambda x: float(x.split(calib_settings.inputs.image_base_string)[-1].split('.')[0]) / len(calib_col.images))
        N_cal = len(img_keys)
        img_zs = np.linspace(start=1/(2*N_cal), stop=1-1/(2*N_cal), num=len(img_keys))
        dfs = pd.DataFrame(data=img_zs, index=img_keys, columns=['z'])
        calib_col_image_stats = pd.concat([dfs, calib_col.image_stats], axis=1)
        """
        img_z = []
        img_key = []
        for img in self.images.values():
            img_key.append(img.filename)
            img_z.append(img.z)

        dfs = pd.DataFrame(data=img_z, index=img_key, columns=['z'])
        calib_col_image_stats = pd.concat([dfs, self.image_stats], axis=1)

        return calib_col_image_stats

    def calculate_image_particle_similarity(self, inspect_particle_ids=None):
        img_z = []
        img_sim = []
        collection_particle_sims = []
        for img in self.images.values():
            logger.info("Calculating particle similarity in image {}".format(img.filename))

            # compute particle similarity in image
            img_average_sim, img_particles_sims = img.infer_particles_similarity_in_image()

            if img_average_sim is None:
                continue

            # append particle similarities to image ID
            collection_particle_sims.extend(img_particles_sims)
            img_sim.append(img_average_sim)
            img_z.append(img.z)

        df_img_average_sim = pd.DataFrame(data=img_sim, index=img_z, columns=['cm'])
        df_img_average_sim.index.name = 'z'

        collection_particle_sims = np.array(collection_particle_sims)
        df_collection_particle_sims = pd.DataFrame(data=collection_particle_sims,
                                                   columns=['frame', 'z', 'image', 'template', 'cm'])

        return df_img_average_sim, df_collection_particle_sims

    def package_particle_similarity_curves(self):

        similarity_dataframes = []
        for img in self.images.values():
            for p in img.particles:
                # frame, id, inference_stack_id, location(x, y), z, z_true,
                dfs = p.similarity_curve
                dfs = dfs.rename(columns={'z': 'z_cm', 'S_SKNCCORR_SUBIMAGEOFF': 'cm'})
                dfs['frame'] = p.frame
                dfs['id'] = p.id
                dfs['inf_stack_id'] = p.inference_stack_id
                dfs['x'] = p.location[0]
                dfs['y'] = p.location[1]
                dfs['z_est'] = p.z
                dfs['z_true'] = p.z_true

                similarity_dataframes.append(dfs)

        df_sim = pd.concat(similarity_dataframes)
        df_sim = df_sim.sort_values('id')

        return df_sim

    def calculate_image_stats(self):

        stats = {
            'mean_snr_filtered': np.mean(self.image_stats.snr_filtered),
            'mean_signal': np.mean(self.image_stats.mean_signal),
            'mean_background': np.mean(self.image_stats.mean_background),
            'std_background': np.mean(self.image_stats.std_background),
            'mean_particle_density': np.mean(self.image_stats.particle_density),
            'mean_pixel_density': np.mean(self.image_stats.pixel_density),
            'percent_particles_idd': np.mean(self.image_stats.percent_particles_idd),
            'true_num_particles': np.mean(self.image_stats.true_num_particles)
        }
        return stats

    def package_test_particle_image_stats(self):
        coords = []
        for img in self.images.values():
            [coords.append([p.frame, p.id, p.z_true, p.z, p.location[0], p.location[1], p.cm,
                            p.match_location[0], p.match_location[1],
                            p.match_localization[0], p.match_localization[1],
                            p.gauss_A, p.area, p.diameter, p.aspect_ratio, p.thinness_ratio, p.solidity,
                            p.gauss_xc, p.gauss_yc,
                            p.gauss_sigma_x, p.gauss_sigma_y]) for p in img.particles]

        # read coords into dataframe
        dft = pd.DataFrame(data=coords, columns=['frame', 'id', 'z_true', 'z', 'x', 'y', 'cm', 'xm', 'ym', 'xg', 'yg',
                                                 'gauss_A', 'contour_area', 'contour_diameter', 'aspect_ratio',
                                                 'thinness_ratio', 'solidity',
                                                 'gauss_xc', 'gauss_yc', 'gauss_sigma_x', 'gauss_sigma_y'])
        return dft

    # ----------------------------------------- ASSESS MEASUREMENT QUALITY (OLD) ---------------------------------------

    def calculate_measurement_quality_global(self, local=None):
        """
        Calculate the global measurement quality by taking the mean of local evaluations.
        >Return as a dictionary.
        """
        if local is None:
            df = self.calculate_measurement_quality_local().mean()
        else:
            df = local.mean()

        return df.to_dict()

    def calculate_measurement_quality_local(self, num_bins=20, min_cm=0.9, true_xy=False, true_num_particles=None):
        """
        Method Notes:
        """
        # get the percent of identified particles that were successfully measured (assigned a z-coordinate)
        coords = []
        for img in self.images.values():
            [coords.append([p.id, p.z, p.z_true, p.cm, p.max_sim]) for p in
             img.particles]

        # read coords into dataframe
        dfz = pd.DataFrame(data=coords, columns=['id', 'z', 'true_z', 'cm', 'max_sim'])

        collection_measurement_quality_local = binning.bin_local_rmse_z(dfz, column_to_bin='true_z', bins=num_bins,
                                                                        min_cm=min_cm, z_range=None, round_to_decimal=0,
                                                                        true_num_particles=true_num_particles)

        """# round the true_z value (which is not important for this particular analysis so we do it early)
        # we have to correct to the correct decimal place to get at least 10-20 data points depending on our measurement
        # range (which can be 0-1 for normalized analyses or 0-NUM_TEST_IMAGES * Z_STEP_PER for meta analyses)
        dfz = self.bin_local_quantities(dfz)

        # sort the dataframe by true_z (now it's necessary to maintain this sorting throughout the analysis)
        dfz = dfz.sort_values(by='true_z', inplace=False)

        # FIRST, get the number of invalid measurements per rounded true_z:
        # Note, we do this in two ways and assert they equal each other to ensure we are correct.
        # Note, we store the original dataframe "dfz" to allow us to check our results after analysis.

        # make a copy of dfz, create a new column where a "1" indicates z is NaN and "0" that z is valid.
        dfz_nans = dfz.copy()
        dfz_nans.loc[pd.isna(dfz_nans['z']), 'invalid'] = 1
        dfz_nans.loc[pd.notna(dfz_nans['z']), 'invalid'] = 0

        # replace NaN with 1 b/c otherwise the groupby function will drop them
        dfz_nans = dfz_nans.fillna(value=1)

        # groupby.sum() to count NaN's per true_z grouping
        dfz_count_nans = dfz_nans.invalid.groupby(dfz_nans['true_z'], sort=False).sum().astype(int)

        # groupby.count() to count rows (particles) per true_z grouping (both valid or NaN z-values)
        dfz_count_idd = dfz_nans.groupby(dfz_nans['true_z'], sort=False).count().astype(int)

        # SECOND, we drop all rows with NaN and count the number of particles per true_z
        dfz_valid = dfz.dropna().copy()

        # count the number of rows (particles) per true_z with a valid z-measurement
        dfz_count_valid = dfz_valid.groupby(dfz_valid['true_z'], sort=False).count().astype(int)

        # THIRD, we must be very careful to maintain the order of the Panda Series and when stacking into an array
        # get: true_z (from index)
        identified_particles_true_z = dfz_count_idd.index.to_numpy(copy=True)  # [:, 0]

        # get: number of particles per true_z
        identified_particles = dfz_count_idd.to_numpy(copy=True)[:, 1]

        # get: number of particles w/ invalid z-value per true_z
        non_measured_particles = dfz_count_nans.to_numpy(copy=True)

        # get: number of particles w/ valid z-value per true_z
        measured_particles = identified_particles - non_measured_particles

        # calculate the percent of particles w/ valid z-value
        percent_measured = (identified_particles - non_measured_particles) / identified_particles * 100
        percent_measured = np.around(percent_measured, decimals=1)

        # FOURTH, we would like to know the mean_cm and mean_max_sim (max sim = interpolated cm)
        dfz_cm = dfz.copy()
        dfz_cm = dfz_cm.fillna(value=0)
        dfz_cm = dfz_cm.groupby(dfz_cm['true_z'], sort=False).mean()
        z_cm = dfz_cm.cm.to_numpy(copy=True)
        z_max_sim = dfz_cm.max_sim.to_numpy(copy=True)

        # stack 1D arrays to 2D
        particle_measure_quality = np.column_stack((identified_particles_true_z, identified_particles,
                                                    measured_particles, percent_measured, z_cm, z_max_sim))

        # create a dataframe from 2D array
        df_particle_measure_quality = pd.DataFrame(data=particle_measure_quality,
                                                   columns=['true_z', 'num_idd', 'num_valid_z_measure',
                                                            'percent_measure', 'cm', 'max_sim'])
        df_particle_measure_quality.set_index(keys='true_z', inplace=True)"""

        # NOW, we can move on and calculate the root mean squared error
        # get the local rmse uncertainty for each true_z
        # coords = []

        # if true_xy:
        """
        Case: X, Y, AND Z GROUND TRUTH IS KNOWN

        Outputs:
            * z: the mean z-value per true_z grouping.

            This can be useful in understanding if there is an inherent bias between calibration stack "labeling" 
            and the truth z-value.

            IMPORTANT NOTE:
                There can often be a discrepancy between the calibration stack labeling (or assigned z-value) and
                the true z-value of an identical particle at an identical height. This can arise from "inaccurately"
                assigning the z-value to calibration layers. The z-value is assigned based on the label from the
                filename (i.e. 'calib_1.tif' where the "1" indicates it's the bottom of the calibration stack) and 
                the known z-step size. Often the division of "1" * step size / # of steps can lead to slightly
                incorrect values. This can be even worse when analyzing synthetic datasets where the z=0 and
                z=measurement depth are not identified.

            * true_z: this is the ROUNDED true_z.
            * error_z: the mean of the difference between the RAW true_z and measured_z value by rounded true_z.
            * rmse_x,y,z: the mean root mean squared error by rounded true_z.
            * std_x,y,z: the mean of the standard deviation of the x, y, and z values by rounded true_z.
        

        # get particle locations from all images
        for img in self.images.values():
            [coords.append([p.id, p.location[0], p.location[1], p.z, p.x_true, p.y_true, p.z_true]) for p in
             img.particles]

        # read coords into dataframe
        df_rmse = pd.DataFrame(data=coords, columns=['id', 'x', 'y', 'z', 'true_x', 'true_y', 'true_z'])

        # prepare for analysis (drop NaN's and sort by z)
        df_rmse = df_rmse.dropna(axis=0, how='any')
        df_rmse = df_rmse.sort_values(by='true_z', inplace=False)

        # calculate the errors wrt the "raw" true values
        df_rmse['error_x'] = df_rmse['true_x'] - df_rmse['x']
        df_rmse['error_y'] = df_rmse['true_y'] - df_rmse['y']
        df_rmse['error_z'] = df_rmse['true_z'] - df_rmse['z']
        df_rmse['square_error_x'] = df_rmse['error_x'] ** 2
        df_rmse['square_error_y'] = df_rmse['error_y'] ** 2
        df_rmse['square_error_z'] = df_rmse['error_z'] ** 2

        # after we calculate the raw error, we can round the true_z value for the purposes of aggregate analysis
        df_rmse = self.bin_local_quantities(df_rmse)

        # RMSE uncertainty = sqrt ( sum ( error ** 2 ) / number of measurements )
        # Note: the root mean square x-, y-, and z-error is now a panda series so maintaining order is important.
        dfsum = df_rmse.copy().groupby(df_rmse['true_z'], sort=False).sum()
        dfcount = df_rmse.copy().groupby(df_rmse['true_z'], sort=False).count()
        rmse_x = (dfsum.square_error_x / dfcount.square_error_x) ** 0.5
        rmse_y = (dfsum.square_error_y / dfcount.square_error_y) ** 0.5
        rmse_xy = (rmse_x ** 2 + rmse_y ** 2) ** 0.5
        rmse_z = (dfsum.square_error_z / dfcount.square_error_z) ** 0.5

        # the standard deviation of the grouped by true_z columns (do not sort to maintain order)
        dfstd = df_rmse.copy().groupby(df_rmse['true_z'], sort=False).std()

        # the mean of the grouped by true_z columns (do not sort to maintain order)
        dfmean = df_rmse.copy().groupby(df_rmse['true_z'], sort=False).mean()

        # drop columns which are no longer necessary (do not sort to maintain order)
        dfmean.drop(
            columns=['id', 'x', 'y', 'true_x', 'true_y', 'square_error_x', 'square_error_y', 'square_error_z'],
            inplace=True)

        # assign new columns to the dataframe
        df_mean_rmse = dfmean.assign(rmse_x=rmse_x.values, rmse_y=rmse_y.values, rmse_xy=rmse_xy.values,
                                     rmse_z=rmse_z.values,
                                     std_x=dfstd.x.values, std_y=dfstd.y.values, std_z=dfstd.z.values)"""
        # else:
        """
        Case: ONLY Z GROUND TRUTH IS KNOWN
        
        Outputs:
            * z: the mean z-value per true_z grouping.
            
            This can be useful in understanding if there is an inherent bias between calibration stack "labeling" 
            and the truth z-value.
            
            IMPORTANT NOTE:
                There can often be a discrepancy between the calibration stack labeling (or assigned z-value) and
                the true z-value of an identical particle at an identical height. This can arise from "inaccurately"
                assigning the z-value to calibration layers. The z-value is assigned based on the label from the
                filename (i.e. 'calib_1.tif' where the "1" indicates it's the bottom of the calibration stack) and 
                the known z-step size. Often the division of "1" * step size / # of steps can lead to slightly
                incorrect values. This can be even worse when analyzing synthetic datasets where the z=0 and
                z=measurement depth are not identified.
                
            * true_z: this is the ROUNDED true_z.
            * error_z: the mean of the difference between the RAW true_z and measured_z value by rounded true_z.
            * rmse_z: the mean root mean squared error by rounded true_z.
            * std_x,y,z: the mean of the standard deviation of the x, y, and z values by rounded true_z.

        # get x, y, z, and z_true for all particles in image collection
        for img in self.images.values():
            [coords.append([p.id, p.location[0], p.location[1], p.z, p.z_true]) for p in img.particles]

        # read coords into dataframe
        df_rmse = pd.DataFrame(data=coords, columns=['id', 'x', 'y', 'z', 'true_z'])

        # prepare dataframe for analysis (drop NaN's and sort by z)
        df_rmse = df_rmse.dropna(axis=0, how='any')
        df_rmse = df_rmse.sort_values(by='true_z', inplace=False)

        # calculate error wrt "raw" true z
        df_rmse['error_z'] = df_rmse['true_z'] - df_rmse['z']
        df_rmse['square_error_z'] = df_rmse['error_z'] ** 2

        # we can now round the z-value in order to perform an aggregate analysis
        df_rmse = self.bin_local_quantities(df_rmse)

        # RMSE uncertainty = sqrt ( sum ( error ** 2 ) / number of measurements )
        # Note: the root mean square z-error is now a panda series so maintaining order is super important.
        dfsum = df_rmse.copy().groupby(df_rmse['true_z'], sort=False).sum()
        dfcount = df_rmse.copy().groupby(df_rmse['true_z'], sort=False).count()
        rmse_z = (dfsum.square_error_z / dfcount.square_error_z) ** 0.5

        # b/c no ground truth, get standard deviation and mean of x, y, z
        dfstd = df_rmse.copy().groupby(df_rmse['true_z'], sort=False).std()

        # take the mean after grouping by true_z (which has already been rounded to 2 decimals). Note, it is
        # important to not sort on the groupby because we need to maintain the order to re-stitch dataframes.
        dfmean = df_rmse.copy().groupby(df_rmse['true_z'], sort=False).mean()

        # drop columns that aren't necessary (inplace to be careful about order)
        dfmean.drop(columns=['id', 'x', 'y', 'square_error_z'], inplace=True)

        # assign new columns for: root mean square error and the standard deviation of the raw x, y,
        df_mean_rmse = dfmean.assign(rmse_z=rmse_z.values, std_x=dfstd.x.values, std_y=dfstd.y.values,
                                     std_z=dfstd.z.values)

        collection_measurement_quality_local = pd.concat([df_mean_rmse, df_particle_measure_quality], axis=1)"""

        return collection_measurement_quality_local

    def bin_local_uncertainty(self, dfz, bins=20, min_cm=None):
        dfz = binning.bin_local_rmse_z(dfz, column_to_bin='true_z', bins=bins,
                                       min_cm=min_cm, z_range=None, round_to_decimal=0,
                                       true_num_particles=None)
        return dfz

    def bin_local_quantities(self, dfz, column_to_bin='true_z', bins=20, min_cm=None, true_num_particles=None):
        if true_num_particles is None:
            true_num_particles = self._true_num_particles
        dfz = binning.bin_local(dfz, column_to_bin=column_to_bin, bins=bins, min_cm=min_cm, z_range=None,
                                round_to_decimal=0,
                                true_num_particles=true_num_particles)
        return dfz

    # ----------------------------------------- PLOTTING FUNCTIONS -----------------------------------------------------

    def plot(self, raw=True, draw_particles=True, exclude=[], **kwargs):
        fig = plot_img_collection(self, raw=raw, draw_particles=draw_particles, exclude=exclude, **kwargs)
        return fig

    def plot_calib_col_image_stats(self, data):
        fig = plot_calib_col_image_stats(data)
        return fig

    def plot_single_particle_stack(self, particle_id):
        fig = plot_single_particle_stack(collection=self, particle_id=particle_id)
        return fig

    def plot_particle_trajectories(self, sort_images=None):
        fig = plot_particle_trajectories(self, sort_images=sort_images)
        return fig

    def plot_particle_coordinate(self, coordinate='z', sort_images=None, particle_ids=None):
        fig = plot_particle_coordinate(self, coordinate=coordinate, sort_images=sort_images, particle_id=particle_ids)
        return fig

    def plot_particle_coordinate_calibration(self, measurement_quality, measurement_depth=None, true_xy=False,
                                             measurement_width=None):
        fig = plot_particle_coordinate_calibration(self, measurement_quality=measurement_quality,
                                                   measurement_depth=measurement_depth, true_xy=true_xy,
                                                   measurement_width=measurement_width)
        return fig

    def plot_similarity_curve(self, sub_image, method=None, min_cm=0, particle_id=None, image_id=None):
        if particle_id is None and image_id is None:
            raise ValueError("Must input either particle_id or image_id to plot similarity curve for.")
        fig = plot_similarity_curve(self, sub_image=sub_image, method=method, min_cm=min_cm, particle_id=particle_id,
                                    image_id=image_id)
        return fig

    def plot_every_image_particle_stack_similarity(self, calib_set, save_results_path, plot=False,
                                                   infer_sub_image=True, min_cm=0.5, measurement_depth=None):
        plot_every_image_particle_stack_similarity(self, calib_set=calib_set, plot=plot,
                                                   save_results_path=save_results_path, infer_sub_image=infer_sub_image,
                                                   min_cm=min_cm, measurement_depth=measurement_depth)

    def plot_bin_local_rmse_z(self, measurement_quality, measurement_depth=1, second_plot=None):
        fig = plot_bin_local_rmse_z(self, measurement_quality, measurement_depth=measurement_depth,
                                    second_plot=second_plot)
        return fig

    def plot_local_rmse_uncertainty(self, measurement_quality, measurement_depth=None, true_xy=False,
                                    measurement_width=None, second_plot=None):
        fig = plot_local_rmse_uncertainty(self, measurement_quality, measurement_depth=measurement_depth,
                                          true_xy=true_xy, measurement_width=measurement_width, second_plot=second_plot)
        return fig

    def plot_baseline_image_and_particle_ids(self, filename=None):
        fig = plot_baseline_image_and_particle_ids(self, filename=filename)
        return fig

    def plot_num_particles_per_image(self):
        fig = plot_num_particles_per_image(self)
        return fig

    def plot_particles_stats(self, stat='area'):
        fig = plot_particles_stats(self, stat=stat)
        return fig

    def plot_particle_snr_and(self, second_plot='area', particle_id=None):
        fig = plot_particle_snr_and(self, second_plot=second_plot, particle_id=particle_id)
        return fig

    def plot_particle_peak_intensity(self, particle_id=None):
        fig = plot_particle_peak_intensity(self, particle_id=particle_id)
        return fig

    def plot_particle_signal(self, optics, collection_image_stats, particle_id, intensity_max_or_mean='max'):
        fig = plot_particle_signal(self, optics, collection_image_stats, particle_id,
                                   intensity_max_or_mean=intensity_max_or_mean)
        return fig

    def plot_particle_diameter(self, optics, collection_image_stats, particle_id):
        fig = plot_particle_diameter(self, optics, collection_image_stats, particle_id)
        return fig

    def plot_gaussian_ax_ay(self, plot_type='one', p_inspect=[0]):
        fig = plot_gaussian_ax_ay(self, plot_type=plot_type, p_inspect=p_inspect)
        return fig

    def plot_particle_templates_and_z(self, particle_id, z=None, cmap='binary', draw_contours=False,
                                      fill_contours=False):
        z_true, z, template = plot_single_particle_template_and_z(self, particle_id, z, cmap, draw_contours,
                                                                  fill_contours)
        return z_true, z, template

    def plot_animated_surface(self, sort_images=None, fps=10, save_as=None):
        fig = plot_animated_surface(self, sort_images=sort_images, fps=fps, save_as=save_as)
        return fig

    @property
    def image_collection_type(self):
        return self._image_collection_type

    @property
    def folder(self):
        return self._folder

    @property
    def filetype(self):
        return self._filetype

    @property
    def files(self):
        return self._files

    @property
    def images(self):
        return self._images

    @property
    def file_basestring(self):
        return self._file_basestring

    @property
    def num_images(self):
        return len(self.images)

    @property
    def num_images_if_mean(self):
        return self._num_images_if_mean

    @property
    def measurement_depth(self):
        return self._measurement_depth

    @property
    def measurement_range(self):
        """Note, the measurement range is the actual z-distance spanned by the collection"""
        return self._measurement_range

    @property
    def particle_ids(self):
        return self.get_unique_particle_ids()

    @property
    def true_num_particles(self):
        return self._true_num_particles

    @property
    def background_img(self):
        return self._background_img

    @property
    def image_stats(self):
        image_stats = [image.stats for image in self.images.values()]
        df = pd.concat(image_stats, ignore_index=False, keys=list(self.images.keys()), names=['Image']).droplevel(1)
        return df

    @property
    def shape_tol(self):
        return self._shape_tol

    @property
    def overlap_threshold(self):
        return self._overlap_threshold

    @property
    def stacks_use_raw(self):
        return self._stacks_use_raw