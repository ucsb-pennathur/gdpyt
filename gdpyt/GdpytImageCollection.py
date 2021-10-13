import matplotlib.pyplot as plt

from .GdpytImage import GdpytImage
from .GdpytCalibrationSet import GdpytCalibrationSet
from gdpyt.utils.plotting import *
from gdpyt.similarity.correlation import parabola
from gdpyt.utils import binning
from gdpyt.subpixel_localization.gaussian import fit as fit_gaussian_subpixel

from os.path import join, isdir
from os import listdir
import re
from collections import OrderedDict

from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import curve_fit
from gdpyt.subpixel_localization.gaussian import gaussian1D
from sklearn.neighbors import NearestNeighbors
from skimage.filters.rank import median
import pandas as pd
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class GdpytImageCollection(object):

    def __init__(self, folder, filetype, crop_specs=None, processing_specs=None, thresholding_specs=None,
                 background_subtraction=None, min_particle_size=None, max_particle_size=None,
                 shape_tol=0.2, overlap_threshold=0.3, exclude=[], subset=None, folder_ground_truth=None,
                 stacks_use_raw=False, infer_sub_image=True, measurement_depth=None, true_num_particles=None,
                 if_img_stack_take='mean', take_subset_mean=None, inspect_contours_for_every_image=False,
                 template_padding=3, file_basestring=None, same_id_threshold=10, image_collection_type=None,
                 calibration_stack_z_step=None,
                 baseline=None, hard_baseline=False, particle_id_image=None, static_templates=False):
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

        # properties of the image collection
        self._image_collection_type = image_collection_type
        self._folder = folder
        self._filetype = filetype
        self._file_basestring = file_basestring
        self._folder_ground_truth = folder_ground_truth

        # image collection data
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
        self.particle_id_image = particle_id_image
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
        self.uniformize_particle_ids(baseline=baseline)

        self.identify_particles_ground_truth()
        self.refine_particles_ground_truth()

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
                random_files = [rf for rf in random.sample(set(save_files), subset)]
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
                    if file_index >= start or file_index <= stop:
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

    def create_calibration(self, name_to_z, dilate=True, template_padding=0, min_num_layers=None,
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
                                   min_num_layers=min_num_layers, self_similarity_method=self_similarity_method,
                                   exclude=exclude)

        # find the in-focus coordinates of all particles and set the particle's in-focus z-coordinate and area
        self.find_particle_in_focus_z(use_true_z=False)

        # determine the z-coordinate where most particles are in focus and set this as the collection in_focus_z
        self.find_collection_z_of_min_area()
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

        sizey, sizex = np.shape((list(self.images.values())[0]._raw))

        # --- compute the mean image intensity percentile across all images ---
        background_add = np.zeros((sizey, sizex), dtype=np.uint16)

        for i in self.images.values():
            image = i._raw.copy()
            background_add = background_add + image  # add images

        # take mean
        background_mean = np.divide(background_add, len(self.images))

        # compute percentile limits
        vmin, vmax = np.percentile(background_mean,
                                   q=(0.005, 99.995))  # clip the bottom 0.005% and top 0.005% intensities

        # --- compute the minimum pixel intensity across all images ---
        background_img = np.ones((sizey, sizex), dtype=np.uint16) * 2 ** 16
        # loop through images
        for i in self.images.values():
            image = i._raw.copy()
            if self._background_subtraction == 'min':
                image = np.where(image < vmax, image, vmax)  # clip upper percentile
                image = np.where(image > vmin, image, vmin)  # clip lower percentile
                background_img = np.where(image < background_img, image, background_img)  # take min value
            else:
                raise ValueError("{} background subtraction method is not yet implemented".format(self._background_subtraction))

        # store background image
        self._background_img = background_img

        # perform background subtraction
        for image in self.images.values():
            image.subtract_background(self._background_subtraction, self._background_img)
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
            logger.warning("Filtered image {}".format(image.filename))

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
                                        padding=self._template_padding, image_collection_type=self._image_collection_type,
                                        particle_id_image=particle_identification_image)
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

    def uniformize_particle_ids_and_groups(self, baseline=None, baseline_img=None, hard_baseline=False,  uv=None):
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
                            baseline_locations.loc[particle.id, ('x', 'y')] = (particle.location[0], particle.location[1])
                        else:
                            baseline_locations.loc[particle.id, ('x', 'y')] = (particle.location[0], particle.location[1])


                    elif indic == 0:
                        # If the particle is not in the baseline, we may remove it via two methods:
                        #   1. if the baseline is a CalibrationSet, as including will significantly reduce accuracy.
                        #   2. if we designate a "hard baseline" where we don't want to add new particles.

                        # filter if not in CalibrationSet baseline:
                        if isinstance(baseline, GdpytCalibrationSet):
                            remove_p_not_in_calib.append(particle)
                            logger.warning("Removed particle {} at location {} b/c not in Calibration Set baseline".format(particle.id,
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
            nneigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(baseline_locations.values + uv)
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
                        logger.warning("Removed particle {} at location {} b/c not in Calibration Set baseline".format(particle.id,
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

    def find_particle_in_focus_z(self, use_true_z=False):
        """
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
        for pid in particle_ids: # for every particle ID

            # initialize lists for particle z-coordinate and area
            zs = []
            areas = []

            for img in self.images.values(): # for every image
                for particle in img.particles: # for every particle

                    # append the in-focus data to lists
                    if particle.id == pid:

                        # get particle z-coordinate
                        if use_true_z:
                            zs.append(particle.z_true)
                        else:
                            zs.append(particle.z)

                        # get particle area
                        areas.append(particle.area)


            # get min/max z-coordinates
            zl, zh = (min(zs), max(zs))

            # get index of minimum area
            amin_index = int(np.argmin(areas))

            # create lower and upper bounds (+/- 5 images) - Note, 5 images were chosen to smooth out the contour noise
            # UPDATED 9/25: to 1 images
            if self.image_collection_type == 'calibration':
                fit_pad = 1
            else:
                fit_pad = 2

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

                if self.image_collection_type == 'calibration' and len(particle_ids) == 1:
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
                    plt.savefig(join(savedir, savename))

            # if less than three points, get the minimum of the areas
            else:
                z_zero = zs[np.argmin(areas)]
                areas_zero = np.min(areas)

            # round the z-coordinate and area to a reasonable value
            z_zero = np.round(z_zero, 3)
            areas_zero = np.round(areas_zero, 1)

            # Set in-focus plane z-coordinate for all particles:
            for imgg in self.images.values(): # for every image
                for p_set in imgg.particles:
                    if p_set.id == pid:
                        p_set.set_in_focus_z(z_zero)
                        p_set.set_in_focus_area(areas_zero)


    def find_collection_in_focus_z(self):
        """
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

        show_plot = True
        if show_plot:
            fig, ax = plt.subplots()
            ax.scatter(in_focus_zs, in_focus_areas, color='tab:blue', alpha=0.75, label='particle area')
            ax.scatter(in_focus_z, in_focus_area, s=50, color='red', marker='.', label='min = median(area)')
            ax.set_xlabel('z')
            ax.set_ylabel('area')
            ax.grid(alpha=0.25)

            ax.set_title("Calibration collection mininum area {} at z_true = {}".format(in_focus_area, in_focus_z))
            plt.tight_layout()

            savedir = '/Users/mackenzie/Desktop/dumpfigures/'
            savefigpath = join(savedir + '_calibration_collection_minimum_z_area.png')
            fig.savefig(fname=savefigpath, bbox_inches='tight')
            plt.close()

    def find_collection_z_of_min_area(self):
        """
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

            show_plot = True
            if show_plot:
                fig, ax = plt.subplots()
                ax.scatter(zs[lower_index:upper_index + 1], areas[lower_index:upper_index + 1], color='tab:blue',
                           alpha=0.75, label='particle area')
                ax.plot(z_local, areas_interp, color='black', alpha=0.5, label='interpolated')
                ax.scatter(z_zero, areas_zero, s=50, color='red', marker='.', label='min')
                ax.set_xlabel('z')
                ax.set_ylabel('area')
                ax.grid(alpha=0.25)

                ax.set_title("Min area {} at z_true = {}".format(areas[amin_index], zs[amin_index]))
                plt.suptitle("{} collection mininum interpolated area {} at z_true = {}".format(self._image_collection_type,
                                                                                                np.round(areas_zero, 3),
                                                                                                np.round(z_zero, 3)))
                plt.tight_layout()

                savedir = '/Users/mackenzie/Desktop/dumpfigures/'
                savefigpath = join(savedir + '_{}_collection_minimum_z_area.png'.format(self._image_collection_type))
                fig.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()

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

    def calculate_measurement_quality_local(self, num_bins=20, min_cm=0.9, true_xy=False):
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

        collection_measurement_quality_local = binning.bin_local_rmse_z(dfz, num_bins=num_bins, min_cm=min_cm)

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

    def bin_local_uncertainty(self, dfz, num_bins=20, min_cm=None):
        dfz = binning.bin_local_rmse_z(dfz, num_bins=num_bins, min_cm=min_cm)
        return dfz

    def bin_local_quantities(self, dfz, column_to_bin='true_z', num_bins=20):
        dfz = binning.bin_local(dfz, column_to_bin=column_to_bin, num_bins=num_bins)
        return dfz

    def plot(self, raw=True, draw_particles=True, exclude=[], **kwargs):
        fig = plot_img_collection(self, raw=raw, draw_particles=draw_particles, exclude=exclude, **kwargs)
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

    def plot_gaussian_ax_ay(self, plot_type='one', p_inspect=[0]):
        fig = plot_gaussian_ax_ay(self, plot_type=plot_type, p_inspect=p_inspect)
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
        fig = plot_bin_local_rmse_z(self, measurement_quality, measurement_depth=measurement_depth, second_plot=second_plot)
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

    def plot_particle_peak_intensity(self):
        fig = plot_particle_peak_intensity(self)
        return fig

    def plot_particle_signal(self, optics, collection_image_stats, particle_id):
        fig = plot_particle_signal(self, optics, collection_image_stats, particle_id)
        return fig

    def plot_calib_col_image_stats(self, data):
        fig = plot_calib_col_image_stats(data)
        return fig

    def plot_animated_surface(self, sort_images=None, fps=10, save_as=None):
        fig = plot_animated_surface(self, sort_images=sort_images, fps=fps, save_as=save_as)
        return fig

    def get_particles_in_collection_coords(self, true_xy=False):
        """
        Export the particle coordinates: frame, id, x, y, z, x_true, y_true, z_true, cm, max_sim, inference_stack_id
        """
        # get the percent of identified particles that were successfully measured (assigned a z-coordinate)
        coords = []

        if true_xy:
            for img in self.images.values():
                frame_id = float(img.filename.split(self.file_basestring)[-1].split(self.filetype)[0])
                [coords.append([frame_id, p.id, p.inference_stack_id, p.z_true, p.z, p.location[0], p.location[1], p.x_true, p.y_true, p.cm,
                                    p.max_sim]) for p in img.particles]

            # read coords into dataframe
            df = pd.DataFrame(data=coords, columns=['frame', 'id', 'stack_id', 'z_true', 'z', 'x', 'y', 'x_true',
                                                    'y_true', 'cm', 'max_sim'])
        else:
            for img in self.images.values():
                frame_id = float(img.filename.split(self.file_basestring)[-1].split(self.filetype)[0])
                [coords.append([frame_id, p.id, p.inference_stack_id, p.z_true, p.z, p.location[0], p.location[1], p.cm,
                                p.max_sim]) for p in img.particles]

            # read coords into dataframe
            df = pd.DataFrame(data=coords, columns=['frame', 'id', 'stack_id', 'z_true', 'z', 'x', 'y', 'cm', 'max_sim'])

        # sort the dataframe by true_z
        df = df.sort_values(by='z_true', inplace=False)

        # return the particle coordinates
        return df

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