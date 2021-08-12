from .GdpytImage import GdpytImage
from .GdpytCalibrationSet import GdpytCalibrationSet
from gdpyt.utils.plotting import *
from os.path import join, isdir
from os import listdir
import re
from collections import OrderedDict
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
                 shape_tol=0.2, overlap_threshold=0.3, exclude=[], subset=[], folder_ground_truth=None,
                 stacks_use_raw=False, infer_sub_image=True):
        super(GdpytImageCollection, self).__init__()
        if not isdir(folder):
            raise ValueError("Specified folder {} does not exist".format(folder))
        
        self._folder = folder
        self._folder_ground_truth = folder_ground_truth
        self._filetype = filetype
        self._stacks_use_raw = stacks_use_raw
        self._infer_sub_image = infer_sub_image

        if len(subset) > 0:
            self._get_exclusion_subset(exclude=exclude, subset=subset)
        self._find_files(exclude=exclude)

        if folder_ground_truth is not None:
            self._find_files_ground_truth(exclude=exclude)

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
        if thresholding_specs is None:
            self._thresholding_specs = {'otsu': []}
        else:
            self._thresholding_specs = thresholding_specs

        # Minimum and maximum particle size for image in this collection
        self._min_particle_size = min_particle_size
        self._max_particle_size = max_particle_size

        # Shape tolerance
        if shape_tol is not None:
            if not 0 < shape_tol < 1:
                raise ValueError("Shape tolerance parameter shape_tol must be between 0 and 1")
        self._shape_tol = shape_tol

        # Overlap threshold for merging particles
        self._overlap_threshold = overlap_threshold

        self.filter_images()
        self.identify_particles()

        if folder_ground_truth is not None:
            self.identify_particles_ground_truth()

        #self.uniformize_particle_ids()

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
        for file in self._files:
            img = GdpytImage(join(self.folder, file))
            images.update({img.filename: img})
            logger.warning('Loaded image {}'.format(img.filename))
        self._images = images

    def _get_exclusion_subset(self, exclude, subset: list):
        """
        modify the variable [exclude] to include all files not in the [subset]
        """
        base_string = subset[0]
        start = subset[1]
        stop = subset[2]
        if len(subset) > 0:
            if len(subset) < 3:
                raise ValueError("Subset requires a file base string and least two values: start and stop index.")
            elif len(subset) == 3:
                all_files = listdir(self._folder)
                save_files = []
                for file in all_files:
                    if file.endswith(self._filetype):
                        if file in exclude:
                            continue
                        save_files.append(file)
                for f in save_files:
                    search_string = base_string + '(.*)' + self._filetype
                    file_index = int(re.search(search_string, f).group(1))
                    if file_index < start or file_index > stop:
                        exclude.append(f)
            else:
                raise ValueError("Collecting multiple subsets is not implemented at the moment.")
        else:
            pass

    def _find_files(self, exclude=[]):
        """
        Identifies all files of filetype filetype in folder
        :return:
        """
        all_files = listdir(self._folder)
        save_files = []
        for file in all_files:
            if file.endswith(self._filetype):
                if file in exclude:
                    continue
                save_files.append(file)

        logger.warning(
            "Found {} files with filetype {} in folder {}".format(len(save_files), self.filetype, self.folder))
        # Save all the files of the right filetype in this attribute
        self._files = save_files

    def _find_files_ground_truth(self, exclude=[]):
        if self._folder_ground_truth == 'standard_gdpt':
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
                "Found {} files with filetype {} in ground truth folder {}".format(len(save_files), '.txt', self._folder_ground_truth))
            # Save all the files of the right filetype in this attribute
            self._files_ground_truth = save_files


    def create_calibration(self, name_to_z, exclude=[], dilate=True):
        """
        This creates a calibration from this image collection
        :param name_to_z: dictionary, maps each filename to a height. e.g {'run5_0.tif': 0.0, 'run5_1.tif': 1.0 ...}
                        This could also be done differently
        :return: A list of GdptCalibrationStacks. One for each particle in the images
        """
        return GdpytCalibrationSet(self, name_to_z, exclude=exclude, dilate=dilate)

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
        vmin, vmax = np.percentile(background_mean, q=(0.005, 99.995))  # clip the bottom 0.005% and top 0.005% intensities

        # --- compute the minimum pixel intensity across all images ---
        background_img = np.ones((sizey, sizex), dtype=np.uint16) * 2 ** 16
        # loop through images
        for i in self.images.values():
            image = i._raw.copy()
            if self._background_subtraction == 'min':
                image = np.where(image < vmax, image, vmax)  # clip upper percentile
                image = np.where(image > vmin, image, vmin)  # clip lower percentile
                background_img = np.where(image < background_img, image, background_img)  # take min value

        # store background image
        self._background_img = background_img

        # perform background subtraction
        for image in self.images.values():
            image.subtract_background(self._background_subtraction, self._background_img)
            logger.warning("Background subtraction image {}".format(image.filename))


    def filter_images(self):
        for image in self.images.values():
            image.filter_image(self._processing_specs)
            logger.warning("Filtered image {}".format(image.filename))

    def identify_particles(self):
        for image in self.images.values():
            image.identify_particles(self._thresholding_specs,
                                     min_size=self._min_particle_size, max_size=self._max_particle_size,
                                     shape_tol=self._shape_tol, overlap_threshold=self._overlap_threshold)
            logger.info("Identified {} particles on image {}".format(len(image.particles), image.filename))

    def identify_particles_ground_truth(self):

        if self._folder_ground_truth == 'standard_gdpt':
            for image in self.images.values():
                for p in image.particles:
                    x_true = 29
                    y_true = 29
                    N_cal = len(self.images)
                    z_filename = float(image.filename.split('B00')[-1].split('.')[0])
                    z_measurement_volume = 86.0
                    z_true = z_filename/N_cal
                    # measurement volume = 86 microns
                    p._set_location_true(x=x_true, y=y_true, z=z_true)

        else:
            for image in self.images.values():
                filename = image.filename[:-4]
                ground_truth = np.loadtxt(join(self._folder_ground_truth, filename + '.txt'))
                ground_truth_xyz = ground_truth[:, 0:3]

                for p in image.particles:
                    neigh = NearestNeighbors(n_neighbors=1)
                    neigh.fit(ground_truth_xyz[:, 0:2])
                    result = neigh.kneighbors([p.location])
                    x_true = ground_truth_xyz[result[1][0][0]][0]
                    y_true = ground_truth_xyz[result[1][0][0]][1]
                    z_true = ground_truth_xyz[result[1][0][0]][2]
                    p._set_location_true(x=x_true, y=y_true, z=z_true)

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

    def plot(self, raw=True, draw_particles=True, exclude=[], **kwargs):
        fig = plot_img_collection(self, raw=raw, draw_particles=draw_particles, exclude=exclude, **kwargs)
        return fig

    def plot_particle_trajectories(self, sort_images=None):
        fig = plot_particle_trajectories(self, sort_images=sort_images)
        return fig

    def plot_particle_coordinate(self, coordinate='z', sort_images=None, particle_ids=None):
        fig = plot_particle_coordinate(self, coordinate=coordinate, sort_images=sort_images, particle_id=particle_ids)
        return fig

    def plot_particle_coordinate_calibration(self, measurement_quality, measurement_depth=None, measurement_width=None):
        fig = plot_particle_coordinate_calibration(self, measurement_quality=measurement_quality, measurement_depth=measurement_depth, measurement_width=measurement_width)
        return fig

    def plot_similarity_curve(self, sub_image, method=None, min_cm=0, particle_id=None, image_id=None):
        if particle_id is None and image_id is None:
            raise ValueError("Must input either particle_id or image_id to plot similarity curve for.")
        fig = plot_similarity_curve(self, sub_image=sub_image, method=method, min_cm=min_cm, particle_id=particle_id, image_id=image_id)
        return fig

    def plot_local_rmse_uncertainty(self, measurement_quality, measurement_depth=None, measurement_width=None):
        fig = plot_local_rmse_uncertainty(self, measurement_quality, measurement_depth=measurement_depth, measurement_width=measurement_width)
        return fig

    def plot_particle_snr_and(self, second_plot='area'):
        fig = plot_particle_snr_and(self, second_plot=second_plot)
        return fig

    def plot_animated_surface(self, sort_images=None, fps=10, save_as=None):
        fig = plot_animated_surface(self, sort_images=sort_images, fps=fps, save_as=save_as)
        return fig

    def uniformize_particle_ids(self, baseline=None, threshold=50, uv=[[0,0]], baseline_img=None):
        """

        Parameters
        ----------
        baseline
        threshold
        uv
        baseline_img: should be the filename (e.g. calib_23.tif) of the image file to use as the baseline for ID assignment

        Returns
        -------

        """
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

            # NearestNeighbors(x+u,y+v): previous location (x,y) + displacement guess (u,v)
            nneigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(baseline_locations.values+uv)
            # NOTE: the displacement guess (u,v) could incorporate a "time" variable (image number or time data)
            # such that [u_i,v_i] = [u(t), v(t)] in order to better match time-dependent or periodic displacements.

            distances, indices = nneigh.kneighbors(np.array(locations))

            remove_p_not_in_calib = []
            for distance, idx, particle in zip(distances, indices, particles):
                # If the particle is close enough to a particle of the baseline, give that particle the same ID as the
                # particle in the baseline
                if distance < threshold:
                    particle.set_id(baseline_locations.index.values[idx.squeeze()])

                    # assign the baseline coordinates (x,y) to the matched particle coordinates (x,y)
                    baseline_locations.loc[particle.id, ('x','y')] = (particle.location[0],particle.location[1])
                    # the baseline is essentially the "current" location for that particle ID and after each
                    # successful identification, we update the "current" location of that particle ID.

                else:
                    # If the particle is not in the baseline, assign it a new, non-existent id and add it to the baseline
                    # for the subsequent images
                    if isinstance(baseline, GdpytCalibrationSet):
                        remove_p_not_in_calib.append(particle)
                        continue
                    logger.warning("File {}: New IDs: {}".format(file, next_id))
                    particle.set_id(next_id)
                    assert (next_id not in baseline_locations.index)
                    baseline_locations = baseline_locations.append(pd.DataFrame({'x': particle.location[0], 'y': particle.location[1]},
                                                           index=[particle.id]))
                    next_id += 1
            for p in remove_p_not_in_calib:
                image.particles.remove(p)
            # The nearest neighbor mapping creates particles with duplicate IDs under some circumstances
            # These need to be merged to one giant particle
            image.merge_duplicate_particles(thresh_specs=self._thresholding_specs)


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

    def calculate_image_stats(self):
        stats = {
            'mean_snr_filtered': np.mean(self.image_stats.snr_filtered),
            'mean_signal': np.mean(self.image_stats.mean_signal),
            'mean_background': np.mean(self.image_stats.mean_background),
            'std_background': np.mean(self.image_stats.std_background),
            'mean_particle_density': np.mean(self.image_stats.particle_density),
            'mean_pixel_density': np.mean(self.image_stats.pixel_density),
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

    def calculate_measurement_quality_local(self):
        """
        Methods:
            1. Get the number of particles identified at each true_z.
            2. Get the number of particles assigned valid z-measurement at each true_z.
            3. Calculate the percent measured particles at each true_z.
            4. Calculate the root mean squared error at each true_z.
            5. Calculate the mean and standard deviation at each true_z.
        Parameters
        ----------
        measurement_volume

        Returns
        -------

        """
        # get the percent of identified particles that were successfully measured (assigned a z-coordinate)
        coords = []
        for img in self.images.values():
            [coords.append([p.id, p.z, p.z_true]) for p in img.particles] #TODO - there needs to be a long term solution for dealing with particles w/o ground truth.

        dfz = pd.DataFrame(data=coords, columns=['id', 'z', 'true_z'])
        dfz_count = dfz.id.groupby(dfz['true_z']).count().astype(int).reset_index(name='counts')
        dfz_count_nan = dfz.z.isnull().groupby(dfz['true_z']).sum().astype(int).reset_index(name='counts')

        identified_particles_true_z = dfz_count.to_numpy(copy=True)[:, 0]
        identified_particles = dfz_count.to_numpy(copy=True)[:, 1]
        measured_particles = dfz_count_nan.to_numpy(copy=True)[:, 1]
        percent_measured = (identified_particles - measured_particles) / identified_particles * 100
        particle_measure_quality = np.column_stack((identified_particles_true_z, identified_particles, measured_particles, percent_measured))
        df_particle_measure_quality = pd.DataFrame(data=particle_measure_quality, columns=['true_z', 'num_idd', 'num_invalid_z_measure', 'percent_measure'])
        df_particle_measure_quality.set_index(keys='true_z', inplace=True)

        # get the local rmse uncertainty for each true_z
        coords = []
        for img in self.images.values():
            [coords.append([p.id, p.location[0], p.location[1], p.z, p.x_true, p.y_true, p.z_true]) for p in img.particles] #TODO - there needs to be a long term solution for dealing with particles w/o ground truth.
        df_rmse = pd.DataFrame(data=coords, columns=['id', 'x', 'y', 'z', 'true_x', 'true_y', 'true_z'])
        df_rmse.dropna(axis=0, how='any', inplace=True)
        df_rmse.sort_values(by='true_z', inplace=True)
        df_rmse['error_x'] = df_rmse['true_x'] - df_rmse['x']
        df_rmse['error_y'] = df_rmse['true_y'] - df_rmse['y']
        df_rmse['error_z'] = df_rmse['true_z'] - df_rmse['z']
        df_rmse['square_error_x'] = df_rmse['error_x'] ** 2
        df_rmse['square_error_y'] = df_rmse['error_y'] ** 2
        df_rmse['square_error_z'] = df_rmse['error_z'] ** 2

        dfsum = df_rmse.copy().groupby(df_rmse['true_z'], sort=False).sum()
        dfcount = df_rmse.copy().groupby(df_rmse['true_z'], sort=False).count()
        rmse_x = (dfsum.square_error_x / dfcount.square_error_x) ** 0.5
        rmse_y = (dfsum.square_error_y / dfcount.square_error_y) ** 0.5
        rmse_xy = (rmse_x ** 2 + rmse_y ** 2) ** 0.5
        rmse_z = (dfsum.square_error_z / dfcount.square_error_z) ** 0.5

        dfstd = df_rmse.copy().groupby(df_rmse['true_z'], sort=False).std()
        dfmean = df_rmse.copy().groupby(df_rmse['true_z'], sort=False).mean()
        dfmean.drop(columns=['id', 'x', 'y', 'true_x', 'true_y', 'square_error_x', 'square_error_y', 'square_error_z'], inplace=True)
        df_mean_rmse = dfmean.assign(rmse_x=rmse_x.values, rmse_y=rmse_y.values, rmse_xy=rmse_xy.values, rmse_z=rmse_z.values,
                                     std_x=dfstd.x.values, std_y=dfstd.y.values, std_z=dfstd.z.values)

        collection_measurement_quality_local = pd.concat([df_mean_rmse, df_particle_measure_quality], axis=1)

        return collection_measurement_quality_local



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
    def stacks_use_raw(self):
        return self._stacks_use_raw

    @property
    def background_img(self):
        return self._background_img

    @property
    def image_stats(self):
        image_stats = [image.stats for image in self.images.values()]
        return pd.concat(image_stats, ignore_index=False, keys=list(self.images.keys()), names=['Image']).droplevel(1)

    @property
    def shape_tol(self):
        return self._shape_tol

    @property
    def overlap_threshold(self):
        return self._overlap_threshold