from .GdpytImage import GdpytImage#, GdptCalibrationStack
from .GdpytCalibrationSet import GdpytCalibrationSet
from os.path import join, isdir
from os import listdir
from collections import OrderedDict
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger()

class GdpytImageCollection(object):

    def __init__(self, folder, filetype, processing_specs=None, min_particle_size=None, max_particle_size=None):
        super(GdpytImageCollection, self).__init__()
        if not isdir(folder):
            raise ValueError("Specified folder {} does not exist".format(folder))
        
        self._folder = folder
        self._filetype = filetype
        self._find_files()
        self._add_images()

        # Define the processing done on all the images in this collection
        self._processing_specs = processing_specs

        # Minimum and maximum particle size for image in this collection
        self._min_particle_size = min_particle_size
        self._max_particle_size = max_particle_size

        self.filter_images()
        self.identify_particles()
        #self.uniformize_particle_ids()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        key = list(self.images.keys())[item]
        return self.images[key]

    def _add_images(self):
        images = OrderedDict()
        for file in self._files:
            img = GdpytImage(join(self.folder, file))
            images.update({img.filename: img})
            logger.warning('Loaded image {}'.format(img.filename))
        self._images = images

    def _find_files(self):
        """
        Identifies all files of filetype filetype in folder
        :return:
        """
        all_files = listdir(self._folder)
        save_files = []
        for file in all_files:
            if file.endswith(self._filetype):
                save_files.append(file)

        logger.warning(
            "Found {} files with filetype {} in folder {}".format(len(save_files), self.filetype, self.folder))
        # Save all the files of the right filetype in this attribute
        self._files = save_files

    def create_calibration(self, name_to_z):
        """
        This creates a calibration from this image collection
        :param name_to_z: dictionary, maps each filename to a height. e.g {'run5_0.tif': 0.0, 'run5_1.tif': 1.0 ...}
                        This could also be done differently
        :return: A list of GdptCalibrationStacks. One for each particle in the images
        """
        return GdpytCalibrationSet(self, name_to_z)

    def filter_images(self):
        for image in self.images.values():
            image.filter_image(self._processing_specs)

    def identify_particles(self):
        for image in self.images.values():
            image.identify_particles(min_size=self._min_particle_size, max_size=self._max_particle_size)

    def infer_z(self, calib_set, function='ccorr'):
        assert isinstance(calib_set, GdpytCalibrationSet)

        for image in self.images.values():
            calib_set.infer_z(image, function='function')

    def uniformize_particle_ids(self, baseline=None, threshold=50):
        baseline_img = self._files[0]
        baseline_img = self.images[baseline_img]

        baseline_locations = []
        for particle in baseline_img.particles:
            baseline_locations.append(pd.DataFrame({'x': particle.location[0], 'y': particle.location[1]},
                                                   index=[particle.id]))
        baseline_locations = pd.concat(baseline_locations).sort_index()

        # The next particle that can't be matched to a particle in the baseline gets this id
        next_id = len(baseline_locations)

        for file in self._files[1:]:
            image = self.images[file]
            # Convert to list because ordering is important
            particles = [particle for particle in image.particles]
            locations = [list(p.location) for p in particles]

            if len(locations) == 0:
                continue

            nneigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(baseline_locations.values)
            distances, indices = nneigh.kneighbors(np.array(locations))

            for distance, idx, particle in zip(distances, indices, particles):
                # If the particle is close enough to a particle of the baseline, give that particle the same ID as the
                # particle in the baseline
                if distance < threshold:
                    particle.set_id(baseline_locations.index.values[idx.squeeze()])
                else:
                    # If the particle is not in the baseline, assign it a new, non-existent id and add it to the baseline
                    # for the subsequent images
                    logger.warning("File {}: New IDs: {}".format(file, next_id))
                    particle.set_id(next_id)
                    assert (next_id not in baseline_locations.index)
                    baseline_locations = baseline_locations.append(pd.DataFrame({'x': particle.location[0], 'y': particle.location[1]},
                                                           index=[particle.id]))
                    next_id += 1

    # def _check_particle_consistency(self, tolerance):
    #     all_ids = set()
    #     for image in self.images.values():
    #         all_ids.union(set(image.particles.keys()))
    #
    #     particle_locs = pd.DataFrame()
    #     for image in self.images.values():
    #         locs_this_img = []
    #         id_this_img = all_ids.intersection(list(image.particles.keys()))
    #         for id in sorted(id_this_img):
    #             location = image.particles[id].location
    #             locs_this_img.append(pd.DataFrame({'image': image.filename, 'id': id, 'x': location[0], 'y': location[1]}))
    #         particle_locs.append(pd.concat(locs_this_img))
    #     particle_locs = pd.concat(particle_locs)
    #     median_loc = particle_locs[['id', 'x', 'y']].groupby(['id']).median().rename(columns={'x': 'x_med', 'y': 'y_med'})
    #     particle_locs = particle_locs.join(median_loc.set_index('id'), on='id')



    def update_processing(self, processing_specs, erase_old=False):
        if not isinstance(processing_specs, dict):
            raise TypeError("Processing specifications must be specified as a dictionary")
        if not erase_old:
            self._processing_specs.update(processing_specs)
        else:
            self._processing_specs = processing_specs

        self.filter_images()
        self.identify_particles(min_size=self._min_particle_size, max_size=self._max_particle_size)

    def update_particle_size_range(self, min=None, max=None):
        if min is not None:
            self._min_particle_size = min
        if max is not None:
            self._max_particle_size = max

        self.identify_particles()




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






