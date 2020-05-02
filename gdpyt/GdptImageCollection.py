from .GdptImage import GdptImage#, GdptCalibrationStack
from os.path import join, isdir
from os import listdir
import logging

logger = logging.getLogger()

class GdptImageCollection(object):

    def __init__(self, folder, filetype, processing_specs=None, min_particle_size=None, max_particle_size=None):
        super(GdptImageCollection, self).__init__()
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
                
        logger.warning("Found {} files with filetype {} in folder {}".format(len(save_files), self.filetype, self.folder))
        # Save all the files of the right filetype in this attribute
        self._files = save_files

    def _add_images(self):
        images = {}
        for file in self._files:
            img = GdptImage(join(self.folder, file))
            images.update({img.filename: img})
            logger.warning('Loaded image {}'.format(img.filename))
        self._images = images

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
    def images(self):
        return self._images

    def identify_particles(self):
        for image in self.images.values():
            image.identify_particles(min_size=self._min_particle_size, max_size=self._max_particle_size)

    def filter_images(self):
        for image in self.images.values():
            image.filter_image(self._processing_specs)

    def create_calibration(self, name_to_z):
        """
        This creates a calibration from this image collection
        :param name_to_z: dictionary, maps each filename to a height. e.g {'run5_0.tif': 0.0, 'run5_1.tif': 1.0 ...}
                        This could also be done differently
        :return: A list of GdptCalibrationStacks. One for each particle in the images
        """

        # Assign height to each image
        for img_name, z in name_to_z.items():
            self._images[img_name].set_z(z)

        # Get all particle id's
        all_ids = set()
        for image in self._images.values():
            all_ids.add(image.particles.keys())

        calibration_stacks = []
        for id in all_ids:
            # For each id, get the Particle object from each image that corresponds to this id
            this_particle_id = []
            for image in self._images.values():
                this_particle_id = image.particles[id]

            calibration_stack = GdptCalibrationStack(this_particle_id)
            calibration_stacks.append(calibration_stack)

        return calibration_stacks
