from . import GdptImage, GdptCalibrationStack

class GdptImageCollection(object):

    def __init__(self, folder, filetype, processing_specs=None):
        self._folder = folder
        self._filetype = filetype
        self._find_files()
        self._add_images()

        # Define the processing done on all the images in this collection
        self._processing_specs = processing_specs

    # A method that is for internal use is also prefixed with an underscore
    def _find_files(self):
        """ This would be your find_files function"""
        filelist = find_filetype(self.folder, self.filetype)
        # Save all the files of the right filetype in this attribute
        self._files = filelist

    def _add_images(self):
        images = {}
        for file in self._files:
            img = GdptImage(file)
            img.load()
            img.filter_image(self._processing_specs)
            images.update({img.filename: img})


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
        