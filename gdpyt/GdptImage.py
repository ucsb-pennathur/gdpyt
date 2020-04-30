import cv2

class GdptImage(object):
    """
    This class holds an image along with it's properties such as the
    raw image, filtered image, path, filename, particles present in the image. If the image is part of a calibration,
    the z coordinate is passed when creating an instance
    """

    def __init__(self, path, z=None):
        super(GdptImage, self).__init__()
        # Attributes with an underscore as the first character are "internal use". That means that they are only meant
        # to be modified by methods of this class.
        # The reasoning behind this is the following: Imagine an image with a number of particles. The particles are saved
        # in the self._particles attribute. We only want the identify_particles method in this class to modify this attribute.
        # We don't really want any other part of the program to change this attribute, since the particles in an image
        # are always going to be those that were identified with identify_particles. If this attribute isn't marked as
        # "internal use", other parts of the program could by accident add or delete particles.
        self._filepath = path
        self._filename = path.split('\\')[-1]

        # Load the image. This sets the ._raw attribute
        self.load(path)

        # Filtered image. This attribute is assigned by using the filter_image method
        self._processed = None
        self._processing_stats = None

        # Particles: dictionary {particle_id: Particle object}
        # This dictionary is filled up with the identify_particles method
        self._particles = {}

        if z is not None:
            self.set_z(z)
        else:
            self._z = None

    # The property decorator enables the decorated function to be called by GdptImage.raw instead of GdptImage.raw().
    # They make the "internal use" attributes available to the user.
    # e.g. image = GdptImage.raw -> image is a reference to whatever is saved in the ._raw attribute.
    # However, trying to assign something will raise an error, for example: GdptImage.raw = cv2.imread('image.png')
    # We want to make it impossible to change ._raw from outside of this class and that's a way to do it. You can look
    # at what's inside but you can't change it

    @property
    def raw(self):
        return self._raw

    @property
    def processed(self):
        return self._processed

    @property
    def filename(self):
        return self._filename

    @property
    def filepath(self):
        return self._filepath

    def load(self, path):
        self._raw = cv2.imread(self._filepath)

    def set_z(self, z):
        assert isinstance(z, float)
        self._z = z
        # If the image is set to be at a certain height, all the particles in it are assigned that height
        for particle in self.particles.values():
            particle.set_z(z)

    def filter_image(self, filterdict):
        """
        This is basically you image_smoothing function. The argument filterdict are similar to the arguments of the
        image_smoothing function.
        e.g. filterdict: {'median': 5, 'bilateral': 4, 'gaussian': 5} or something along those lines
        :param filterdict:
        :return:

        This method should assign self._processed and self._processing_stats
        """
        pass

    def identify_particles(self):
        """
        PSEUDO CODE
        particle_templates, centers = find_circles(self.processed)

        for id, (template, center) in enumerate(zip(particle_templates, centers)):
            self._particle.update({id, GdptParticle(id, template, center...)
        """
        pass
