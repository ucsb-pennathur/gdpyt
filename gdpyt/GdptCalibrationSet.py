from .GdptImageCollection import GdptImageCollection
from .GdptCalibrationStack import GdptCalibratioStack

class GdptCalibrationSet(object)

    def __init__(self, collection, image_to_z):
        super(GdptCalibrationSet, self).__init__()
        assert isinstance(collection, GdptImageCollection)
        self._collection = collection

        if not isinstance(image_to_z, dict):
            raise TypeError("image_to_z must be a dictionary with keys image names and z coordinates "
                            "as values. Received type {}".format(type(image_to_z)))

        for image in collection.images.values():
            if image.filename not in image_to_z.keys():
                raise ValueError("No z coordinate specified for image {}")
            else:
                image.set_z(image_to_z[image.filename])

    def _create_stacks(self):
        stacks = {}
        for image in self.collection.images.values():
            for particle_id, particle in image.particles.items():
                assert particle_id == particle.id
                if particle_id not in stacks.keys():
                    new_stack = GdptCalibratioStack(particle_id, particle.location)
                    new_stack.add_layer(particle.z, particle.template)
                    stacks.update({particle_id: new_stack})
                else:
                    stacks[particle_id].add_layer(particle.z, particle.template)

        self._calibration_stacks = stacks

    def infer_z(self, image):
        for particle in image.particles.values()
            stack = self.calibration_stacks[particle.id]
            stack.infer_height(particle)

    @property
    def collection(self):
        return self._collection

    def calibration_stacks(self):
        return self._calibration_stacks
