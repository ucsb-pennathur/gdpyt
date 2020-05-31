from .GdpytCalibrationStack import GdpytCalibrationStack

class GdpytCalibrationSet(object):

    def __init__(self, collection, image_to_z, exclude=[]):
        super(GdpytCalibrationSet, self).__init__()
        self._collection = collection

        if not isinstance(image_to_z, dict):
            raise TypeError("image_to_z must be a dictionary with keys image names and z coordinates "
                            "as values. Received type {}".format(type(image_to_z)))

        for image in collection.images.values():
            if image.filename not in exclude:
                if image.filename not in image_to_z.keys():
                    raise ValueError("No z coordinate specified for image {}")
                else:
                    image.set_z(image_to_z[image.filename])
        self._create_stacks(exclude=exclude)

    def __len__(self):
        return len(self.calibration_stacks)

    def __repr__(self):
        class_ = 'GdpytCalibrationSet'
        repr_dict = {'GdpytImageCollection': self._collection.folder,
                     'Calibration stacks for particle IDs': list(self.calibration_stacks.keys())}
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str


    def _create_stacks(self, exclude=[]):
        stacks = {}
        for image in self.collection.images.values():
            if image.filename not in exclude:
                for particle in image.particles:
                    if particle.id not in stacks.keys():
                        new_stack = GdpytCalibrationStack(particle.id, particle.location)
                        new_stack.add_particle(particle)
                        stacks.update({particle.id: new_stack})
                    else:
                        stacks[particle.id].add_particle(particle)

        for stack in stacks.values():
            stack.build_layers()

        self._calibration_stacks = stacks

    def infer_z(self, image, function='ccorr'):
        for particle in image.particles:
            stack = self.calibration_stacks[particle.id]
            stack.infer_z(particle, function=function)

    @property
    def collection(self):
        return self._collection

    @property
    def calibration_stacks(self):
        return self._calibration_stacks
