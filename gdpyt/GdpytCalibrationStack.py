from .GdpytParticle import GdpytParticle
import cv2
from collections import OrderedDict

class GdpytCalibratioStack(object):

    def __init__(self, particle_id, location):
        super(GdpytCalibratioStack, self).__init__()
        self._id = particle_id
        self._location = location
        self._layers = OrderedDict()
        self._particles = []

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, item):
        key = list(self.layers.keys())[item]
        return (key, self.layers[key])

    def add_particle(self, particle):
        assert isinstance(particle, GdpytParticle)
        self._particles.append(particle)

    def infer_z(self, particle):
        assert isinstance(particle, GdpytParticle)
        assert self.id == particle.id

    def build_layers(self):
        self._uniformize_and_center()
        z = []
        templates = []

        for particle in self._particles:
            z.append(particle.z)
            templates.append(particle.template)

        layers = OrderedDict()
        for z, template in sorted(zip(z, templates), key=lambda k: k[0]):
            layers.update({z: template})
        self._layers = layers

    def _uniformize_and_center(self):
        # Find biggest bounding box
        w_max, h_max = (0, 0)
        for particle in self._particles:
            w, h = (particle.bbox[2], particle.bbox[3])
            if w > w_max:
                w_max = w
            if h > h_max:
                h_max = h

        for particle in self._particles:
            particle.resize_bbox(w_max, h_max)

    @property
    def location(self):
        return self._location

    @property
    def id(self):
        return self._id

    @property
    def layers(self):
        return self._layers