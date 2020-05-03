from .GdptParticle import GdpytParticle
import cv2

class GdptCalibratioStack(object):

    def __init__(self, particle_id, location):
        self._id = particle_id
        self._location = location
        self._layers = {}


    def add_layer(self, z, template):
        self._layers.update({z: template})

    def infer_z(self, particle, correlation_func=cv2.NCOEFF):
        assert isinstance(particle, GdpytParticle)
        assert self.id == particle.id

    @property
    def location(self):
        return self._location

    @property
    def id(self):
        return self._id