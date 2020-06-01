from .GdpytParticle import GdpytParticle
from collections import OrderedDict
from .plotting import plot_calib_stack
from .similarity import *
import numpy as np
import logging

logger = logging.getLogger(__name__)

class GdpytCalibrationStack(object):

    def __init__(self, particle_id, location, dilation=None):
        super(GdpytCalibrationStack, self).__init__()
        self._id = particle_id
        self._location = location
        self._layers = OrderedDict()
        self._particles = []
        self._shape = None
        self._template_dilation = dilation

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, item):
        if isinstance(item, int):
            key = list(self.layers.keys())[item]
            return key, self.layers[key]
        else:
            return item, self.layers[item]

    def __repr__(self):
        class_ = 'GdpytCalibrationStack'
        min_z = min(list(self.layers.keys()))
        max_z = max(list(self.layers.keys()))
        repr_dict = {'Particle ID': self.id,
                     'Location (x, y)': self.location,
                     'Particle bounding box dimensions': self.shape,
                     'Template dilation': self._template_dilation,
                     'Number of layers': len(self),
                     'Min. and max. z coordinate': [min_z, max_z]}
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

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
            logger.debug('Stack resize bbox: {}'.format((w_max, h_max)))
            particle.resize_bbox(w_max, h_max)

        self._shape = (w_max, h_max)

    def add_particle(self, particle):
        assert isinstance(particle, GdpytParticle)
        self._particles.append(particle)

    def build_layers(self):
        self._uniformize_and_center()
        z = []
        templates = []

        for particle in self._particles:
            z.append(particle.z)
            templates.append(particle.get_template(dilation=self._template_dilation))

        layers = OrderedDict()
        for z, template in sorted(zip(z, templates), key=lambda k: k[0]):
            layers.update({z: template})
        self._layers = layers

    def get_layers(self, range_z=None):
        if range_z is None:
            return self._layers
        else:
            if not (isinstance(range_z, list) or isinstance(range_z, tuple)):
                raise TypeError("range_z must be a list or tuple with two elements, specifying the lower and upper"
                                "boundary of the interval. Received type {}".format(type(range_z)))
            else:
                return_layers = OrderedDict()
                for key, item in self.layers.items():
                    if range_z[0] < key < range_z[1]:
                        return_layers.update({key, item})
                return return_layers

    def infer_z(self, particle, function='ccorr'):
        if function.lower() == 'ccorr':
            if self._template_dilation is None:
                sim_func = cross_correlation_equal_shape
            else:
                sim_func = max_cross_correlation
            # Optimum for this function is the maximum
            optim = np.argmax
        elif function.lower() == 'nccorr':
            if self._template_dilation is None:
                sim_func = norm_cross_correlation_equal_shape
            else:
                sim_func = max_norm_cross_correlation
            # Optimum for this function is the maximum
            optim = np.argmax
        elif function.lower() == 'znccorr':
            if self._template_dilation is None:
                sim_func = zero_norm_cross_correlation_equal_shape
            else:
                sim_func = max_zero_norm_cross_correlation
            # Optimum for this function is the maximum
            optim = np.argmax
        else:
            raise ValueError("Unknown similarity function {}".format(function))

        z_calib, temp_calib = np.array(list(self.layers.keys())), np.array(list(self.layers.values()))
        particle.resize_bbox(*self.shape)

        sim = []
        for c_temp in temp_calib:
            sim.append(sim_func(c_temp, particle.template))
        sim = np.array(sim)
        max_idx = optim(sim)
        xp, poly = interpolation(z_calib, sim, max_idx)
        particle.set_z(xp[np.argmax(poly)])
        particle.set_max_sim(np.amax(poly))
        particle.set_similarity_curve(z_calib, sim, label_suffix=function)
        particle.set_interpolation_curve(xp, poly, label_suffix=function)

    def plot(self, z=None, draw_contours=True):
        fig = plot_calib_stack(self, z=z, draw_contours=draw_contours)
        return fig

    @property
    def id(self):
        return self._id

    @property
    def location(self):
        return self._location

    @property
    def layers(self):
        return self._layers

    @property
    def shape(self):
        return self._shape
