from .GdpytParticle import GdpytParticle
from collections import OrderedDict
from gdpyt.utils.plotting import plot_calib_stack
from gdpyt.similarity.correlation import *
from scipy.interpolate import Akima1DInterpolator
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
        self._stats = None
        self._zero = None


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

        self._stats = {'mean': np.array(templates).mean(), 'std': np.array(templates).std()}

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

    def infer_z(self, particle, function='ccorr', min_cm=0):
        assert 0 <= min_cm <= 1
        logger.info("Infering particle {}".format(particle.id))
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
        elif function.lower() == 'barnkob_ccorr':
            if self._template_dilation is None:
                sim_func = barnkob_cross_correlation_equal_shape
            else:
                sim_func = max_barnkob_cross_correlation
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
        z_interp, sim_interp = akima_interpolation(z_calib, sim, max_idx)
        # Use optimization function to find optimum z and similarity

        if sim[max_idx] > min_cm:
            particle.set_z(z_interp[optim(sim_interp)])
            particle.set_max_sim(sim[max_idx])#()sim_interp[optim(sim_interp)])
            particle.set_similarity_curve(z_calib, sim, label_suffix=function)
            particle.set_interpolation_curve(z_interp, sim_interp, label_suffix=function)
        else:
            logger.info("Cm of {:.2f} below thresh. of {:.2f} for particle ".format(sim[max_idx], min_cm, particle.id))
            particle.set_z(np.nan)

    def plot(self, z=None, draw_contours=True):
        fig = plot_calib_stack(self, z=z, draw_contours=draw_contours)
        return fig

    def reset_id(self, new_id):
        assert isinstance(new_id, int)
        self._id = new_id

        for particle in self.particles:
            particle.reset_id(new_id)
    def set_zero(self):
        areas = []
        zs = []
        for particle in sorted(self.particles, key=lambda p: p.z):
            areas.append(particle.area)
            zs.append(particle.z)

        zl, zh = (min(zs), max(zs))
        if len(zs) > 3 and len(areas) > 3:
            akima_poly = Akima1DInterpolator(zs, areas)
            z_interp = np.linspace(zl, zh, 200)
            z_zero = z_interp[np.argmin(akima_poly(z_interp))]
        else:
            z_zero = zs[np.argmin(areas)]

        # Add offset to layers and particles:
        for p in self.particles:
            p.set_z(p.z - z_zero)

        new_layers = OrderedDict()
        for z_key, templ in self.layers.items():
            new_layers.update({z_key - z_zero: templ})

        self._layers = new_layers
        self._zero = z_zero
        logger.info("Zeroing calibration stack {}. Found in-focus z position at {}".format(self.id, z_zero))

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

    @property
    def stats(self):
        return self._stats

    @property
    def particles(self):
        return self._particles

    @property
    def zero(self):
        return self._zero