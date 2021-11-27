# import modules
from .GdpytParticle import GdpytParticle
from gdpyt.utils.plotting import plot_calib_stack, plot_calib_stack_self_similarity, plot_calib_stack_3d, plot_adjacent_self_similarity
from gdpyt.similarity.correlation import *

from collections import OrderedDict

import numpy as np
from scipy.interpolate import Akima1DInterpolator

from matplotlib import pyplot as plt

import logging

logger = logging.getLogger(__name__)

class GdpytCalibrationStack(object):

    def __init__(self, particle_id, location, dilation=None, template_padding=0, self_similarity_method='skccorr',
                 print_status=False):
        super(GdpytCalibrationStack, self).__init__()
        self._id = particle_id
        self._location = location
        self._layers = OrderedDict()
        self._particles = []
        self._shape = None
        self._template_dilation = dilation
        self._template_padding = template_padding
        self._stats = None
        self._zero = None
        self.self_similarity_method = self_similarity_method
        self.print_status = print_status


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
                     'Template padding': self.template_padding,
                     'Number of layers': len(self),
                     'Min. and max. z coordinate': [min_z, max_z]}
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def add_particle(self, particle):
        assert isinstance(particle, GdpytParticle)
        self._particles.append(particle)

    def build_layers(self):
        """
        Build the calibration stack layers (each layer is an image (2D array) across the z-height (3rd dim.)

        Steps:
            1. Uniformize image template sizes and center the particles on image by adjusting the bounding box coords.
            2. For each particle in the stack (note, there should only be one particle per stack), get a list of the
            z-coordinates and template (2D image array).
            3. Calculate the max value, mean value, and standard deviation for the entire stack. Note that these aren't
            very useful statistics at present and aren't used elsewhere.
            4. Create an ordered dictionary of z-coord and template sorted by z-coordinate.
            5. Instantiate the _layers attribute/property.
            6. Calculate the self-similarity between adjacent images in the calibration stack.
                * Note, the self-similarity is a useful measure of the 'sensitivity' of the calibration stack.
        """

        # uniformize template size and center
        self._uniformize_and_center()

        # get list of z-coordinates and template images
        z = []
        templates = []
        for particle in self._particles:
            z.append(particle.z)

            if self._template_dilation:
                templates.append(particle.get_template(dilation=self._template_dilation))
            else:
                temp = particle.get_template()

                # do not pad templates here because template padding occurs on the calibration images only immediately
                # prior to template matching (i.e. in GdpytCalibrationStack.infer_z)
                templates.append(temp)

                # check if templates is all NaNs
                array_nans = np.isnan(temp)
                count_nans = np.sum(array_nans)
                if count_nans == temp.size:
                    j = 1


        # calculate stats
        stats = np.array([(np.max(t), np.mean(t), np.std(t)) for t in templates])
        self._stats = {'max': stats[:, 0].max(),
                       'mean': stats[:, 1].mean(),
                       'std': stats[:, 2].std(),
                       }

        # create ordered dict and sort by z-coord
        layers = OrderedDict()
        for z, template in sorted(zip(z, templates), key=lambda k: k[0]):
            layers.update({z: template})

        # instantiate layers attribute
        self._layers = layers

    def _uniformize_and_center(self):
        """
        Uniformize the particle image templates and center the particle in the template.

        Steps:
            1. Loop through all the particles and store the largest bounding box side lengths.
            2. Loop through all the particles and resize the bounding box to match the largest bounding box dimensions.
            3. Update the calibration stack image shape attribute.
        """
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

    def get_layers(self, range_z=None):
        """
        Get the calibration stack layers for the entire calibration stack or for a specified z-range.
        """

        # if no range is supplied, get all layers
        if range_z is None:
            return self._layers

        # else, get the layers from the specified range
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

    def infer_z(self, particle, function='sknccorr', min_cm=0, infer_sub_image=True):
        """
        Infer the z-coordinate given a particle

        Steps:
            1. Get array of z-coords and image templates from the calibration stack.
            2. if the test particle image template is larger than the calibration template, shrink the test particle
            image template in the larger dimension to match the calibration template.
            3. determine cross-correlation method and optimum function (function that determines "best" correlation).
            4. perform the cross-correlation against each layer in calibration stack and append results to a list.
            5. evaluate correlation value
                5.1 apply sub-image interpolation if True
                5.2. if the correlation is below min_cm, print particle ID to console and set the z-height to NaN

        Notes:
            * This method sets several highly important particle attributes:
                1. z: the z-coordinate of the highest correlation.
                2. max_sim: the maximum correlation value
                3. similarity_curve: array of z-coords and the cross-correlation values
                4. interpolation_curve: array of z-coords and interpolated correlation values
            * The sub-image interpolation is performed using scipy.optimize fitting a parabolic function
                * Fit a parabolic function to the three correlation values centered on the z-height of maximum correlatoi
            * The sub-image interpolation was performed previously using scipy.interpolate.Akima1DInterpolator
                * Fit piecewise cubic polynomials which IMPORTANT **pass through the given data points**
                * Note the Akima1Dinterpolator should only be used for "precise" data. The application discussed in
                docs.scipy is "useful for plotting a pleasingly smooth curve through a few given points for the purpose
                of plotting."
                * Read more here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html

        Parameters
        ----------
        particle
        function
        min_cm
        infer_sub_image

        Returns
        -------

        """

        # the minimum correlation value must be between 0 and 1
        assert 0 <= min_cm <= 1

        # option to print particle id being inferred to console
        if self.print_status:
            logger.info("Infering particle {}".format(particle.id))

        """
        Note:
            * If an template is too close to the image border--such that its template would extend beyond the image 
            border--the template is padded with NaNs. This can create a ragged calibration stack (a 3D array consisting
            of 2D arrays of different sizes). The image inference method 'sknccorr' can recognize mismatches in size and
            appropriate choose the appropriate images for the 'image' and 'template'.
        """

        # get array of z-coords and image templates
        z_calib, temp_calib_original = np.array(list(self.layers.keys())), np.array(list(self.layers.values()))

        # pad the calibration image templates to allow for test particle template sliding across the calibration image
        temp_calib = []
        for t in temp_calib_original:
            temp_calib.append(np.pad(t, pad_width=1, mode='constant', constant_values=np.min(t)))
        temp_calib = np.array(temp_calib)

        """ End Note """

        # if the particle image is larger than the calibration template, resize the particle image
        calib_template_x, calib_template_y = np.shape(temp_calib[0])

        if particle.template.shape[0] > calib_template_x and particle.template.shape[1] > calib_template_y:
            particle.resize_bbox(*self.shape)
            print('particle template x and y lengths at true_z {} are too large for calibration template'.format(particle.z_true))
        elif particle.template.shape[0] > calib_template_x:
            new_shape = (calib_template_x, particle.template.shape[1])
            particle.resize_bbox(*new_shape)
            print('particle template x-length at true_z {} is too large for calibration template'.format(particle.z_true))
        elif particle.template.shape[1] > calib_template_y:
            new_shape = (particle.template.shape[0], calib_template_y)
            particle.resize_bbox(*new_shape)
            print('particle template y-length at true_z {} is too large for calibration template'.format(particle.z_true))

        # if/elif function to pass the correct cross-correlation method and optimum function
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
        elif function.lower() == 'barnkob_ccorr' or function.lower() == 'bccorr':
            if self._template_dilation is None:
                sim_func = barnkob_cross_correlation_equal_shape
            else:
                sim_func = max_barnkob_cross_correlation
            # Optimum for this function is the maximum
            optim = np.argmax
        elif function.lower() == 'sknccorr':
            sim_func = sk_norm_cross_correlation
            optim = np.argmax
        else:
            raise ValueError("Unknown similarity function {}".format(function))

        # perform the cross-correlation against each image in the calibration stack and append the results to a list
        sim = []
        for c_temp in temp_calib:
            sim.append(sim_func(c_temp, particle.template))
        sim = np.array(sim)
        max_idx = optim(sim)
        particle.set_cm(sim[max_idx])

        # evaluate correlation value
        if sim[max_idx] > min_cm and infer_sub_image is False:
            particle.set_z(z_calib[optim(sim)])
            particle.set_max_sim(sim[max_idx])
            particle.set_similarity_curve(z_calib, sim, label_suffix=function+'_subimageOFF')

        # apply sub-image interpolation if True
        elif sim[max_idx] > min_cm and infer_sub_image is True:
            z_interp, sim_interp = parabolic_interpolation(z_calib, sim, max_idx) # Use optimization function to find optimum z and similarity
            particle.set_z(z_interp[optim(sim_interp)])
            particle.set_max_sim(sim_interp[optim(sim_interp)])
            particle.set_similarity_curve(z_calib, sim, label_suffix=function+'_subimageOFF')
            particle.set_interpolation_curve(z_interp, sim_interp, label_suffix=function+'_subimageON')

        # always print to console if the correlation is below min_cm and set the z-height to NaN
        else:
            logger.info("Cm of {:.2f} below thresh. of {:.2f} for particle ".format(sim[max_idx], min_cm, particle.id))
            particle.set_z(np.nan)

    def infer_self_similarity(self, function='sknccorr'):
        logger.info("Inferring self-similarity for calibration stack {}".format(self.id))

        if function.lower() == 'barnkob_ccorr' or function.lower() == 'bccorr':
            if self._template_dilation is None:
                sim_func = barnkob_cross_correlation_equal_shape
            else:
                sim_func = max_barnkob_cross_correlation
        elif function.lower() == 'nccorr':
            if self._template_dilation is None:
                sim_func = norm_cross_correlation_equal_shape
            else:
                sim_func = max_norm_cross_correlation
        elif function.lower() == 'znccorr':
            if self._template_dilation is None:
                sim_func = zero_norm_cross_correlation_equal_shape
            else:
                sim_func = max_zero_norm_cross_correlation
        elif function.lower() == 'ccorr':
            if self._template_dilation is None:
                sim_func = cross_correlation_equal_shape
            else:
                sim_func = max_cross_correlation
        elif function.lower() == 'sknccorr':
            sim_func = sk_norm_cross_correlation
        else:
            raise ValueError("Unknown similarity function {}".format(function))

        z_calib, temp_calib = np.array(list(self.layers.keys())), np.array(list(self.layers.values()))
        num_layers = len(temp_calib)

        sim_self_backward = []
        sim_self_forward = []
        sim_self_adjacent = []
        for index, c_temp in enumerate(temp_calib):
            if index < num_layers - 1:

                padded_image = np.pad(temp_calib[index], pad_width=3, mode='constant',
                                      constant_values=np.min(temp_calib[index]))

                forward = sim_func(padded_image, temp_calib[index + 1])
                sim_self_forward.append(forward)
                if index > 0:
                    backward = sim_func(padded_image, temp_calib[index - 1])
                    sim_self_backward.append(backward)

                    # mean similarity
                    center = np.mean([forward, backward])
                    sim_self_adjacent.append(center)

        # forward similarity
        forward_sim_self = np.array(sim_self_forward)
        forward_z_self = np.squeeze(np.array([z_calib[:num_layers-1]]))
        self._self_similarity_forward = np.vstack((forward_z_self, forward_sim_self)).T

        # mean of similarity with both forward and backward images
        sim_self = np.array(sim_self_adjacent)
        z_self = np.squeeze(np.array([z_calib[1:num_layers-1]]))
        self._self_similarity = np.vstack((z_self, sim_self)).T

    def plot_adjacent_self_similarity(self, index=[]):

        if self._self_similarity is None:
            self.infer_self_similarity(function='sknccorr')

        fig = plot_adjacent_self_similarity(self, index=index)

        return fig

    def plot_calib_stack(self, z=None, draw_contours=True, fill_contours=False, imgs_per_row=5, fig=None, ax=None, format_string=False):
        fig = plot_calib_stack(self, z=z, draw_contours=draw_contours, fill_contours=fill_contours, imgs_per_row=imgs_per_row, fig=fig, axes=ax, format_string=format_string)
        return fig

    def plot_3d_stack(self, intensity_percentile=(10, 98.75), stepsize=5, aspect_ratio=3):
        fig = plot_calib_stack_3d(self,  intensity_percentile=intensity_percentile, stepsize=stepsize, aspect_ratio=aspect_ratio)
        return fig

    def plot_self_similarity(self):
        self.infer_self_similarity()
        fig = plot_calib_stack_self_similarity(self)
        return fig

    def reset_id(self, new_id):
        assert isinstance(new_id, int)
        self._id = new_id

        for particle in self.particles:
            particle.reset_id(new_id)

    def set_zero(self, z_zero=0, offset=0):
        """
        This method sets the zero location for all particles based on an 'offset' input and the location of minimum
        area. Note, that it sets all particles to an identical z-coordinate.

        Parameters
        ----------
        offset
        """
        if offset == 0:
            logger.warning("No offset was applied. Consider individual stack zero-ing.")
            """
            areas = []
            zs = []
            for particle in sorted(self.particles, key=lambda p: p.z):
                areas.append(particle.area)
                zs.append(particle.z)

            zl, zh = (min(zs), max(zs))

            if len(zs) > 3 and len(areas) > 3:
                akima_poly = Akima1DInterpolator(zs, areas)
                z_interp = np.linspace(zl, zh, 500) # Note: 500 was chosen because it yielded the best results: 8/11/2021
                z_zero = z_interp[np.argmin(akima_poly(z_interp))]

                # code for testing which interpolation is best for sub-image resolution

                fig, ax = plt.subplots()
                ax.plot(z_interp, akima_poly(z_interp), label='akima min: {}'.format(z_zero))
                order = [9, 11, 15]
                for o in order:
                    z_poly = np.polyfit(zs, areas, o)
                    z_p = np.poly1d(z_poly)
                    z_zero_poly = z_interp[np.argmin(z_p(z_interp))]
                    ax.plot(z_interp, z_p(z_interp), label='{} poly min: {}'.format(o, z_zero_poly))
                ax.scatter(zs, areas, color='black', label='contour area')
                ax.set_xlabel('z/h')
                ax.set_ylabel(r'$A_p$ ($pixles^2$)')
                ax.set_title(r'$GdpytCalibrationStack$.zero_stacks')
                ax.legend(fontsize=8)
                plt.show()


            else:
                z_zero = zs[np.argmin(areas)]

            z_zero = np.around(z_zero, decimals=3)
            z_zero = z_zero - offset
            
            self._layers = new_layers
            self._zero = z_zero
            logger.info("Zeroing calibration stack {}. Found in-focus z position at {}".format(self.id, z_zero))
            """
            pass
        else:
            # Add offset to particles
            for p in self.particles:
                p.set_z(p.z - offset)
                p.set_true_z(p.z_true - offset)
                p.set_in_focus_z(p.in_focus_z - offset)

            # Add offset to layers
            new_layers = OrderedDict()
            for z_key, templ in self.layers.items():
                new_layers.update({z_key - offset: templ})

            self._layers = new_layers

        self._zero = z_zero
        logger.info("Zeroing calibration stack {}. Set in-focus z position to {}".format(self.id, z_zero))

    def calculate_stats(self):
        """

        Parameters
        ----------
        true_num_particles: the TRUE total number of particles across all images
        measurement_volume:

        Returns
        -------

        """
        snrs = []
        areas = []
        for p in self.particles:
            snrs.append(p.snr)
            areas.append(p.area)

        stats = {
            'particle_id': p.id,
            'layers': len(self.layers),
            'avg_snr': np.mean(snrs),
            'avg_area': np.mean(areas),
            'min_particle_area': np.min(areas),
            'max_particle_area': np.max(areas),
            'min_particle_dia': np.sqrt(np.min(areas) * 4 / np.pi),
            'max_particle_dia': np.sqrt(np.max(areas) * 4 / np.pi),
                }
        return stats

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
    def self_similarity(self):
        return self._self_similarity

    @property
    def self_similarity_forward(self):
        return self._self_similarity_forward

    @property
    def particles(self):
        return self._particles

    @property
    def zero(self):
        return self._zero

    @property
    def template_padding(self):
        return self._template_padding