import math

import cv2
import numpy as np
import numpy.ma as ma
import pandas as pd
import skimage.draw
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_dilation, disk
from scipy.signal import convolve2d

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from skimage.draw import ellipse_perimeter
from gdpyt.utils.plotting import filter_ellipse_points_on_image

from gdpyt.particle_identification import binary_mask
from gdpyt.subpixel_localization import gaussian
from gdpyt.subpixel_localization.gaussian import fit as fit_gaussian_subpixel, gauss_2d_function
from gdpyt.subpixel_localization.gaussian import fit_results, plot_2D_image_contours, plot_3D_fit, plot_3D_image
from gdpyt.subpixel_localization.centroid_based_iterative import refine_coords_via_centroid, bandpass, grey_dilation
from gdpyt.subpixel_localization.centroid_based_iterative import plot_2D_image_and_center, validate_tuple

class GdpytParticle(object):

    def __init__(self, image_raw, image_filt, id_, contour, bbox, particle_mask_on_image, particle_collection_type,
                 location=None, frame=None, template_use_raw=False, fit_gauss=True):
        super(GdpytParticle, self).__init__()
        self._id = int(id_)
        self.frame = frame
        assert isinstance(image_raw, np.ndarray)
        assert isinstance(image_filt, np.ndarray)
        self._image_raw = image_raw
        self._image_filt = image_filt
        self._use_raw = template_use_raw
        self._contour = contour
        self._bbox = bbox
        self._particle_collection_type = particle_collection_type
        self._mask_on_image = particle_mask_on_image
        self._template = None
        self._location = None
        self.match_location = None
        self.match_localization = None
        self._location_on_template = None
        self._mask_on_template = None
        self._template_contour = None
        self._fitted_gaussian_on_template = None
        self._in_images = None
        self.inference_stack_id = None
        self._cm = None
        self._similarity_curve = None
        self._interpolation_curve = None
        self._x_true = None
        self._y_true = None
        self._z = None
        self._z_true = None
        self._z_default = None
        self.in_focus_z = None
        self.in_focus_area = None
        self.in_focus_intensity = None
        self._snr = None
        self._peak_intensity = None
        self.int_var = None
        self.int_var_sq = None
        self.int_var_sq_norm = None
        self.int_var_sq_norm_signal = None
        self._mean_signal = None
        self._mean_background = None
        self._std_background = None
        self._mean_intensity_of_template = None
        self._max_sim = None

        self.mean_dx = None
        self.min_dx = None
        self.num_dx = None
        self.mean_dxo = None
        self.num_dxo = None
        self.percent_dx_diameter = None

        # if location is passed, then set; otherwise, compute the center
        if location is not None:
            self._set_location(location)
        else:
            self._compute_center()

        # calculate particle image shape stats: area, aspect ratio, thinness ratio, hull, hull area, solidity.
        self._compute_convex_hull()

        # set the _template, _location_on_template, and _template_contour attributes.
        self._create_template(bbox=bbox)

        # fit a Gaussian profile to find subpixel center and recenter the bbox
        if fit_gauss:
            self.gauss_dia_x = None
            self.gauss_dia_y = None
            self.gauss_A = None
            self.gauss_yc = None
            self.gauss_xc = None
            self.gauss_sigma_y = None
            self.gauss_sigma_x = None

            self._fit_2D_gaussian(normalize=True)

            # self._compute_center_subpixel(method='gaussian', save_plots=False)
        else:
            pass
            # self._compute_center_subpixel(method='centroid', save_plots=False)

        # compute the particle stats on the refined bounding box and mask
        self.compute_local_snr()

    def __repr__(self):
        class_ = 'GdpytParticle'
        repr_dict = {'ID': self.id,
                     'Location': self.location,
                     'Bounding box dimensions': [self.bbox[2], self.bbox[3]],
                     'Area': self.area,
                     'Solidity': self.solidity,
                     'SNR': self.snr,
                     'Z coordinate': self.z}
        out_str = "{}: {} \n".format(class_, self.id)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def add_particle_in_image(self, img_id):
        self._in_images = img_id

    def _compute_convex_hull(self):
        """
        area: the area is computed using Green's formula and thus can be different than the number of non-zero pixels.
        hull: the contour coordinates of the directly connected outer most points of the contour
        hull_area: area of the directly connected outer most points of the contour
        solidity: the circularity of the contour where 1=perfect circle.
        """
        area = float(cv2.contourArea(self.contour))
        diameter = np.round(np.sqrt(area * 4 / np.pi))
        aspect_ratio = self.bbox[2] / self.bbox[3]
        perimeter = cv2.arcLength(self.contour, True)
        thinness_ratio = 4 * np.pi * area / perimeter ** 2
        hull = cv2.convexHull(self.contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area

        self._area = area
        self._diameter = diameter
        self._aspect_ratio = aspect_ratio
        self._thinness_ratio = thinness_ratio
        self._hull = hull
        self._hull_area = hull_area
        self._solidity = solidity


    def _compute_center(self):
        """
        Compute the center of the contour.
        """
        # compute center from contour
        M = cv2.moments(self._contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        self._set_location((cX, cY))

    def _create_template(self, bbox=None, adjusted_template=False, apply_ellipsoid_mask=False):

        # create a string to put in the showed and saved image if the template is original or adjusted after subpix
        if adjusted_template is False:
            adj_temp = 'ORIGINAL'
        else:
            adj_temp = 'ADJUSTED'

        # create template from raw image or filtered image
        if self.use_raw:
            image = self._image_raw
        else:
            image = self._image_filt

        # if no bbox is passed in, use the particle.bbox variable
        if bbox is None:
            x0, y0, w0, h0 = self.bbox
            x, y, w, h = self.bbox
        else:
            x0, y0, w0, h0 = bbox
            x, y, w, h = bbox

        orig_template = image[y: y + h, x: x + w]

        # adjust the bounding box so it doesn't exceed the image bounds
        pad_x_m, pad_x_p, pad_y_m, pad_y_p = 0, 0, 0, 0

        if y + h > image.shape[0]:
            pad_y_p = y + h - image.shape[0]
        if y < 0:
            pad_y_m = - y
            h = y + h
            y = 0
        if x + w > image.shape[1]:
            pad_x_p = x + w - image.shape[1]
        if x < 0:
            pad_x_m = - x
            w = x + w
            x = 0
        pad_x = (pad_x_m, pad_x_p)
        pad_y = (pad_y_m, pad_y_p)

        # if no padding is necessary, instantiate particle variables
        if (pad_x == (0, 0)) and (pad_y == (0, 0)):

            # new method using Silvan's framework
            x0, y0, w0, h0 = self.bbox
            orig_template = image[y: y + h, x: x + w]
            # set the template
            self._template = image[y: y + h, x: x + w]
            #  [y - 1: y + h - 1, x - 1: x + w - 1] [y + 1: y + h + 1, x + 1: x + w + 1]

            # set mask on template
            self._mask_on_template = self.mask_on_image[y: y + h, x: x + w]

            # set particle center location on template
            self._location_on_template = (self.location[0] - x, self.location[1] - y)

            # define the contour in template coordinates
            contr = np.squeeze(self.contour)
            self._template_contour = np.array([contr[:, 0] - x, contr[:, 1] - y]).T

            # check if template is all nans
            array_nans = np.isnan(self.template)
            count_nans = np.sum(array_nans)
            if count_nans == self.template.size:
                j = 1

            """
            # get location of contour center and the contour bounding box side length
            cx, cy = self.location
            bbox_radius = int(np.ceil(w0 / 2))

            xc = np.floor(w/2)
            yc = np.floor(h/2)
            ynew = cy - bbox_radius
            yold = y
            yhnew = cy + bbox_radius
            yhold = y + h

            # these are the corner coordinates of the bounding box
            yl, yr, xl, xr = (cy - bbox_radius, cy + bbox_radius, cx - bbox_radius, cx + bbox_radius)
            # NOTE: these should be a duplicate of: (x0, y0, x0 + w0, y0 + h0)

            # apply ellipsoid mask
            # create an ellipsoid mask w/ radius of bbox width
            mask_radius = int(np.floor(w0 / 2))
            mask = binary_mask(mask_radius, image.ndim).astype(np.uint8)
            mask = binary_dilation(mask).astype(np.uint8)  # dilate the mask by a single pixel

            # set values outside ellipse mask to the minimum template value (not used; needs more testing)
            if apply_ellipsoid_mask:
                template = mask * template
                template_minimum_value = np.min(template)
                template = np.where(template == 0, template_minimum_value, template)
            

            # set the template
            self._template = image[yl:yr, xl:xr]

            # set mask on template
            self._mask_on_template = self.mask_on_image[yl:yr, xl:xr]

            # set particle center location on template
            self._location_on_template = (cx - xl, cy - yl)

            # define the contour in template coordinates
            contr = np.squeeze(self.contour)
            self._template_contour = np.array([contr[:, 0] - xl, contr[:, 1] - yl]).T
            """

            plot = False
            if plot:
                fig, ax = plt.subplots()
                ax.imshow(self.template)
                ax.scatter(self.location_on_template[0], self.location_on_template[1], marker='*', color='red')
                ax.set_title('pid{}: CENTER ON TEMPLATE = ({}, {}) \n This is the {} TEMPLATE'.format(self.id, self.location[0] - x, self.location[1] - y, adj_temp), fontsize=8)
                savedir = '/Users/mackenzie/Desktop/dumpfigures/particle_segmentation'
                plt.savefig(fname=savedir + '/pid{}_cx{}_cy{}_{}_template_rand{}.png'.format(self.id, self.location[0] - x, self.location[1] - y, adj_temp.lower(), np.random.randint(0, 500)))
                plt.close()
                plt.show()

            return self.template

        else:
            # from Silvan's original code
            tempy = image[y: y + h, x: x + w]

            template = np.pad(tempy.astype(np.float), (pad_y, pad_x),
                              'constant', constant_values=np.median(tempy))
            # changed from: "'constant', np.nan)" on 7/23/2022

            # the below are my additions
            # set the template
            self._template = template  # image[yl:yr, xl:xr]

            # set mask on template
            self._mask_on_template = np.pad(self.mask_on_image[y: y + h, x: x + w].astype(np.float), (pad_y, pad_x),
                              'constant', constant_values=0)  # changed from np.nan, 7/23/2022

            # set particle center location on template
            self._location_on_template = (np.shape(template)[0] // 2, np.shape(template)[1] // 2)

            # define the contour in template coordinates
            contr = np.squeeze(self.contour)
            self._template_contour = np.array([contr[:, 0] - x, contr[:, 1] - y]).T

            # check if template is all nans
            array_nans = np.isnan(self.template)
            count_nans = np.sum(array_nans)
            if count_nans == self.template.size:
                fig, ax = plt.subplots()
                ax.imshow(tempy)
                plt.show()

            return self.template

            # plot pretty Gaussian center finding
            """
            padded_image = np.pad(image[y: y + h, x: x + w].astype(np.float), (pad_y, pad_x),
                                    'constant', constant_values=np.min(image))

            # apply a binary circular mask to remove outer edge pixels that may disrupt the sub-pixel localization
            cx = int(np.round(x + w/2, 0))
            cy = int(np.round(y + h / 2, 0))
            contour_radius = int(np.round(np.sqrt(self.area / np.pi), 0) + 1)
            radius = padded_image.shape[0]
            # rect = [cy-r:cy+r+1, cs-radius:cx+radius+1]
            jj = padded_image.ndim
            mask = binary_mask(radius, padded_image.ndim).astype(np.uint8)
            j=1
            neighborhood = mask * padded_image[cy-radius:cy+radius+1, cx-radius:cx+radius+1]

            template = neighborhood
            
            
            template = np.pad(image[y: y + h, x: x + w].astype(np.float), (pad_y, pad_x), 'constant', constant_values=np.min(image))

            self.location_on_template = (self.location[1] - y, self.location[0] - x)

            # particle mask in the template
            #padded_mask_on_image = np.pad(self._mask_on_image[y: y + h, x: x + w].astype(np.float), (pad_y, pad_x), 'constant', constant_values=0)
            #self._mask_on_template =  mask * padded_mask_on_image[cy-radius:cy+radius+1, cx-radius:cx+radius+1]
            self._mask_on_template = np.pad(self._mask_on_image[y: y + h, x: x + w].astype(np.float), (pad_y, pad_x), 'constant', constant_values=0)

            fig, ax = plt.subplots()
            ax.imshow(self._mask_on_template)
            plt.show()

            # template contour
            contr = np.squeeze(self.contour)
            self.template_contour = np.array([contr[:, 0] - x, contr[:, 1] - y]).T

        
        if self.particle_collection_type == 'test':
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 7))

            ax1.imshow(image, interpolation='none')
            ax1.scatter(self.location[0], self.location[1], s=50, marker='*', color='red', alpha=0.5)
            ax1.set_title('image')

            ax2.imshow(template, interpolation='none')
            ax2.scatter(self.location_on_template[1], self.location_on_template[0], s=50, marker='*', color='red', alpha=0.5)
            ax2.set_title('template center (x, y) = ({}, {})'.format(self.location_on_template[1], self.location_on_template[0]))

            ax3.imshow(self._mask_on_template, interpolation='none')
            ax3.scatter(self.location_on_template[1], self.location_on_template[0], s=50, marker='*', color='red', alpha=0.5)
            ax3.axvline(x=self.location_on_template[1], color='red', alpha=0.25, linestyle='--')
            ax3.axhline(y=self.location_on_template[0], color='red', alpha=0.25, linestyle='--')
            ax3.set_title('mask on template')

            for ax in [ax2, ax3]:
                # Major ticks
                ax.set_xticks(np.arange(0, np.shape(self._mask_on_template)[1], 1))
                ax.set_yticks(np.arange(0, np.shape(self._mask_on_template)[0], 1))

                # Labels for major ticks
                ax.set_xticklabels(np.arange(0, np.shape(self._mask_on_template)[1], 1))
                ax.set_yticklabels(np.arange(0, np.shape(self._mask_on_template)[0], 1))

                # Minor ticks
                ax.set_xticks(np.arange(-0.5, np.shape(self._mask_on_template)[1], 1), minor=True)
                ax.set_yticks(np.arange(-0.5, np.shape(self._mask_on_template)[0], 1), minor=True)

                # Gridlines based on minor ticks
                ax.grid(which='minor', color='gray', alpha=0.25, linestyle='-', linewidth=1)

            savedir = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Results/Calibration/updated_meta_characterization'
            max_int = np.max(template)
            plt.savefig(fname=savedir + '/' + self.particle_collection_type + '_col' + '_max-intensity-' + str(max_int) + '_cX' + str(self.location_on_template[0]) + '_cX' + str(self.location_on_template[1]) + '.png')
            #plt.show()
        """

    def estimate_noise(self, I):

        H, W = I.shape

        M = [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]]

        sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
        sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W - 2) * (H - 2))

        return sigma


    def _fit_2D_gaussian(self, normalize=True):

        dia_x, dia_y, A, yc, xc, sigmay, sigmax = gaussian.fit_gaussian_calc_diameter(self._template, normalize=normalize)

        if dia_x is not None:
            self.gauss_dia_x = dia_x
            self.gauss_dia_y = dia_y
            self.gauss_A = A
            self.gauss_yc = yc + self.bbox[1]
            self.gauss_xc = xc + self.bbox[0]
            self.gauss_sigma_y = sigmay
            self.gauss_sigma_x = sigmax

            # plot
            """# gauss_2d_function(xy, a, x0, y0, sigmax, sigmay)
            popt = [A, yc, xc, sigmay, sigmax]
            img = self._template

            X = np.arange(np.shape(img)[1])
            Y = np.arange(np.shape(img)[0])
            X, Y = np.meshgrid(X, Y)

            # flatten arrays
            Xf = X.flatten()
            Yf = Y.flatten()
            Zf = img.flatten()

            # stack for gaussian curve fitting
            XYZ = np.stack([Xf.flatten(), Yf.flatten(), Zf.flatten()]).T
            # popt, pcov = curve_fit(gauss_2d_function, XYZ[:, :2], XYZ[:, 2], p0=guess, bounds=bounds)

            imgf = gauss_2d_function(XYZ[:, :2], *popt)
            imf = np.reshape(imgf, img.shape).T

            # generate ellipse (sigma x y)
            rr, cc = ellipse_perimeter(int(popt[2]), int(popt[1]), int(np.ceil(popt[4] / 2)), int(np.ceil(popt[3] / 2)))
            rr, cc = filter_ellipse_points_on_image(imf, rr, cc)
            imf[rr, cc] = 1

            # diameter (beta = 3.67)
            rr, cc = ellipse_perimeter(int(popt[1]), int(popt[2]), int(dia_x / 2), int(dia_y / 2))
            rr, cc = filter_ellipse_points_on_image(imf, rr, cc)
            # imf[rr, cc] = np.max(imf)

            fig, (ax1, ax3) = plt.subplots(ncols=2, figsize=(12, 6))
            ax1.imshow(img)
            ax1.set_title('Original (max Int = {})'.format(np.max(img)))
            ax3.imshow(imf)
            ax3.set_title('dia_x={}, dia_y = {}'.format(np.round(dia_x, 1), np.round(dia_y, 1)))
            plt.show()
            j = 1"""


    def _compute_center_subpixel(self, method='centroid', save_plots=False, ax=25, ay=25, A=500, fx=1, fy=1):

        if method == 'centroid':

            # modify the image
            particle_image = self.template
            padding = np.min([int(np.shape(particle_image)[0] / 3), 10])
            raw_image = np.pad(particle_image, pad_width=padding, mode='minimum')
            shape = raw_image.shape
            ndim = len(shape)

            # modify the diameter
            diameter = int(self._diameter * 0.95)
            if diameter % 2 == 0:
                diameter += 1

            diameter = validate_tuple(diameter, ndim)
            diameter = tuple([int(x) for x in diameter])
            radius = tuple([x // 2 for x in diameter])
            separation = tuple([x + 1 for x in diameter])
            smoothing_size = diameter  # size of sides of the square kernel in boxcar (rolling average) smoothing
            noise_size = 1  # width of gaussian blurring kernel
            noise_size = validate_tuple(noise_size, ndim)
            threshold = 1  # clip bandpass result below this value. Thresholding is done on background subtracted image.
            percentile = 95  # features must have peak brighter than pixels in this percentile.
            max_iterations = 10 # maximum iterations to find center

            # Convolve with a Gaussian to remove short-wavelength noise and subtract out long-wavelength variations by
            # subtracting a running average. This retains features of intermediate scale.
            image = bandpass(image=raw_image, lshort=noise_size, llong=smoothing_size, threshold=threshold, truncate=4)

            margin = tuple([max(rad, sep // 2 - 1, sm // 2) for (rad, sep, sm) in zip(radius, separation, smoothing_size)])

            # Find local maxima whose brightness is above a given percentile.
            coords = grey_dilation(image, separation, percentile, margin, precise=True)

            if coords.size > 2:
                raise ValueError("Multiple local maxima found. Need to adjust settings")

            refined_coords = refine_coords_via_centroid(raw_image=raw_image, image=image, radius=radius, coords=coords,
                                                        max_iterations=max_iterations, show_plot=False)

            x_refined = refined_coords[0][1] - padding
            y_refined = refined_coords[0][0] - padding
            mass = refined_coords[0][2]

            self._fitted_centroid_on_template = {'x': x_refined, 'y': y_refined, 'mass': mass}

        elif method == 'gaussian':

            # guess parameters (x0, y0, ax, ay, A) and resize scaling (fx, fy) with interpolation
            # NOTE: the ([1], [0]) ordering is because they are passed into array coordinates where (columns x rows)
            guess_prms = [(self.location_on_template[1], self.location_on_template[0], ax, ay, A)]
            # NOTE: location on template is in plotting coordinates.

            # fit to particle image template
            fit_image = self.template

            # perform fitting
            fit_image, popt, pcov, X, Y, rms, padding = fit_gaussian_subpixel(image=fit_image, guess_params=guess_prms, fx=fx, fy=fy)

            x_refined = popt[0]
            y_refined = popt[1]
            alphax = popt[2]
            alphay = popt[3]
            amplitude = popt[4]

            # instantiate fitted gaussian parameters
            # NOTE: saving the fitted coordinates in the plotting coordinate system.
            self._fitted_gaussian_on_template = {'x0': x_refined, 'y0': y_refined, 'ax': alphax, 'ay': alphay, 'A': amplitude}
            self._fitted_gaussian_rms = rms

        else:
            raise ValueError("{} method is not implemented for subpixel localization".format(method))

        # set the location of the subpixel coordinates
        self._location_subpixel = (x_refined + self.bbox[0], y_refined + self.bbox[1])
        location_subpixel = (x_refined + self.bbox[0], y_refined + self.bbox[1])
        location = self.location

        # calculate difference in particle centers
        xc_oldd, yc_oldd = self.location
        xc_new_floatt = x_refined + self.bbox[0]
        yc_new_floatt = y_refined + self.bbox[1]
        xc_old, yc_old = self.location_on_template
        xc_new_float = x_refined
        yc_new_float = y_refined
        xdist = np.abs(xc_new_float - xc_old)
        ydist = np.abs(yc_new_float - yc_old)


        # if difference is small, take Gaussian fitting, else take original centroid center.
        lb = 0.75  # lower bound for resizing the bbox
        ub = 2.4  # upper bound for discarding subpixel center finding

        if xdist < lb and ydist < lb:
            good_fit = 'green'
            # print('Good: RMS = {}'.format(rms))

        elif (lb < xdist < ub or lb < ydist < ub) and (xdist < ub and ydist < ub):
            xc_new = int(np.round(xc_new_float, 0) + self.bbox[0])
            yc_new = int(np.round(yc_new_float, 0) + self.bbox[1])

            # set new center location on image
            self._set_location((xc_new, yc_new))

            if save_plots is True:
                fig, ax = plt.subplots()
                ax.imshow(self.template)
                ax.scatter(xc_old, yc_old, s=50, color='black', marker='+', alpha=0.75, label='original center')
                ax.scatter(x_refined, y_refined, s=50, color='red', marker='*', label='subpixel center')
                ax.set_title('Original Template and Center on Template')
                ax.legend()
                savedir = '/Users/mackenzie/Desktop/dumpfigures'
                savename = 'Template_Original_pid{}_{}col_x{}_y{}_rand{}.png'.format(self.id,
                                                                                     self.particle_collection_type,
                                                                                     np.round(self.location[0], 2),
                                                                                     np.round(self.location[1], 2),
                                                                                     np.random.randint(0, 500))
                plt.tight_layout()
                plt.savefig(fname=savedir + '/' + savename)
                plt.close()

            # old bounding box
            old_bbox = self.bbox

            # re-set bounding box
            new_bbox = self._resized_bbox(resize=(self.template.shape[1], self.template.shape[0]))
            self._create_template(bbox=new_bbox, adjusted_template=True)

            if save_plots is True:
                fig, ax = plt.subplots()
                ax.imshow(self.template)
                ax.scatter(xc_old, yc_old, s=50, color='black', marker='+', alpha=0.75, label='original center')
                ax.scatter(self.location_on_template[0], self.location_on_template[1], s=50, color='red', marker='*', label='subpixel center')
                ax.set_title('Adjusted Template and Center on Template')
                ax.legend()
                savedir = '/Users/mackenzie/Desktop/dumpfigures'
                savename = 'Template_Adjusted_pid{}_{}col_x{}_y{}_rand{}.png'.format(self.id,
                                                                                         self.particle_collection_type,
                                                                                         np.round(self.location[0], 2),
                                                                                         np.round(self.location[1], 2),
                                                                                         np.random.randint(0, 500))
                plt.tight_layout()
                plt.savefig(fname=savedir + '/' + savename)
                plt.close()

            # recalculate contour stats
            self._compute_convex_hull()

            good_fit = 'magenta'
            #print('Good: RMS = {}'.format(rms))

        else:
            good_fit = 'red'
            #print('Bad: RMS = {}'.format(rms))

        # plot Gaussian contours
        if save_plots is True:

            if method == 'centroid':

                # plot the original image
                fig, ax = plt.subplots()
                ax.imshow(self._image_raw)
                ax.scatter(self._location_subpixel[0], self._location_subpixel[1], s=20, marker='.', color=good_fit, alpha=0.95,
                           label='pxcyc')
                ax.legend(fontsize=10, bbox_to_anchor=(1, 1), loc='upper left')
                plt.suptitle('pid{}(xc, yc) = ({}, {}) in {} collection'.format(self.id, self.location[0], self.location[1], self.particle_collection_type))
                savedir = '/Users/mackenzie/Desktop/dumpfigures'
                savename = 'Centroid_fit_full_img_pid{}_{}col_x{}_y{}_rand{}.png'.format(self.id, self.particle_collection_type,
                                                                           np.round(self.location[0], 2), np.round(self.location[1], 2), np.random.randint(0, 200))
                plt.tight_layout()
                plt.savefig(fname=savedir + '/' + savename)
                plt.close()
                # plt.show()

                # plot centroid-found center on template
                fig = plot_2D_image_and_center(self, good_fit=good_fit)
                plt.scatter(xc_old, yc_old, s=25, color='black', marker='o', label='old center')
                plt.suptitle(r'$p_{ID}$' + '= {} in {} collection'.format(self.id, self.particle_collection_type))
                savedir = '/Users/mackenzie/Desktop/dumpfigures'  # TODO: update plotting function so it's in Gdpyt.plotting and not here
                savename = 'Centroid_fit_{}_col_x{}_y{}_mass{}.png'.format(self.particle_collection_type,
                                                                           np.round(x_refined, 2), np.round(y_refined, 2),
                                                                           int(np.round(self._fitted_centroid_on_template['mass'], 0)))
                ax.legend(fontsize=10, bbox_to_anchor=(1, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(fname=savedir + '/' + savename)
                plt.close()
                # plt.show()

            elif method == 'gaussian':

                # plot the original image
                fig, ax = plt.subplots()
                ax.imshow(self._image_raw)
                ax.scatter(self.location[0], self.location[1], s=100, marker='*', color=good_fit, alpha=0.5,
                           label='pxcyc')
                ax.legend(fontsize=10, bbox_to_anchor=(1, 1), loc='upper left')
                plt.suptitle(r'$p_{ID}(xc, yc)$' + '= ({}, {}) in {} collection'.format(self.location[0], self.location[1], self.particle_collection_type))
                savedir = '/Users/mackenzie/Desktop/dumpfigures'
                savename = 'Gauss_fit_on_full_image_{}_col_x{}_y{}.png'.format(self.particle_collection_type,
                                                                           np.round(popt[0], 2), np.round(popt[1], 2))
                plt.tight_layout()
                plt.savefig(fname=savedir + '/' + savename)
                plt.close()
                # plt.show()

                # plot Gaussian contours for sigma = 1 and sigma = 2 after Gaussian centering
                fig = plot_2D_image_contours(self, X, Y, good_fit=good_fit, pad=padding)
                plt.scatter(xc_old, yc_old, color='black', marker='o')
                plt.suptitle(r'$p_{ID}$' + '= {} in {} collection'.format(self.id, self.particle_collection_type))
                savedir = '/Users/mackenzie/Desktop/dumpfigures' # TODO: update plotting function so it's in Gdpyt.plotting and not here
                savename = 'Gauss_fit_{}_col_x{}_y{}_ax{}_ay{}.png'.format(self.particle_collection_type, np.round(popt[0], 2), np.round(popt[1], 2), np.round(popt[2], 1), np.round(popt[3], 1))
                plt.tight_layout()
                plt.savefig(fname=savedir + '/' + savename)
                plt.close()
                # plt.show()

                # plot particle contour after Gaussian blurring
                fig, ax = plt.subplots()
                ax.imshow(fit_image)
                xgc = x_refined+padding
                ygc = y_refined+padding
                ax.scatter(xgc, ygc, s=25, marker='.', color='black', alpha=0.95, label='pxcyc')
                ax.axvline(x=xgc, color='black', alpha=0.35, linestyle='--')
                ax.axhline(y=ygc, color='black', alpha=0.35, linestyle='--')

                # plot ellipse on image with ax and ay principal diameters
                sigmas = [1, 2]
                for sigma in sigmas:
                    if sigma * 0.7 * alphax < float(np.shape(fit_image)[0]) and sigma * 0.7 * alphay < float(
                            np.shape(fit_image)[1]):
                        ellipse = Ellipse(xy=(xgc, ygc), width=alphax * sigma, height=alphay * sigma, fill=False,
                                          color='black', alpha=0.75 / sigma, label=r'$\sigma_{x,y}$' +
                                                                                   str(sigma) + '=({}, {})'.format(
                                np.round(sigma * alphax, 1), np.round(sigma * alphay, 1)))
                        ax.add_patch(ellipse)
                ax.legend(fontsize=10, bbox_to_anchor=(1, 1), loc='upper left')
                plt.suptitle(r'$p_{ID}$(xc, yc)' + '= {}, {}'.format(xgc, y_refined+padding))
                savename = 'Contours_Gaussian_{}_col_x{}_y{}_ax{}_ay{}.png'.format(self.particle_collection_type, np.round(popt[0], 2), np.round(popt[1], 2), np.round(popt[2], 1), np.round(popt[3], 1))
                plt.tight_layout()
                plt.savefig(fname=savedir + '/' + savename)
                plt.close()

    def compute_local_snr(self):

        img_f = self.template
        img_f_bkg = img_f.copy()
        background_mask = self.mask_on_template

        # get mean intensity of template
        mean_intensity_of_template_pixels = np.mean(img_f)

        # get peak intensity
        peak_intensity = np.max(img_f)

        # get variance
        variance = np.var(img_f)
        variance_squared = variance**2
        variance_squared_norm = (variance / np.mean(img_f))**2

        # apply background mask to get background
        img_f_mask_inv = ma.masked_array(img_f, mask=self.mask_on_template)

        # apply particle mask to get signal
        particle_mask = np.logical_not(background_mask)
        img_f_mask = ma.masked_array(img_f_bkg, mask=particle_mask)

        # calculate SNR for filtered image
        mean_signal_f = img_f_mask.mean()
        variance_squared_norm_signal = (img_f_mask.var() / img_f_mask.mean())**2
        mean_background_f = img_f_mask_inv.mean()
        std_background_f = img_f_mask_inv.std()

        # minimum meaningful std for noise-less (synthetic) images
        if std_background_f < 1:
            std_background_f = 1

        snr_filtered = (mean_signal_f - mean_background_f) / std_background_f

        # maximum snr for noise-less (synthetic) images
        if snr_filtered > 1000:
            snr_filtered = 1000

        # store particle image statistics
        self._peak_intensity = peak_intensity
        self.int_var = variance
        self.int_var_sq = variance_squared
        self.int_var_sq_norm = variance_squared_norm
        self.int_var_sq_norm_signal = variance_squared_norm_signal
        self._snr = snr_filtered
        self._mean_signal = mean_signal_f
        self._mean_background = mean_background_f
        self._std_background = std_background_f
        self._mean_intensity_of_template = mean_intensity_of_template_pixels

    def _dilated_bbox(self, dilation=None, dims=None):
        if dims is None:
            w, h = self.bbox[2], self.bbox[3]
        else:
            w, h = dims
        if dilation is None:
            return self.bbox
        elif isinstance(dilation, tuple):
            assert len(dilation) == 2
            dil_x, dil_y = dilation
        elif isinstance(dilation, float) or isinstance(dilation, int):
            dil_x = dilation
            dil_y = dilation
        else:
            raise TypeError("Wrong type for dilation (Received {})".format(type(dilation)))

        wl, ht = int(w * dil_x / 2), int(h * dil_y / 2)
        top_corner = np.array(self.location).astype(int) - np.array([wl, ht])
        dilated_bbox = (top_corner[0], top_corner[1], int(w * dil_x), int(h * dil_y))
        return dilated_bbox

    def _resized_bbox(self, resize=None):
        """

        Variables in method:
            w: the width of the bounding box. Should be an odd number where the middle value is the center.
            h: the height of the bounding box. Should be an odd number where the middle value is the center.
            wl: the width of the bounding box. Should be an odd number where the center is located at floor(w/2)
            ht: the height of the bounding box. Should be an odd number where the center is located at floor (h/2)
        """

        if resize is None:
            return self.bbox
        else:
            w, h = resize
            wl, ht = int(np.floor(w / 2)), int(np.floor(h / 2))
            top_corner = np.array(self.location).astype(int) - np.array([wl, ht])
            return top_corner[0], top_corner[1], w, h

    def resize_bbox(self, w, h):
        """
        Adjust bounding box to size w x h and adjust the center to the center of the contour
        :param w: new width (int)
        :param h: new height (int)
        :return:
        """
        self._bbox = self._resized_bbox(resize=(w, h))
        self._create_template()

    def get_template(self, dilation=None, resize=None):
        if dilation is None and resize is None:
            return self._create_template()
        elif dilation is not None and resize is None:
            dil_bbox = self._dilated_bbox(dilation=dilation)
            return self._create_template(bbox=dil_bbox)
        elif dilation is None and resize is not None:
            resized_bbox = self._resized_bbox(resize)
            return self._create_template(bbox=resized_bbox)
        else:
            resized_bbox = self._resized_bbox(resize=resize)
            dil_bbox = self._dilated_bbox(dilation=dilation, dims=resized_bbox[2:])
            return self._create_template(bbox=dil_bbox)

    def reset_id(self, new_id):
        assert isinstance(new_id, int)
        #logger.warning("Particle ID {}: Reset ID to {}".format(self.id, new_id))
        self._id = new_id

    def set_interpolation_curve(self, z, sim, label_suffix=None):
        assert len(z) == len(sim)
        columns = ['z', 'S_{}'.format(label_suffix.upper())]
        self._interpolation_curve = pd.DataFrame({columns[0]: z, columns[1]: sim})

    def set_similarity_curve(self, z, sim, label_suffix=None):
        assert len(z) == len(sim)
        columns = ['z', 'S_{}'.format(label_suffix.upper())]
        self._similarity_curve = pd.DataFrame({columns[0]: z, columns[1]: sim})

    def set_particle_to_particle_spacing(self, ptps):
        """
        ptps = [mean_dx_all, min_dx_all, num_dx_all, mean_dxo, num_dxo]

        mean_dx_all: mean distance to {num_dx_all} number of particles
        min_dx_all: minimum distance to another particle
        num_dx_all: number of particles considered as "all"
        mean_dxo: mean distance to "overlapping" particles (where distance < mean(gauss_dia_x, gauss_dia_y))
        num_dxo: number of "overlapping" particles
        percent_dx_diameter: percent of diameter that particle image is overlapped = (diameter - min_dx) / diameter
        """
        self.mean_dx = ptps[0]
        self.min_dx = ptps[1]
        self.num_dx = ptps[2]
        self.mean_dxo = ptps[3]
        self.num_dxo = ptps[4]
        self.percent_dx_diameter = ptps[5]

    def _set_location(self, location):
        assert len(location) == 2
        self._location = location

    def set_match_location(self, match_loc):
        self.match_location = (match_loc[0], match_loc[1])

    def set_match_localization(self, match_loc):
        self.match_localization = (match_loc[0], match_loc[1])

    def _set_location_true(self, x, y, z=None):
        self._x_true = x
        self._y_true = y
        if z:
            self._z_true = z

    def set_z(self, z):
        assert isinstance(z, float)
        self._z = z
        # The value originally received is stored in a separate argument
        if self._z_default is None:
            self._z_default = z

    def set_true_z(self, z):
        if self._z_true is None:
            assert isinstance(z, float)
            self._z_true = z

    def _set_true_z(self, z):
        self._z_true = z

    def set_in_focus_z(self, z):
        assert isinstance(z, float)
        self.in_focus_z = z

    def set_in_focus_area(self, area):
        assert isinstance(area, float)
        self.in_focus_area = area

    def set_in_focus_intensity(self, intensity):
        self.in_focus_intensity = intensity

    def set_inference_stack_id(self, stack):
        self.inference_stack_id = stack

    def set_cm(self, c_measured):
        assert isinstance(c_measured, float)
        self._cm = c_measured

    def set_id(self, id_):
        self._id = id_

    def set_max_sim(self, sim):
        self._max_sim = sim

    def set_use_raw(self, use_raw):
        assert isinstance(use_raw, bool)
        self._use_raw = use_raw

    @property
    def use_raw(self):
        return self._use_raw

    @property
    def area(self):
        return self._area

    @property
    def diameter(self):
        return self._diameter

    @property
    def aspect_ratio(self):
        return self._aspect_ratio

    @property
    def thinness_ratio(self):
        return self._thinness_ratio

    @property
    def hull_area(self):
        return self._hull_area

    @property
    def bbox(self):
        """
        x_int = int(np.round(self._bbox[0], 0))
        y_int = int(np.round(self._bbox[1], 0))
        bbox = (x_int, y_int, self._bbox[2], self._bbox[3])
        """
        bbox = self._bbox
        xmax = bbox[0] + bbox[2]
        ymax = bbox[1] + bbox[3]
        xmin = bbox[0]
        ymin = bbox[1]

        return self._bbox

    @property
    def contour(self):
        return self._contour

    @property
    def hull(self):
        return self._hull

    @property
    def id(self):
        return self._id

    @property
    def interpolation_curve(self):
        return self._interpolation_curve

    @property
    def in_images(self):
        return self._in_images

    @property
    def location(self):
        """
        Notes: the location is in index-array coordinates. Meaning, the furthest "left" or "top" value can be 0.
        """
        return self._location

    @property
    def mask_on_image(self):
        """
        Notes: the mask_on_image array (Nx x Ny) is in array coordinates (i.e. Nx IS the columns of the array).
        So, if you wanted to plot the mask_on_image array using imshow(), you would need to modify the location_on_template
        coordinates in order to get the location coordinates and mask coordinates into the same coordinate system.
        """
        return self._mask_on_image

    @property
    def image_raw(self):
        return self._image_raw

    @property
    def template(self):
        return self._template

    @property
    def location_on_template(self):
        """
        Important Notes:
            * the location_on_template tuple (x, y) is in plotting coordinates (so you can scatter plot).
            * plotting coordinates means if the x-location on template was 10, then the x-location on the template
            array would be the 11th index of the columns.
            * b/c the template should always have an odd-numbered side length, this means the location on template
            should always be an odd number.

            For example:
                template x-shape = 47
                template x-indices = 0 : 46
                x-location template = 23
        """
        return self._location_on_template

    @property
    def mask_on_template(self):
        return self._mask_on_template

    @property
    def template_contour(self):
        return self._template_contour

    @property
    def x_true(self):
        return self._x_true

    @property
    def y_true(self):
        return self._y_true

    @property
    def z_true(self):
        return self._z_true

    @property
    def cm(self):
        return self._cm

    @property
    def max_sim(self):
        return self._max_sim

    @property
    def solidity(self):
        return self._solidity

    @property
    def snr(self):
        return self._snr

    @property
    def peak_intensity(self):
        return self._peak_intensity

    @property
    def mean_signal(self):
        return self._mean_signal

    @property
    def mean_background(self):
        return self._mean_background

    @property
    def std_background(self):
        return self._std_background

    @property
    def mean_intensity_of_template(self):
        return self._mean_intensity_of_template

    @property
    def similarity_curve(self):
        return self._similarity_curve

    @property
    def z(self):
        return self._z

    @property
    def true_num_particles(self):
        return self._true_num_particles

    @property
    def particle_collection_type(self):
        return self._particle_collection_type

    @property
    def fitted_gaussian_on_template(self):
        return self._fitted_gaussian_on_template

    @property
    def fitted_gaussian_rms(self):
        return self._fitted_gaussian_rms