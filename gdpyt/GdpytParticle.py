import cv2

import numpy as np
import numpy.ma as ma

from skimage.draw import rectangle_perimeter

import pandas as pd


from matplotlib import pyplot as plt

from .particle_identification import apply_threshold

class GdpytParticle(object):

    def __init__(self, image_raw, image_filt, id_, contour, bbox, thresh_specs=None):
        super(GdpytParticle, self).__init__()
        self._id = id_
        assert isinstance(image_raw, np.ndarray)
        assert isinstance(image_filt, np.ndarray)
        self._image_raw = image_raw
        self._image_filt = image_filt
        self._contour = contour
        self._bbox = bbox
        self._compute_center()
        self._compute_convex_hull()
        self._similarity_curve = None
        self._interpolation_curve = None
        self._z = None
        self._z_default = None
        self._max_sim = None
        self._use_raw = True

        if thresh_specs is not None:
            self._compute_local_snr(thresh_specs=thresh_specs)

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

    def _create_template(self, bbox=None):
        if self.use_raw:
            image = self._image_raw
        else:
            image = self._image_filt

        if bbox is None:
            x0, y0, w0, h0 = self._bbox
            x, y, w, h = self._bbox
        else:
            x0, y0, w0, h0 = bbox
            x, y, w, h = bbox
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

        if (pad_x == (0, 0)) and (pad_y == (0, 0)):
            template = image[y: y + h, x: x + w]
            jjj = np.squeeze(self.contour)
            self.template_contour = np.array([jjj[:,0]-y, jjj[:,1]-x]).T
        else:
            template = np.pad(image[y: y + h, x: x + w].astype(np.float), (pad_y, pad_x),
                                    'constant', constant_values=np.nan)

        """        
        fig, ax = plt.subplots(ncols=2)
        rr, cc = rectangle_perimeter(start=(x, y), end=( x + w, y + h), shape=self._image_raw.shape)
        img = np.zeros_like(self._image_raw, dtype=np.uint16)
        img[rr, cc] = 2**15
        ax[0].imshow(self._image_raw, cmap='gray', alpha=0.95)
        ax[0].imshow(img, cmap='Reds', alpha=0.5)
        jj = np.squeeze(self.contour)
        ax[0].plot(jj[:,0], jj[:,1], color='blue', linewidth=1)
        ax[1].imshow(template, cmap='gray')
        ax[1].plot(template_contour[:, 1], template_contour[:, 0], color='blue', linewidth=1)
        ax[0].set_title('hahahah')
        plt.show()"""

        assert template.shape == (h0, w0)
        return template

    def _compute_convex_hull(self):
        hull = cv2.convexHull(self.contour)
        self._hull = hull
        self._area = float(cv2.contourArea(hull))
        self._solidity = cv2.contourArea(self.contour) / self._area

    def _compute_center(self):
        M = cv2.moments(self._contour)

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        self._set_location((cX, cY))

    def _compute_local_snr(self, thresh_specs):
        #mask = np.zeros_like(self._image_raw)
        #cv2.drawContours(mask, self.contour, -1, 255, -1)
        #mean_p_raw = self._image_raw[mask != 0].mean()
        #bckr_raw = self._image_raw[self.bbox[1]:self.bbox[1] + self.bbox[3], self.bbox[0]

        img_f = self.template
        img_f_bkg = img_f.copy()

        particle_mask = apply_threshold(img_f, parameter=thresh_specs).astype(np.uint8)
        background_mask = particle_mask.astype(bool)

        # apply background mask to get background
        img_f_mask_inv = ma.masked_array(img_f, mask=particle_mask)

        # apply particle mask to get signal
        particle_mask = np.logical_not(background_mask).astype(bool)
        img_f_mask = ma.masked_array(img_f_bkg, mask=particle_mask)

        # calculate SNR for filtered image
        mean_signal_f = img_f_mask.mean()
        mean_background_f = img_f_mask_inv.mean()
        std_background_f = img_f_mask_inv.std()
        snr_filtered = (mean_signal_f - mean_background_f) / std_background_f

        self._snr = snr_filtered



    def _dilated_bbox(self, dilation=None, dims=None):
        if dims is None:
            w, h = self.bbox[2], self.bbox[3]
        else:
            w, h = dims
        if dilation is None:
            return self._bbox
        elif isinstance(dilation, tuple):
            assert len(dilation) == 2
            dil_x, dil_y = dilation
        elif isinstance(dilation, float) or isinstance(dilation, int):
            dil_x = dilation
            dil_y = dilation
        else:
            raise TypeError("Wrong type for dilation (Received {})".format(type(dilation)))

        wl, ht = int(w * dil_x / 2), int(h * dil_y / 2)
        top_corner = np.array(self.location) - np.array([wl, ht])
        dilated_bbox = (top_corner[0], top_corner[1], int(w * dil_x), int(h * dil_y))
        return dilated_bbox

    def _resized_bbox(self, resize=None):
        if resize is None:
            return self._bbox
        else:
            w, h = resize
            wl, ht = int(w / 2), int(h / 2)
            top_corner = np.array(self.location) - np.array([wl, ht])
            return top_corner[0], top_corner[1], w, h

    def _set_location(self, location):
        assert len(location) == 2
        self._location = location

    def _set_location_true(self, x, y, z):
        self._x_true = x
        self._y_true = y
        self._z_true = z

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

    def resize_bbox(self, w, h):
        """
        Adjust bounding box to size w x h and adjust the center to the center of the contour
        :param w: new width (int)
        :param h: new height (int)
        :return:
        """
        self._bbox = self._resized_bbox(resize=(w, h))

    def set_interpolation_curve(self, z, sim, label_suffix=None):
        assert len(z) == len(sim)
        columns = ['z', 'S_{}'.format(label_suffix.upper())]
        self._interpolation_curve = pd.DataFrame({columns[0]: z, columns[1]: sim})

    def set_similarity_curve(self, z, sim, label_suffix=None):
        assert len(z) == len(sim)
        columns = ['z', 'S_{}'.format(label_suffix.upper())]
        self._similarity_curve = pd.DataFrame({columns[0]: z, columns[1]: sim})

    def set_z(self, z):
        assert isinstance(z, float)
        self._z = z
        # The value originally received is stored in a separate argument
        if self._z_default is None:
            self._z_default = z

    def set_cm(self, c_measured):
        assert isinstance(c_measured, float)
        self._cm = c_measured

    def set_id(self, id_):
        self._id = id_

    def set_max_sim(self, sim):
        self._max_sim = sim

    def use_raw(self, use_raw):
        assert isinstance(use_raw, bool)
        self._use_raw = use_raw

    @property
    def area(self):
        return self._area

    @property
    def bbox(self):
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
    def location(self):
        return self._location

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
    def similarity_curve(self):
        return self._similarity_curve

    @property
    def template(self):
        return self._create_template()

    @property
    def z(self):
        return self._z