import cv2
import numpy as np
import pandas as pd

class GdpytParticle(object):

    def __init__(self, image, id_, contour, bbox):
        super(GdpytParticle, self).__init__()
        self._id = id_
        assert isinstance(image, np.ndarray)
        self._image = image
        self._contour = contour
        self._bbox = bbox
        self._compute_center()
        self._compute_convex_hull()
        self._similarity_curve = None
        self._interpolation_curve = None
        self._z = None
        self._max_sim = None

    def __repr__(self):
        class_ = 'GdpytParticle'
        repr_dict = {'ID': self.id,
                     'Location': self.location,
                     'Bounding box dimensions': [self.bbox[2], self.bbox[3]],
                     'Area': self.area,
                     'Solidity': self.solidity,
                     'Z coordinate': self.z}
        out_str = "{}: {} \n".format(class_, self.id)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def _create_template(self, bbox=None):
        if bbox is None:
            x0, y0, w0, h0 = self._bbox
            x, y, w, h = self._bbox
        else:
            x0, y0, w0, h0 = bbox
            x, y, w, h = bbox
        pad_x_m, pad_x_p, pad_y_m, pad_y_p = 0, 0, 0, 0
        if y + h > self._image.shape[0]:
            pad_y_p = y + h - self._image.shape[0]
        if y < 0:
            pad_y_m = - y
            h = y + h
            y = 0
        if x + w > self._image.shape[1]:
            pad_x_p = x + w - self._image.shape[1]
        if x < 0:
            pad_x_m = - x
            w = x + w
            x = 0

        pad_x = (pad_x_m, pad_x_p)
        pad_y = (pad_y_m, pad_y_p)

        if (pad_x == (0, 0)) and (pad_y == (0, 0)):
            template = self._image[y: y + h, x: x + w]
        else:
            template = np.pad(self._image[y: y + h, x: x + w].astype(np.float), (pad_y, pad_x),
                                    'constant', constant_values=np.nan)
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

    def set_id(self, id_):
        self._id = id_

    def set_max_sim(self, sim):
        self._max_sim = sim

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
    def max_sim(self):
        return self._max_sim

    @property
    def solidity(self):
        return self._solidity

    @property
    def similarity_curve(self):
        return self._similarity_curve

    @property
    def template(self):
        return self._create_template()

    @property
    def z(self):
        return self._z