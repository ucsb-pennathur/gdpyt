import cv2

class GdpytParticle(object):

    def __init__(self, image, id, template, contour, bbox):
        self._id = id
        self._image = image
        self._template = template
        self._contour = contour
        self._bbox = bbox
        self._compute_center()
        self._compute_convex_hull()
        #location are the x,y coordinates of the particle. eg [45,65] or (45, 67)
        # The assert statement will raise an error if this is not a length two iterable

    # This sets the height
    #
    def set_z(self, z):
        assert isinstance(z, float)
        self._z = z

    def set_id(self, id):
        self._id = id

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

    def _set_location(self, location):
        assert len(location) == 2
        self._location = location

    @property
    def id(self):
        return self._id

    @property
    def template(self):
        return self._template

    @property
    def contour(self):
        return self._contour

    @property
    def hull(self):
        return self._hull

    @property
    def area(self):
        return self._hull

    @property
    def solidity(self):
        return self._solidity

    @property
    def location(self):
        return self._location