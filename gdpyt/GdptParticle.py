

class GdpytParticle(object):

    def __init__(self, id, template, location, z=None):
        self._id = id
        self._template = template

        #location are the x,y coordinates of the particle. eg [45,65] or (45, 67)
        # The assert statement will raise an error if this is not a length two iterable
        assert len(location) == 2
        self._location = location

        if z is not None:
            self.set_z(z)
        else:
            self._z = None

    # This sets the height
    #
    def set_z(self, z):
        assert isinstance(z, float)
        self._z = z

    @property
    def id(self):
        return self._id

    @property
    def template(self):
        return self._template