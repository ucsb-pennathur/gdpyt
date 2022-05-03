
import math
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit, minimize
import functools


def calculate_z_of_3d_plane(x, y, popt):
    """
    Calculate the z-coordinate of a point lying on a 3D plane.
    """

    a, b, c, d, normal = popt[0], popt[1], popt[2], popt[3], popt[4]

    z = (-normal[0] * x - normal[1] * y - d) * 1. / normal[2]

    return z


def fit_3d_plane(points):

    fun = functools.partial(error, points=points)
    params0 = np.array([0, 0, 0])
    res = minimize(fun, params0)

    a = res.x[0]
    b = res.x[1]
    c = res.x[2]

    point = np.array([0.0, 0.0, c])
    normal = np.array(cross([1, 0, a], [0, 1, b]))
    d = -point.dot(normal)

    popt = [a, b, c, d, normal]

    minx = np.min(points[:, 0])
    miny = np.min(points[:, 1])
    maxx = np.max(points[:, 0])
    maxy = np.max(points[:, 1])

    xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    return xx, yy, z, popt


def error(params, points):
    result = 0
    for (x, y, z) in points:
        plane_z = plane(x, y, params)
        diff = abs(plane_z - z)
        result += diff**2
    return result


def plane(x, y, params):
    a = params[0]
    b = params[1]
    c = params[2]
    z = a*x + b*y + c
    return z


def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]