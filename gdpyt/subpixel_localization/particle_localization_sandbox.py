"""
Sandbox for testing sub pixel localization methods

Other links:
https://scipython.com/blog/non-linear-least-squares-fitting-of-a-two-dimensional-data/
https://lmfit.github.io/lmfit-py/examples/example_two_dimensional_peak.html

"""

import numpy as np

# best link: https://scipython.com/blog/non-linear-least-squares-fitting-of-a-two-dimensional-data/
def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                        + c*((y-yo)**2)))
    return g.ravel()