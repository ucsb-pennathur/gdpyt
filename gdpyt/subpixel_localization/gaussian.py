"""
This method fits a 2D Gaussian to a 2D image (numpy array).

Link to original article:

A good reference for PSF-based z-determination: https://link.springer.com/content/pdf/10.1007/s00348-014-1809-2.pdf
"""

# import modules
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# define 2D Gaussian
def gaussian2D(x, y, x0, y0, xalpha, yalpha, A):
    """
    x: x-coordinate of pixel
    y: y-coordinate of pixel
    x0: x-location of peak
    y0: y-location of peak
    xalpha: x-size of principal axis
    yalpha: y-size of principal axis
    A: amplitude
    """
    return A * np.exp(-((x-x0)/xalpha)**2 -((y-y0)/yalpha)**2)

# define the callable that is passed to curve_fit
def _gaussian2D(M, *args):
    """
    gaussian: function to fit
    M: (2, N) array where N is the total number of data points in the image (which will be raveled to 1D)
    args:
    """
    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//5):
       arr += gaussian2D(x, y, *args[i*5:i*5+5])
    return arr

def fit(image, guess_params, fx=1, fy=1, print_results=True):
    """
    image: array to fit
    guess_params: guessed Gaussian distribution parameters (x0, y0, xalpha, yalpha, A)
    _gaussian: helper function to read raveled ordering of data points
    fx: x-resolution of fitted domain space
    fy: y-resolution of fitted domain space
    """
    # flatten the guess parameter list
    p0 = [p for params in guess_params for p in params]

    # determine the 2D domain extents and resolution of the fit
    xdim = np.shape(image)[0]
    ydim = np.shape(image)[1]
    xmin, xmax, nx = 1, xdim, xdim * fx
    ymin, ymax, ny = 1, ydim, ydim * fy

    # create the 2D domain of the fit
    x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    # ravel the meshgrids of X, Y points to a pair of 1D arrays
    xdata = np.vstack((X.ravel(), Y.ravel()))

    # fit custom _gaussian2D function which understands the raveled ordering of the data points
    popt, pcov = curve_fit(_gaussian2D, xdata, image.ravel(), p0)

    return popt, pcov, X, Y

def fit_results(image, X, Y, popt):
    """
    image: image to fit
    X: x-domain of fit
    Y: y-domain of fit
    gaussian: fitting function
    popt: fitted parameters
    """
    fitted = np.zeros(image.shape)
    for i in range(len(popt) // 5):
        fitted += gaussian2D(X, Y, *popt[i*5:i*5+5])

    print('Fitted parameters (x0, y0, xalpha, yalpha, A):')
    print(popt)

    rms = np.sqrt(np.mean((image - fitted) ** 2))
    print('RMS residual =', rms)

    return fitted, rms

def plot_2D_image(image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='viridis', origin='lower')
    return fig

def plot_3D_image(X, Y, image):
    fig = plt.figure()
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(X, Y, image, cmap='viridis')
    ax.set_zlim(0, np.max(image) + 2)
    return fig

def plot_3D_fit(X, Y, image, fit):
    fig = plt.figure()
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(X, Y, fit, cmap='viridis')
    cset = ax.contourf(X, Y, image - fit, zdir='z', offset=-4, cmap='viridis')
    ax.set_zlim(-4, np.max(fit))
    return fig

def plot_2D_image_contours(image, fitted, X, Y):
    x = X[0, :]
    y = Y[:, 0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='viridis', origin='lower',
              extent=(x.min(), x.max(), y.min(), y.max()))
    ax.contour(X, Y, fitted, colors='w')
    return fig