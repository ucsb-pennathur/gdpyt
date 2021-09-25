"""
This method fits a 2D Gaussian to a 2D image (numpy array).

Link to original article:

A good reference for PSF-based z-determination: https://link.springer.com/content/pdf/10.1007/s00348-014-1809-2.pdf
"""

# import modules
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from skimage.transform import resize, downscale_local_mean, rescale
from skimage.filters import gaussian
from matplotlib.patches import Ellipse

# define 1D Gaussian
def gaussian1D(x, a, x0, sigma):
    return a*np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

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

    Notes:
        * the "*args[i*5:i*5+5]" portion allows you to pass in a list of guess parameters.
    """
    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//5):
       arr += gaussian2D(x, y, *args[i*5:i*5+5])
    return arr

def fit(image, guess_params, fx=1, fy=1, pad=4):
    """
    image: array to fit
        NOTE: the image is upscaled so that it has different dimensions than the input image.
    guess_params: guessed Gaussian distribution parameters (x0, y0, xalpha, yalpha, A)
    _gaussian: helper function to read raveled ordering of data points
    fx: x-resolution of fitted domain space (should be an integer)
    fy: y-resolution of fitted domain space (should be an integer)
    """
    assert fx == fy

    # flatten the guess parameter list
    p0 = [p * 2 + pad for params in guess_params for p in params]

    # store original image
    original_image = image.copy()

    # pad the input image with the image minimum value to provide a larger fitting space. Note, the edge value could
    # be extended in the padding but the minimum image value was chosen to reduce image stretching.
    if pad != 0:
        image = np.pad(image, pad_width=pad, constant_values=np.min(image))

    # determine the 2D domain extents and resolution of the fit
    xdim = np.shape(image)[0]
    ydim = np.shape(image)[1]
    xmin, xmax, nx = 0, xdim, xdim * fx
    ymin, ymax, ny = 0, ydim, ydim * fy

    # create the 2D domain of the fit
    x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    # ravel the meshgrids of X, Y points to a pair of 1D arrays
    xdata = np.vstack((X.ravel(), Y.ravel()))

    # resize and interpolate the image to fit the mesh grid shape
    if fx > 1:
        image = rescale(image, scale=fx, preserve_range=True)

    # apply a Gaussian smoothing function to the image
    image = gaussian(image, sigma=1, preserve_range=True)

    # fit custom _gaussian2D function which understands the raveled ordering of the data points
    popt, pcov = curve_fit(_gaussian2D, xdata, image.ravel(), p0)

    # fit results
    fitted, rms = fit_results(image, X, Y, popt)

    # adjust fitted center coordinates by padding
    popt = np.array([popt[1] - pad - 0.5, popt[0] - pad - 0.5, popt[3], popt[2], popt[4]])
    # NOTE: putting the "- 0.5" in because that seems to be better
    # NOTE: UPDATED 9/24 - getting rid of the "- 1" because it doesn't make any sense why it should be here.
    # NOTE: putting the "- 1" back in
    # NOTE: the "- 1" cannot be in there because when it is, the plot on the actual fitted image is off by 1 pixel.
    # NOTE: putting the "- 1" back in
    # NOTE: UPDATED 9/24 - getting rid of the "- 1"
    # NOTE: I'm not exactly sure why the "- 1" is necessary but it makes the cv2.contourCenter and Gaussian fit
    # much more closely aligned so for now I'm going to assume it's an inherent offset.

    # take absolute value of a_x and a_y which can sometimes be negative
    popt = np.abs(popt)

    return image, popt, pcov, X, Y, rms, pad

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
    rms = np.sqrt(np.mean((image - fitted) ** 2))

    return fitted, rms

def plot_2D_image(image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='viridis', origin='lower')
    ax.scatter(7, 14, marker='X', color='blue')
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

def plot_2D_image_contours(particle, X, Y, guess_params=None, good_fit='black', pad=2):
    """
    particle=self.template, X, Y, popt, guess_prms, good_fit=good_fit, pad=padding
    Plot the particle image template, Gaussian located center, and principal axes of the fitted Gaussian

    Notes:
        * the Gaussian located center (xc = popt[1], yc = popt[0]) is in array coordinates and thus should always be
        odd numbered because the template has odd-numbered side lengths and the image is plotted with image array
        index[0] == 0 (i.e. image array coordinates).
        * the location_on_template attribute of GdpytParticle should always be even numbered because this parameter
        stores the particle location in plotting coordinates.
        * we WOULD need to convert xc, yc from index-array coordinates to plotting coordinates (by changing xc to xc =
        popt[1] + 1 BUT because we are plotting using ax.imshow(..., extent(x.min(), ...)) we do not need to convert.
    Parameters
    ----------
    image
    X
    Y
    popt
    guess_params

    Returns
    -------

    """
    image = particle.template
    popt = particle.fitted_gaussian_on_template
    xc = particle._location_subpixel[0] - particle.bbox[0]
    yc = particle._location_subpixel[1] - particle.bbox[1]
    alphax = popt['ax']
    ay = popt['ay']
    A = popt['A']

    x = X[0, :]
    y = Y[:, 0]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if popt is not None:

        # plot location of center
        ax.scatter(xc, yc, s=100, marker='*', color=good_fit, alpha=0.5, label=r'$p_{xc,yc}$' +
                                                                                      '(Ampl.={})'.format(int(np.round(A, -1))))
        ax.axvline(x=xc, color=good_fit, alpha=0.35, linestyle='--')
        ax.axhline(y=yc, color=good_fit, alpha=0.35, linestyle='--')

        # plot ellipse on image with ax and ay principal diameters
        sigmas = [1, 2]
        for sigma in sigmas:
            if sigma * 0.7 * alphax < float(np.shape(image)[0]) and sigma * 0.7 * ay < float(np.shape(image)[1]):
                ellipse = Ellipse(xy=(xc, yc), width=alphax*sigma, height=ay*sigma, fill=False,
                                  color='black', alpha=0.75/sigma, label=r'$\sigma_{x,y}$' +
                                  str(sigma)+'=({}, {})'.format(np.round(sigma * alphax, 1), np.round(sigma * ay, 1)))
                ax.add_patch(ellipse)

        ax.set_title(r'$p_{xc, yc}$' + '(sub-pix) = ({}, {}) \n center = {}, {}'.format(particle.location[0], particle.location[1], np.round(xc, 2), np.round(yc, 2)))

    ax.set_xlim([0, image.shape[0] - 0.5])
    ax.set_ylim([0, image.shape[1] - 0.5])
    ax.legend(fontsize=10, bbox_to_anchor=(1, 1), loc='upper left',)

    # Major ticks
    ax.set_xticks(np.arange(0, image.shape[0]+1, 2))
    ax.set_yticks(np.arange(0, image.shape[1]+1, 2))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(0, image.shape[0]+1, 2))
    ax.set_yticklabels(np.arange(0, image.shape[0]+1, 2))

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, image.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, image.shape[1], 1), minor=True)

    # invert the y-axis
    plt.gca().invert_yaxis()

    # show the image last
    ax.imshow(image, cmap='viridis', interpolation='none')

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='gray', alpha=0.125, linestyle='-', linewidth=1)
    # ax.grid(color='gray', alpha=0.125)

    return fig