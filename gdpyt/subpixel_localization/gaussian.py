"""
This method fits a 2D Gaussian to a 2D image (numpy array).

Link to original article:

A good reference for PSF-based z-determination: https://link.springer.com/content/pdf/10.1007/s00348-014-1809-2.pdf
"""

# import modules
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
from skimage.transform import resize, downscale_local_mean, rescale
from skimage.filters import gaussian
from matplotlib.patches import Ellipse

from skimage.draw import ellipse_perimeter
from gdpyt.utils.plotting import filter_ellipse_points_on_image
from gdpyt.utils.gdpyt_analysis import evaluate_fit_gaussian_and_plot_3d

"""
Note: the following scripts are new (2022 and later)
"""

# ---

# ---------------------------------------- GAUSSIAN MODELS -------------------------------------------------------------


def gauss_1d_function(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def gauss_2d_function(xy, a, x0, y0, sigmax, sigmay):
    return a * np.exp(-((xy[:, 0] - x0) ** 2 / (2 * sigmax ** 2) + (xy[:, 1] - y0) ** 2 / (2 * sigmay ** 2)))


def bivariate_gaussian_pdf(xy, a, x0, y0, sigmax, sigmay, rho):
    return a * np.exp(
        -((1 / (2 * (1 - rho ** 2))) * ((xy[:, 0] - x0) ** 2 / sigmax ** 2 - 2 * rho * (xy[:, 0] - x0) * (
                    xy[:, 1] - y0) / (sigmax * sigmay) +
          (xy[:, 1] - y0) ** 2 / sigmay ** 2)
          )
    )


def bivariate_gaussian_pdf_bkg(xy, a, x0, y0, sigmax, sigmay, rho, bkg):
    return a * np.exp(
        -((1 / (2 * (1 - rho ** 2))) * ((xy[:, 0] - x0) ** 2 / sigmax ** 2 - 2 * rho * (xy[:, 0] - x0) * (
                    xy[:, 1] - y0) / (sigmax * sigmay) +
          (xy[:, 1] - y0) ** 2 / sigmay ** 2)
          )
    ) + bkg

def bivariate_gaussian_pdf_bkgg(xy, a, x0, y0, sigmax, sigmay, rho, bkg):
    return a * np.exp(
        -((1 / (2 * (1 - rho ** 2))) * ((xy[:, 0] - x0) ** 2 / sigmax ** 2 - 2 * rho * (xy[:, 0] - x0) * (
                    xy[:, 1] - y0) / (sigmax * sigmay) +
          (xy[:, 1] - y0) ** 2 / sigmay ** 2)
          )
    ) + bkg


# ---

# -------------------------------------------- FITTING FUNCTIONS -------------------------------------------------------


def fit_2d_gaussian_on_image(img, normalize=True, guess='sigma_improved', rotate_degrees=0, bivariate_pdf=False):
    if rotate_degrees != 0:
        img = rotate(img, angle=rotate_degrees, reshape=False, mode='grid-constant', cval=np.percentile(img, 5))

    if normalize:
        img = img - img.min() + 1

    y, x = np.shape(img)
    xc, yc = x // 2, y // 2

    bounds = None
    if guess == 'center':
        guess_A = np.max(img) / 2
        guess_sigma = xc / 2
        bounds = ([0, x / 8, y / 8, 0, 0], [2 ** 16, 7 * x / 8, 7 * y / 8, x, y])
    elif guess == 'sigma':
        guess_A, guess_c, guess_sigma = get_amplitude_center_sigma(x_space=None, y_profile=None, img=img, y_slice=yc)
    elif guess == 'sigma_improved':
        guess_A, guess_c, guess_sigma = get_amplitude_center_sigma_improved(x_space=None, y_profile=None, img=img)

    # make grid
    X = np.arange(np.shape(img)[1])
    Y = np.arange(np.shape(img)[0])
    X, Y = np.meshgrid(X, Y)

    # flatten arrays
    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = img.flatten()

    # stack for gaussian curve fitting
    XYZ = np.stack([Xf.flatten(), Yf.flatten(), Zf.flatten()]).T

    # fit 2D gaussian
    guess = [guess_A, xc, yc, guess_sigma, guess_sigma]

    try:
        if bivariate_pdf is True and normalize is False:
            guess = [guess_A, xc, yc, guess_sigma, guess_sigma, 0, 100]
            try:
                popt, pcov = curve_fit(bivariate_gaussian_pdf_bkg, XYZ[:, :2], XYZ[:, 2],
                                       guess,
                                       bounds=([0, 0, 0, 0, 0, -0.99, 0],
                                               [2**16, 512, 512, 100, 100, 0.99, 2**16])
                                       )
            except ValueError:
                j = 1
            # NOTE: the below function is used to drop the background intensity 'bkg' from the results
            # popt = popt[:-1]

            # NOTE: on 11/21/22, I am changing the script to include the 'bkg' in order to calculate the rmse per fit.

        elif bivariate_pdf:
            guess = [guess_A, xc, yc, guess_sigma, guess_sigma, 0]
            popt, pcov = curve_fit(bivariate_gaussian_pdf, XYZ[:, :2], XYZ[:, 2], guess)
        elif bounds is not None:
            popt, pcov = curve_fit(gauss_2d_function, XYZ[:, :2], XYZ[:, 2], p0=guess, bounds=bounds)
        else:
            popt, pcov = curve_fit(gauss_2d_function, XYZ[:, :2], XYZ[:, 2], guess)
    except RuntimeError:
        popt = None

    """
    if popt is not None:
        # experimental
        y_slice = np.unravel_index(np.argmax(img, axis=None), img.shape)[0]
        y_profile = img[y_slice, :]
        x_space = np.arange(len(y_profile))

        # Gaussian
        imgf = gauss_2d_function(XYZ[:, :2], *popt)
        imf = np.reshape(imgf, img.shape)

        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax1 = ax[0, 0]
        ax2 = ax[0, 1]
        ax3 = ax[1, 0]
        ax4 = ax[1, 1]

        ax1.imshow(img)
        ax1.set_title('Normalized (max={})'.format(np.round(np.max(img), 1)))

        ax2.imshow(imf)
        ax2.set_title('A={}, wx={}, wy={}'.format(np.round(popt[0], 1), np.round(popt[3], 1), np.round(popt[4], 1)))

        ax3.plot(x_space, y_profile)
        ax3.set_ylabel('Img Intensity')

        x_gauss = np.linspace(-popt[4] * 2, len(y_profile) + popt[4] * 2, 200)
        y_gauss = gauss_1d_function(x_gauss, popt[0], popt[1], popt[3])
        ax4.plot(x_gauss, y_gauss)
        ax4.set_ylabel('1D Gauss')

        plt.tight_layout()
        plt.show()
        j = 1
    """

    return popt


def fit_2d_gaussian_on_ccorr(res, xc, yc):
    # xc_original = xc
    # yc_original = yc

    scaling_factor = 1
    # res = rescale(res, scaling_factor)

    # make grid
    X = np.arange(np.shape(res)[1])
    Y = np.arange(np.shape(res)[0])
    X, Y = np.meshgrid(X, Y)

    # flatten arrays
    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = res.flatten()

    # stack for gaussian curve fitting
    XYZ = np.stack([Xf.flatten(), Yf.flatten(), Zf.flatten()]).T

    # fit 2D gaussian
    guess = [1, xc * scaling_factor, yc * scaling_factor, 1.5 * scaling_factor, 1.5 * scaling_factor]

    try:
        popt, pcov = curve_fit(gauss_2d_function, XYZ[:, :2], XYZ[:, 2], p0=guess)
        A, xc, yc, sigmax, sigmay = popt


        """
        popt, img_norm = processing.fit_2d_gaussian_on_image(img, normalize=True, bkg_mean=bkg_mean)
        A, xc, yc, sigmax, sigmay = popt

        # calculate the fit error
        XYZ, fZ, rmse, r_squared, residuals = processing.evaluate_fit_2d_gaussian_on_image(img_norm, popt)
        """

        # xc = xc / scaling_factor
        # yc = yc / scaling_factor

        # 3D plot similarity map and 2D Gaussian fit
        # evaluate_fit_gaussian_and_plot_3d(res, popt, scaling_factor)
        # print("Original ({}, {}); Fitted ({}, {})".format(xc_original, yc_original, np.round(xc, 3), np.round(yc, 3)))

    except RuntimeError:
        xc, yc = None, None

    return xc, yc


def fit_gaussian_calc_diameter(img, normalize=True, rotate_degrees=0, bivariate_pdf=False):
    popt = fit_2d_gaussian_on_image(img, normalize=normalize, guess='sigma_improved',
                                    rotate_degrees=rotate_degrees, bivariate_pdf=bivariate_pdf)

    if popt is None:
        return None, None, None, None, None, None, None, None, None
    elif len(popt) == 5:
        popt = np.append(popt, None)
        popt = np.append(popt, None)

    A, xc, yc, sigmax, sigmay, rho, bkg = popt

    dia_x, dia_y = calc_diameter_from_theory(img, A, xc, yc, sigmax, sigmay)
    # dia_x, dia_y = calc_diameter_from_pixel_intensities(img, A, xc, yc, sigmax, sigmay)

    return dia_x, dia_y, A, yc, xc, sigmay, sigmax, rho, bkg


# ---

# --------------------------------------------- EVALUATION FUNCTIONS ---------------------------------------------------

def evaluate_fit_2d_gaussian_on_image(img, fit_func, popt):
    """ XYZ, fZ, rmse, r_squared, residuals = evaluate_fit_2d_gaussian_on_image(img, fit_func, popt) """

    XYZ = flatten_image(img)

    # 2D Gaussian from fit
    if fit_func == 'bivariate_pdf':
        fZ = bivariate_gaussian_pdf_bkg(XYZ[:, :2], *popt)
    else:
        raise ValueError("Currently only 'bivariate_pdf' is implemented.")

    # data used for fitting
    img_arr = XYZ[:, 2]

    # get residuals
    residuals = calculate_residuals(fit_results=fZ, data_fit_to=img_arr)

    # get goodness of fit
    rmse, r_squared = calculate_fit_error(fit_results=fZ, data_fit_to=img_arr)

    return XYZ, fZ, rmse, r_squared, residuals


# ---

# ----------------------------------------- HELPER EVALUATION FUNCTIONS ------------------------------------------------


def calculate_residuals(fit_results, data_fit_to):
    residuals = fit_results - data_fit_to
    return residuals


def calculate_fit_error(fit_results, data_fit_to, fit_func=None, fit_params=None, data_fit_on=None):
    """
    See 'gdpyt-analysis' for more details.
    """

    # --- calculate prediction errors
    if fit_results is None:
        fit_results = fit_func(data_fit_on, *fit_params)

    residuals = calculate_residuals(fit_results, data_fit_to)

    se = np.square(residuals)  # squared errors
    mse = np.mean(se)  # mean squared errors
    rmse = np.sqrt(mse)  # Root Mean Squared Error, RMSE
    r_squared = 1.0 - (np.var(np.abs(residuals)) / np.var(data_fit_to))

    return rmse, r_squared


def flatten_image(img):
    """ XYZ = flatten_image(img) """

    # make grid
    X = np.arange(np.shape(img)[1])
    Y = np.arange(np.shape(img)[0])
    X, Y = np.meshgrid(X, Y)

    # flatten arrays
    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = img.flatten()

    # stack for gaussian curve fitting
    XYZ = np.stack([Xf.flatten(), Yf.flatten(), Zf.flatten()]).T

    return XYZ


# ---

# --------------------------------------------- HELPER FUNCTIONS -------------------------------------------------------


def get_amplitude_center_sigma_improved(x_space=None, y_profile=None, img=None):
    """ raw_amplitude, raw_c, raw_sigma = get_amplitude_center_sigma_improved(x_space=None, y_profile=None, img=None) """
    if y_profile is None:
        y_slice = np.unravel_index(np.argmax(img, axis=None), img.shape)[0]
        y_profile = img[y_slice, :]

    if x_space is None:
        x_space = np.arange(len(y_profile))

    # get amplitude
    raw_amplitude = y_profile.max() - y_profile.min()

    # get center
    raw_c = x_space[np.argmax(y_profile)]

    # get sigma
    y_pl_zero = len(np.where(y_profile[:np.argmax(y_profile)] - np.mean(y_profile) < 0)[0])
    y_pr_zero = len(np.where(y_profile[np.argmax(y_profile):] - np.mean(y_profile) < 0)[0])

    """y_pl_zero = np.where(y_profile[:np.argmax(y_profile)] - np.mean(y_profile) < 0)[0][0]
    y_pr_zero = np.where(y_profile[np.argmax(y_profile):] - np.mean(y_profile) < 0)[0][0]"""
    raw_sigma = np.mean([y_pl_zero, y_pr_zero])

    return raw_amplitude, raw_c, raw_sigma


def get_amplitude_center_sigma(x_space=None, y_profile=None, img=None, y_slice=None):
    if y_profile is None:
        # get sub-image slice
        y_profile = img[y_slice, :]

    if x_space is None:
        x_space = np.arange(len(y_profile))

    # get amplitude
    raw_amplitude = y_profile.max() - y_profile.min()

    # get center
    raw_c = x_space[np.argmax(y_profile)]

    # get sigma
    raw_profile_diff = np.diff(y_profile)
    diff_peaks = np.argpartition(np.abs(raw_profile_diff), -2)[-2:]
    diff_width = np.abs(x_space[diff_peaks[1]] - x_space[diff_peaks[0]])
    raw_sigma = diff_width / 2

    return raw_amplitude, raw_c, raw_sigma


def calc_diameter_from_theory(img, A, xc, yc, sigmax, sigmay):
    beta = np.sqrt(3.67)

    # diameter threshold: intensity < np.exp(-3.67 ** 2)
    diameter_threshold = np.exp(-1 * beta ** 2) * A

    # spatial arrays
    x_arr = np.linspace(0, sigmax * 5, 1000)
    y_arr = np.linspace(0, sigmay * 5, 1000)

    # xy-radius intensity distribution (fitted Gaussian distribution)
    x_intensity = gauss_1d_function(x=x_arr, a=A, x0=0, sigma=sigmax)
    y_intensity = gauss_1d_function(x=y_arr, a=A, x0=0, sigma=sigmay)

    # find where intensity distribution, I_xy(x, y) < np.exp(-3.67 ** 2) * maximum intensity at the center
    # NOTE: is "maximum intensity at the center" defined as A (fitted Gaussian amplitude) or peak pixel intensity?
    x_intensity_raw = x_intensity - np.exp(-1 * beta ** 2) * A
    y_intensity_raw = y_intensity - np.exp(-1 * beta ** 2) * A
    x_intensity_rel = np.abs(x_intensity_raw)
    y_intensity_rel = np.abs(y_intensity_raw)

    # radius (in pixels) is equal to xy-coordinate where xy_intensity_rel is minimized
    radius_x = x_arr[np.argmin(x_intensity_rel)]
    radius_y = y_arr[np.argmin(y_intensity_rel)]

    """
    Important Note:
    You --DO NOT-- multiply the radius by two.
        > This is because this particular location on a Gaussian distribution corresponds to the location on an 
        Airy distribution that would be the radius. The location on the Airy distribution would be multiplied by 2
        because it is the radius. However, for a Gaussian distribution, this point defines the diameter of the Airy disk
    """
    dia_x = radius_x
    dia_y = radius_y

    # experimental slice
    """
    y_slice_exp = np.unravel_index(np.argmax(img, axis=None), img.shape)[0]
    y_profile_exp = img[y_slice_exp, :]
    x_space_exp = np.arange(len(y_profile_exp))

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    ax1.plot(x_space_exp, y_profile_exp)
    ax1.set_xlabel('image pixels')
    ax1.set_ylabel('image intensity')
    ax2.plot(x_arr, x_intensity, label='dx={}'.format(np.round(dia_x, 3)))
    ax2.axhline(diameter_threshold, color='black', label='Threshold={}'.format(np.round(diameter_threshold, 1)))
    ax2.set_xlabel('resampled pixels')
    ax2.set_ylabel('Gaussian intensity')
    ax2.legend()
    plt.show()
    
    j = 1
    """

    return dia_x, dia_y


def calc_diameter_from_pixel_intensities(img, A, xc, yc, sigmax, sigmay):
    beta = np.sqrt(3.67)

    # diameter threshold: intensity < np.exp(-3.67 ** 2)
    diameter_threshold = np.exp(-1 * beta ** 2) * A

    # Fit x-diameter
    # resample and plot fitted Gaussian
    yslice = np.arange(0, np.shape(img)[1])
    fit_yslice = np.linspace(-yslice.max() * 2, yslice.max() * 3, len(yslice) * 10)
    fit_yint = gauss_1d_function(fit_yslice, A, xc, sigmax)

    # calculate width
    arr = np.abs(fit_yint - diameter_threshold)
    fit_widths = fit_yslice[np.argpartition(arr, 2)[:2]]
    dia_x = np.abs(fit_widths[1] - fit_widths[0])

    # Fit y-diameter
    # resample and plot fitted Gaussian
    xslice = np.arange(0, np.shape(img)[0])
    fit_xslice = np.linspace(-xslice.max() * 2, xslice.max() * 3, len(xslice) * 10)
    fit_xint = gauss_1d_function(fit_xslice, A, yc, sigmay)

    # calculate width
    arr = np.abs(fit_xint - diameter_threshold)
    fit_widths = fit_xslice[np.argpartition(arr, 2)[:2]]
    dia_y = np.abs(fit_widths[1] - fit_widths[0])

    # experimental slice
    y_slice_exp = np.unravel_index(np.argmax(img, axis=None), img.shape)[0]
    y_profile_exp = img[y_slice_exp, :]
    x_space_exp = np.arange(len(y_profile_exp))

    """fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    ax1.plot(x_space_exp, y_profile_exp)
    ax1.set_xlabel('image pixels')
    ax1.set_ylabel('image intensity')
    ax2.plot(fit_yslice, fit_yint, label='dx={}'.format(dia_x, 1))
    ax2.axhline(diameter_threshold, color='black', label='Threshold={}'.format(np.round(diameter_threshold, 1)))
    ax2.set_xlabel('resampled pixels')
    ax2.set_ylabel('Gaussian intensity')
    ax2.legend()
    plt.show()"""

    return dia_x, dia_y


def calculate_maximum_of_fitted_gaussian_1d(x, y, normalize=True, show_plot=False, ylabels=None):
    if isinstance(y, list):
        y_list = y
    else:
        y_list = [y]

    if show_plot:
        if len(y_list) > 1:
            fig, ax = plt.subplots(nrows=len(y_list), sharex=True)
        else:
            fig, ax = plt.subplots()

    x_maxs = []
    for i, y in enumerate(y_list):

        if normalize:
            y = y - y.min()

        # initialize fitting parameters
        guess_amplitude, guess_c, guess_sigma = get_amplitude_center_sigma(x_space=x, y_profile=y)
        guess_params = [guess_amplitude, guess_c, guess_sigma]

        # fit
        try:
            popt, pcov = curve_fit(gauss_1d_function, x, y, p0=guess_params)
        except RuntimeError:
            fig, ax = plt.subplots()
            ax.scatter(x, y)
            ax.set_ylabel(ylabels[i])
            ax.set_title('Failed Gauss 1D fit on {}'.format(ylabels[i]))
            plt.show()
            popt = None

        if popt is None:
            x_maxs.append(np.nan)
            continue

        xc = popt[1]
        x_maxs.append(xc)

        # resample
        x_fit = np.linspace(x.min(), x.max(), len(x) * 10)
        y_fit = gauss_1d_function(x_fit, *popt)

        # find x-value at maximum of fitted 1D Gaussian
        x_max = x_fit[np.argmax(y_fit)]

        # OPTIONAL: plot
        if show_plot:
            ax[i].scatter(x, y, color='black')
            ax[i].plot(x_fit, y_fit, color='tab:blue')
            ax[i].scatter(x_max, np.max(y_fit), marker='*', color='red')
            ax[i].set_xlabel('z')
            if ylabels:
                ax[i].set_ylabel(ylabels[i])
            else:
                ax[i].set_ylabel('y')

    if show_plot:
        plt.show()

    return x_maxs


def calculate_minimum_of_fitted_gaussian_diameter(x, y, fit_function, guess_x0, show_plot=False):
    guess_c1, guess_c2 = 0.15, 0.65

    if isinstance(y, list):
        y_list = y
    else:
        y_list = [y]

    if show_plot:
        fig, ax = plt.subplots()
        data_colors = ['blue', 'lime']
        fit_colors = ['tab:blue', 'tab:green']

    x0_c1_c2s = []
    dia_zfs = []
    dia_zmins = []
    dia_zmaxs = []
    for i, y in enumerate(y_list):

        # fit diameter function
        try:
            popt, pcov = curve_fit(fit_function,
                                   x,
                                   y,
                                   p0=[guess_x0, guess_c1, guess_c2],
                                   bounds=([x.min(), 0, 0], [x.max(), 1, 1])
                                   )
        except RuntimeError:
            continue

        x0, c1, c2 = popt[0], popt[1], popt[2]
        x0_c1_c2s.append([x0, c1, c2])

        # resample
        x_fit = np.linspace(x.min(), x.max(), len(x) * 25)
        y_fit = fit_function(x_fit, *popt)

        # calculate diameter @ z-in-focus, z-min, z-max
        dia_zfs.append(np.min(y_fit))
        dia_zmins.append(y_fit[0])
        dia_zmaxs.append(y_fit[-1])

        if show_plot:
            ax.scatter(x, y, color=data_colors[i], label=y.name)
            ax.plot(x_fit, y_fit, color=fit_colors[i])
            ax.scatter(x_fit[np.argmin(y_fit)], np.min(y_fit), marker='*', color='red')

    if show_plot:
        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel('Diameter (pixels)')
        ax.legend()
        plt.tight_layout()
        plt.show()

    dia_zfs = np.array(dia_zfs)
    dia_zmins = np.array(dia_zmins)
    dia_zmaxs = np.array(dia_zmaxs)

    return x0_c1_c2s, dia_zfs, dia_zmins, dia_zmaxs


"""
NOTE: the below script are all old (2021 and earlier)
"""


# define 1D Gaussian
def gaussian1D(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


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
    return A * np.exp(-((x - x0) / xalpha) ** 2 - ((y - y0) / yalpha) ** 2)


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
    for i in range(len(args) // 5):
        arr += gaussian2D(x, y, *args[i * 5:i * 5 + 5])
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
        fitted += gaussian2D(X, Y, *popt[i * 5:i * 5 + 5])
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
                                                                               '(Ampl.={})'.format(
                                                                                   int(np.round(A, -1))))
        ax.axvline(x=xc, color=good_fit, alpha=0.35, linestyle='--')
        ax.axhline(y=yc, color=good_fit, alpha=0.35, linestyle='--')

        # plot ellipse on image with ax and ay principal diameters
        sigmas = [1, 2]
        for sigma in sigmas:
            if sigma * 0.7 * alphax < float(np.shape(image)[0]) and sigma * 0.7 * ay < float(np.shape(image)[1]):
                ellipse = Ellipse(xy=(xc, yc), width=alphax * sigma, height=ay * sigma, fill=False,
                                  color='black', alpha=0.75 / sigma, label=r'$\sigma_{x,y}$' +
                                                                           str(sigma) + '=({}, {})'.format(
                        np.round(sigma * alphax, 1), np.round(sigma * ay, 1)))
                ax.add_patch(ellipse)

        ax.set_title(r'$p_{xc, yc}$' + '(sub-pix) = ({}, {}) \n center = {}, {}'.format(particle.location[0],
                                                                                        particle.location[1],
                                                                                        np.round(xc, 2),
                                                                                        np.round(yc, 2)))

    ax.set_xlim([0, image.shape[0] - 0.5])
    ax.set_ylim([0, image.shape[1] - 0.5])
    ax.legend(fontsize=10, bbox_to_anchor=(1, 1), loc='upper left', )

    # Major ticks
    ax.set_xticks(np.arange(0, image.shape[0] + 1, 2))
    ax.set_yticks(np.arange(0, image.shape[1] + 1, 2))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(0, image.shape[0] + 1, 2))
    ax.set_yticklabels(np.arange(0, image.shape[0] + 1, 2))

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