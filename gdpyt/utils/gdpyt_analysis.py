import numpy as np
import numpy.ma as ma

from scipy.optimize import curve_fit
from skimage.io import imread

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

"""plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)"""

# Evaluate Fit and Plot

def evaluate_fit_gaussian_and_plot_3d(img, popt, scaling_factor=1):

    # params
    A, xc, yc, sigmax, sigmay = popt
    xc = xc / scaling_factor
    yc = yc / scaling_factor

    # calculate the fit error
    XYZ, fZ, rmse, r_squared, residuals = evaluate_fit_2d_gaussian_on_image(img, popt)

    # reshape
    x, y, z, fz = reshape_flattened(img, XYZ, fZ)
    fz_residuals = np.reshape(residuals, np.shape(img))

    # fig, ax1 = plot_2d_corr_map_2d(z)
    # plt.savefig('/Users/mackenzie/Desktop/corr-map' + '/2d-corr-map_2D_residuals={}.svg'.format(np.round(np.mean(fz_residuals), 5)))
    # plt.close()

    """fig, ax1 = plot_2d_corr_map_3d(x, y, z)
    plt.savefig('/Users/mackenzie/Desktop/corr-map' + '/2d-corr-map_3D_residuals={}.svg'.format(np.round(np.mean(fz_residuals), 5)))
    plt.close()

    fig, (ax1, ax2) = plot_image_and_gaussian_3d(x, y, z, fz)
    plt.suptitle("x, y, rmse = {}, {}, {}".format(np.round(xc, 3), np.round(yc, 3), np.round(rmse, 3)))
    plt.savefig('/Users/mackenzie/Desktop/corr-map' + '/corr-map_residuals={}.svg'.format(np.round(np.mean(fz_residuals), 5)))
    plt.close()"""

# ---

# fit Gaussian

def gauss_2d_function(xy, a, x0, y0, sigmax, sigmay):
    return a * np.exp(-((xy[:, 0] - x0)**2 / (2 * sigmax**2) + (xy[:, 1] - y0)**2 / (2 * sigmay**2)))


def fit_2d_gaussian_on_image(img, normalize=True, bkg_mean=100):
    """ popt, img_norm = fit_2d_gaussian_on_image(img, normalize, bkg_mean) """

    img_norm, XYZ = flatten_image(img, normalize, bkg_mean)

    yc, xc = np.shape(img_norm)
    xc, yc = xc // 2, yc // 2

    guess_A, guess_c, guess_sigma = get_amplitude_center_sigma(x_space=None, y_profile=None, img=img_norm, y_slice=yc)

    # fit 2D gaussian
    guess = [guess_A, xc, yc, guess_sigma, guess_sigma]
    try:
        popt, pcov = curve_fit(gauss_2d_function, XYZ[:, :2], XYZ[:, 2], guess)
    except RuntimeError:
        popt = None

    return popt, img_norm


def evaluate_fit_2d_gaussian_on_image(img_norm, popt):
    """ XYZ, fZ, rmse, r_squared = evaluate_fit_2d_gaussian_on_image(img_norm, popt) """

    img_norm, XYZ = flatten_image(img_norm, normalize=False, bkg_mean=None)

    # 2D Gaussian from fit
    fZ = gauss_2d_function(XYZ[:, :2], *popt)

    # data used for fitting
    img_arr = XYZ[:, 2]

    # get residuals
    residuals = calculate_residuals(fit_results=fZ, data_fit_to=img_arr)

    # get goodness of fit
    rmse, r_squared = calculate_fit_error(fit_results=fZ, data_fit_to=img_arr)

    return XYZ, fZ, rmse, r_squared, residuals

# ---

# plotting

def plot_2d_corr_map_2d(z1):
    """ fig, ax1 = plot_2d_corr_map_2d(z1) """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # set up a figure twice as wide as it is tall
    #fig = plt.figure(figsize=plt.figaspect(0.5))
    #ax1 = fig.add_subplot(1, 1, 1)

    # set up the axes for the first plot
    fig, ax1 = plt.subplots(figsize=(size_x_inches * 0.7, size_y_inches * 0.6))

    im = ax1.imshow(z1 * 0.9, cmap=cm.coolwarm, vmin=0, vmax=1.0)
    ax1.set_xlabel(r'$X$')
    ax1.set_ylabel(r'$Y$')
    ax1.set_xticks(ticks=[0, 5, 10], labels=[r'$-\Delta L/2$', r'$X^{o}$', r'$\Delta L/2$'], minor=False)
    ax1.set_yticks(ticks=[0, 5, 10], labels=[r'$-\Delta L/2$', r'$Y^{o}$', r'$\Delta L/2$'], minor=False)
    # ax1.set_xticks(ticks=[0, np.shape(z1)[0] // 2, np.shape(z1)[0] - 1], labels=[r'$-\Delta L/2$', r'X^{o}$', r'$\Delta L/2$'], minor=False)
    # ax1.set_yticks(ticks=[0, np.shape(z1)[1] // 2, np.shape(z1)[1] - 1], labels=[r'$-\Delta L/2$', r'Y^{o}$', r'$\Delta L/2$'], minor=False)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im, cax=cax, ticks=[0, 1], label=r'$S$')
    # cbar.set_label(r'$S$')

    return fig, ax1

def plot_2d_corr_map_3d(x, y, z1):
    """ fig, ax1 = plot_2d_corr_map_3d(x, y, z1) """
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=(size_x_inches * 0.6, size_y_inches * 0.6))

    # set up the axes for the first plot
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    surf1 = ax1.plot_surface(x, y, z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax1.set_xlabel(r'$X$')
    ax1.set_ylabel(r'$Y$')
    ax1.set_zlabel(r'$S$')

    s = np.sqrt(len(z1))
    ax1.set_xticks(ticks=[0, 5, 10], labels=[r'$-\Delta L/2$', r'$X^{o}$', r'$\Delta L/2$'], minor=False)
    ax1.set_yticks(ticks=[0, 5, 10], labels=[r'$-\Delta L/2$', r'$Y^{o}$', r'$\Delta L/2$'], minor=False)
    # ax1.set_xticks(ticks=[0, s // 2, s], labels=[r'$-\Delta L/2$', r'X^{o}$', r'$\Delta L/2$'], minor=False)
    # ax1.set_yticks(ticks=[0, s // 2, s], labels=[r'$-\Delta L/2$', r'Y^{o}$', r'$\Delta L/2$'], minor=False)
    ax1.set_zticks(ticks=[0, 1], minor=False)

    # make the panes transparent
    ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax1.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax1.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax1.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    return fig, ax1

def plot_gaussian_3d(x, y, z):
    # Plot the surface
    fig, ax = plt.subplots(figsize=(size_x_inches * 0.6, size_y_inches * 0.6), subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel(r'$X$')
    ax.set_ylabel(r'$Y$')
    ax.set_zlabel(r'$S$')

    s = np.sqrt(len(z))
    ax.set_xticks(ticks=[0, 5, 10], labels=[r'$-\Delta L/2$', r'$X^{o}$', r'$\Delta L/2$'], minor=False)
    ax.set_yticks(ticks=[0, 5, 10], labels=[r'$-\Delta L/2$', r'$Y^{o}$', r'$\Delta L/2$'], minor=False)
    # ax.set_xticks(ticks=[0, s // 2, s], labels=[r'$-\Delta L/2$', r'X^{o}$', r'$\Delta L/2$'], minor=False)
    # ax.set_yticks(ticks=[0, s // 2, s], labels=[r'$-\Delta L/2$', r'Y^{o}$', r'$\Delta L/2$'], minor=False)
    ax.set_zticks(ticks=[0, 1], minor=False)

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    return fig, ax


def plot_image_and_gaussian_3d(x, y, z1, z2):
    """ fig, (ax1, ax2) = plot_image_and_gaussian_3d(x, y, z1, z2) """
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # set up the axes for the first plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(x, y, z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax1.set_xlabel(r'$X$')
    ax1.set_ylabel(r'$Y$')
    ax1.set_zlabel(r'$S$')

    s = np.sqrt(len(z1))
    # ax1.set_xticks(ticks=[0, s // 2, s], labels=[r'$-\Delta L/2$', r'X^{o}$', r'$\Delta L/2$'], minor=False)
    # ax1.set_yticks(ticks=[0, s // 2, s], labels=[r'$-\Delta L/2$', r'Y^{o}$', r'$\Delta L/2$'], minor=False)
    ax1.set_zticks(ticks=[0, 1], minor=False)

    # set up the axes for the second plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(x, y, z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax2.set_xlabel(r'$X$')
    ax2.set_ylabel(r'$Y$')
    ax2.set_zlabel(r'$S$')

    s = np.sqrt(len(z2))
    # ax2.set_xticks(ticks=[0, s // 2, s], labels=[r'$-\Delta L/2$', r'X^{o}$', r'$\Delta L/2$'], minor=False)
    # ax2.set_yticks(ticks=[0, s // 2, s], labels=[r'$-\Delta L/2$', r'Y^{o}$', r'$\Delta L/2$'], minor=False)
    ax2.set_zticks(ticks=[0, 1], minor=False)

    return fig, (ax1, ax2)


def plot_image_and_gaussian_2d(z1, z2):
    """ fig, (ax1, ax2) = plot_image_and_gaussian_2d(z1, z2) """

    fig, (ax1, ax2) = plt.subplots(figsize=(size_x_inches, size_y_inches), ncols=2)

    # first plot
    p1 = ax1.imshow(z1, cmap='magma', interpolation='none')
    cbar1 = fig.colorbar(p1, ax=ax1, extend='both', shrink=0.5)
    cbar1.minorticks_on()

    # Plot both positive and negative values between +/- 1.2
    p2 = ax2.imshow(z2, cmap='RdBu', interpolation='none')
    cbar2 = fig.colorbar(p2, ax=ax2, extend='both', shrink=0.5)
    cbar2.minorticks_on()

    plt.tight_layout()
    return fig, (ax1, ax2)

# ---


# helper functions

def reshape_flattened(img, XYZ, fZ):
    """ x, y, z, fz = reshape_flattened(img, XYZ, fZ) """
    x = np.reshape(XYZ[:, 0], np.shape(img))
    y = np.reshape(XYZ[:, 1], np.shape(img))
    z = np.reshape(XYZ[:, 2], np.shape(img))
    fz = np.reshape(fZ, np.shape(img))
    return x, y, z, fz


def flatten_image(img, normalize=True, bkg_mean=None):
    """ img_norm, XYZ = flatten_image(img, normalize=True, bkg_mean=None) """
    if normalize:
        img = normalize_image(img, bkg_mean)

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

    return img, XYZ

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

def get_background(image, threshold):
    particle_mask = image > threshold
    bkg = ma.masked_array(image, mask=particle_mask)
    return bkg


def normalize_image(img, bkg_mean):
    if bkg_mean is None:
        bkg_mean = np.percentile(img, 15)

    norm_img = img - bkg_mean

    norm_img = np.where(norm_img > 1, norm_img, 1)

    return norm_img

def calculate_residuals(fit_results, data_fit_to):
    residuals = fit_results - data_fit_to
    return residuals

def calculate_fit_error(fit_results, data_fit_to, fit_func=None, fit_params=None, data_fit_on=None):
    """
    To run:
    rmse, r_squared = fit.calculate_fit_error(fit_results, data_fit_to)

    Two options for calculating fit error:
        1. fit_func + fit_params: the fit results are calculated.
        2. fit_results: the fit results are known for each data point.

    Old way of doing this (updated 6/11/22):
    abs_error = fit_results - data_fit_to
    r_squared = 1.0 - (np.var(abs_error) / np.var(data_fit_to))

    :param fit_func: the function used to calculate the fit.
    :param fit_params: generally, popt.
    :param fit_results: the outputs at each input data point ('data_fit_on')
    :param data_fit_on: the input data that was inputted to fit_func to generate the fit.
    :param data_fit_to: the output data that fit_func was fit to.
    :return:
    """

    # --- calculate prediction errors
    if fit_results is None:
        fit_results = fit_func(data_fit_on, *fit_params)

    residuals = calculate_residuals(fit_results, data_fit_to)
    r_squared_me = 1 - (np.sum(np.square(residuals))) / (np.sum(np.square(fit_results - np.mean(fit_results))))

    se = np.square(residuals)  # squared errors
    mse = np.mean(se)  # mean squared errors
    rmse = np.sqrt(mse)  # Root Mean Squared Error, RMSE
    r_squared = 1.0 - (np.var(np.abs(residuals)) / np.var(data_fit_to))

    # print("wiki r-squared: {}; old r-squared: {}".format(np.round(r_squared_me, 4), np.round(r_squared, 4)))
    # I think the "wiki r-squared" is probably the correct one...

    return rmse, r_squared