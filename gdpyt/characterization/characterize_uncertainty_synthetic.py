from gdpyt import GdpytImageCollection, GdpytCalibrationSet
from os.path import join
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# image path
meas_path = r'/Users/mackenzie/Desktop/gdpyt-characterization/synthetic'
calib_img_path = join(meas_path, 'calibration_images')
calib_txt_path = join(meas_path, 'calibration_input')
calib_results_path = join(meas_path, 'calibration_results')
settings_txt_path = join(meas_path, 'settings.txt')
filetype = '.tif'


# read .csv to panda DataFrame
dirs = ['no', '0.1', '0.2', '0.3']
pre_path = r'/Users/mackenzie/Desktop/gdpyt-characterization/synthetic_'
post_path = '-noise/calibration_results/'

gf = []

for d in dirs:
    calibration_files = os.listdir(pre_path + d + post_path)
    for cf in calibration_files:
        if cf.endswith('.csv'):
            gf.append(pre_path + d + post_path + cf)


def calc_uncertainty_z_by_noise(fp_stack, median_size='2.5', infer_method= 'bccorr', method='rmse', d_polyfit=2):
    """
    Calculate and plot the z-uncertainty for all measured particles.

    Parameters
    ----------
    fp_stack

    Returns
    -------

    """
    savedir = '/Users/mackenzie/Desktop/gdpyt-characterization/results/z_vs_uncertainty_z'

    fig, ax = plt.subplots()

    for fp in fp_stack:


        noise = re.search('_noise-(.*)_median', fp).group(1)
        median = re.search('_median-(.*)_infer', fp).group(1)
        infer = re.search('_infer-(.*).csv', fp).group(1)

        if infer == infer_method:

            if median == median_size:
                figtitle = '-- Median-{} Infer-{}'.format(median, infer)
                df = pd.read_csv(filepath_or_buffer=fp)

                if method == 'rmse':
                    error_z = df.groupby(['true_z']).mean()
                    df['mean_square_error_z'] = df['err_z']**2
                    error_z_sum = df.groupby(['true_z']).sum()
                    error_z_sum['counts'] = error_z_sum['cm_threshold'] / 0.8
                    error_z_sum['rmse_z'] = np.sqrt(error_z_sum['mean_square_error_z'] / error_z_sum['counts'])

                    # 2D polyfit
                    fit = np.polyfit(error_z_sum.index, error_z_sum.rmse_z, d_polyfit)
                    fit = np.poly1d(fit)

                    # plot
                    ax.scatter(error_z_sum.index, error_z_sum.rmse_z, s=10, label=noise, alpha=0.5)
                    ax.plot(error_z_sum.index, fit(error_z_sum.index), alpha=0.95, linewidth=2)

                    # formatting
                    ax.set_ylim([-0.1, 1])
                    ax.set_title('RMSE: ' + figtitle)

                elif method == 'avg':
                    error_z = df.groupby(['true_z']).mean()
                    stdev_z = df.groupby(['true_z']).std()

                    # plot
                    ax.errorbar(error_z.index, error_z.err_z, yerr=stdev_z.err_z*2, fmt='o', label=noise)

                    # formatting
                    ax.set_ylim([-1, 1])
                    ax.set_title(r'$Error(z)_{avg. +/- stdev*2}$' + figtitle)


    ax.set_xlabel(r'$\Delta z$ $(\mu m)$')
    ax.set_ylabel(r'$\epsilon_z$')
    ax.legend(title='Gaussian noise')
    #plt.savefig(join(savedir, figtitle + '.png'))
    plt.show()

# run
# calc_uncertainty_z_by_noise(fp_stack=[gf[13], gf[24], gf[44], gf[64]], method='rmse') # w/ median = 2
for i in ['2', '2.5', '3']:
    calc_uncertainty_z_by_noise(fp_stack=gf, median_size=i, infer_method= 'bccorr', method='rmse', d_polyfit=2)


def calc_uncertainty_z_by_median(fp_stack, noise_level='23', infer_method='bccorr', method='rmse', d_polyfit=2, median_sizes=['2', '2.5', '3']):
    """
    Calculate and plot the z-uncertainty for all measured particles.

    Parameters
    ----------
    fp_stack

    Returns
    -------

    """
    savedir = '/Users/mackenzie/Desktop/gdpyt-characterization/results/z_vs_uncertainty_z'

    fig, ax = plt.subplots()

    for fp in fp_stack:


        noise = re.search('_noise-(.*)_median', fp).group(1)
        median = re.search('_median-(.*)_infer', fp).group(1)
        infer = re.search('_infer-(.*).csv', fp).group(1)

        if infer == infer_method:

            if median in median_sizes:

                if noise == noise_level:
                    figtitle = '-- Noise-{} Infer-{}'.format(noise, infer)
                    df = pd.read_csv(filepath_or_buffer=fp)

                    if method == 'rmse':
                        error_z = df.groupby(['true_z']).mean()
                        df['mean_square_error_z'] = df['err_z']**2
                        error_z_sum = df.groupby(['true_z']).sum()
                        error_z_sum['counts'] = error_z_sum['cm_threshold'] / 0.8
                        error_z_sum['rmse_z'] = np.sqrt(error_z_sum['mean_square_error_z'] / error_z_sum['counts'])

                        # 2D polyfit
                        fit = np.polyfit(error_z_sum.index, error_z_sum.rmse_z, d_polyfit)
                        fit = np.poly1d(fit)

                        # plot
                        ax.scatter(error_z_sum.index, error_z_sum.rmse_z, s=10, label=median, alpha=0.5)
                        ax.plot(error_z_sum.index, fit(error_z_sum.index), alpha=0.95, linewidth=2)

                        # formatting
                        ax.set_ylim([-0.1, 1])
                        ax.set_title('RMSE: ' + figtitle)

                    elif method == 'avg':
                        error_z = df.groupby(['true_z']).mean()
                        stdev_z = df.groupby(['true_z']).std()
                        ax.errorbar(error_z.index, error_z.err_z, yerr=stdev_z.err_z*2, fmt='o', label=median)
                        ax.set_ylim([-5, 5])
                        ax.set_title(r'$Error(z)_{avg. +/- stdev*2}$' + figtitle)


    ax.set_xlabel(r'$\Delta z$ $(\mu m)$')
    ax.set_ylabel(r'$\epsilon_z$')
    ax.legend(title='Median filter size')
    #plt.savefig(join(savedir, figtitle + '.png'))
    plt.show()

# run
for i in ['0', '11.5', '23', '34.5']:
    calc_uncertainty_z_by_median(fp_stack=gf, noise_level=i, median_sizes=['2', '2.5', '3'], infer_method='bccorr', method='rmse', d_polyfit=4)
j=1

def calc_uncertainty_xy(fp_stack):
    """
    Calculate and plot the x- and y-uncertainty for all measured particles.

    Parameters
    ----------
    fp_stack

    Returns
    -------

    """
    fig, ax = plt.subplots()

    for fp in fp_stack:
        df = pd.read_csv(filepath_or_buffer=fp)
        error_xy = df.groupby(['true_z']).mean()
        error_xy['error_dist'] = np.sqrt(error_xy.err_y**2 + error_xy.err_x**2)
        stdev_xy = df.groupby(['true_z']).std()
        stdev_xy['stdev_dist'] = np.sqrt(stdev_xy.err_x**2 + stdev_xy.err_y**2)

        ax.errorbar(error_xy.index, error_xy.error_dist, yerr=stdev_xy.stdev_dist*2, fmt='o')

    ax.set_ylim([-1, 5])
    ax.set_xlabel(r'$\Delta z$ $(\mu m)$')
    ax.set_ylabel(r'$\epsilon_{xy}$')
    ax.set_title(r'Particle Localization Uncertainty: $\epsilon_{xy}=\sqrt{\epsilon_x^2+\epsilon_y^2}$')
    ax.legend(['1', '2', '3', '4', '5'], title='Median filter size')
    plt.show()

# run
#calc_uncertainty_xy(fp_stack=[fp1, fp2, fp3, fp4, fp5])

def calc_particle_diameter(fp_stack):
    """
    Plot theoretical defocused and measured particle diameter.

    Parameters
    ----------
    fp_stack

    Returns
    -------

    """
    fig, ax = plt.subplots()

    for fp in fp_stack:
        df = pd.read_csv(filepath_or_buffer=fp)
        pd_mean = df.groupby(['true_z']).mean()
        pd_std = df.groupby(['true_z']).std()

        ax.errorbar(pd_mean.index, pd_mean.p_dia_meas, yerr=pd_std.p_dia_meas * 2, fmt='o')
        #ax.errorbar(pd_mean.index, pd_mean.p_dia_def, yerr=pd_std.p_dia_def*2, fmt='o')


    ax.set_ylim([0, 20])
    ax.set_xlabel(r'$\Delta z$ $(\mu m)$')
    ax.set_ylabel(r'$D_{p}$ (pixels)')
    #ax.set_title(r'')
    ax.legend(['1', '2', '3', '4', '5'], title='Median filter size')
    plt.show()

# run
#calc_particle_diameter(fp_stack=[fp1, fp2, fp3, fp4, fp5])

def calc_particle_area(fp_stack, second_plot='num_particles'):
    """
    Plots as a functino of true_z:
     1. measured particle area on left y-axis
     2. number of detected and measured particles on right y-axis

    Parameters
    ----------
    fp_stack

    Returns
    -------

    """

    save_loc = '/Users/mackenzie/Desktop/gdpyt-characterization/results/z_vs_area_and_numberOfParticles'

    for fp in fp_stack:
        noise = re.search('_noise-(.*)_median', fp).group(1)
        median = re.search('_median-(.*)_infer', fp).group(1)
        infer = re.search('_infer-(.*).csv', fp).group(1)

        if infer == 'bccorr':
            figtitle = 'Noise{} Median{}'.format(noise, median)

            fig, ax = plt.subplots()
            df = pd.read_csv(filepath_or_buffer=fp)
            pd_mean = df.groupby(['true_z']).mean()
            pd_std = df.groupby(['true_z']).std()
            pd_sum = df.groupby(['true_z']).sum()
            pd_sum['counts'] = pd_sum['cm_threshold'] / 0.8

            # plot particle area
            ax.errorbar(pd_mean.index, pd_mean.p_area, yerr=pd_std.p_area * 2, fmt='o', label='Area')
            ax.set_xlabel(r'$\Delta z$ $(\mu m)$')
            ax.set_ylabel(r'$A_{p}$ $(pixels^2)$', color='tab:blue')
            # ax.set_ylim([0, 20])

            if second_plot == 'num_particles':
                # plot # of particles
                ax2 = ax.twinx()
                ax2.scatter(pd_sum.index, pd_sum.counts, color='gray', alpha=0.5, label='# of particles')
                ax2.plot(pd_sum.index, pd_sum.counts, color='lightgray', alpha=0.5)
                ax2.set_ylabel('# of particles', color='gray')
                ax2.set_ylim([0, 105])

            plt.title(figtitle)
            savename = join(save_loc, figtitle + '.png')
            plt.savefig(savename)
            #plt.show()

# run
#calc_particle_area(fp_stack=gf)


j=1