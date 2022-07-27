
# imports
from gdpyt.utils import get

from os.path import splitext, join
import re
import logging

from math import floor
import random
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.draw import polygon, ellipse_perimeter
from skimage.exposure import rescale_intensity
from skimage import exposure
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib as mpl

logger = logging.getLogger(__name__)


def plot_baseline_image_and_particle_ids(collection, filename=None):

    if filename is None:
        if collection.image_collection_type == 'calibration':
            baseline_image_filename = collection.baseline
        else:
            baseline_image_filename = collection.particle_id_image
    else:
        baseline_image_filename = filename

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(collection.images[baseline_image_filename].draw_particles(raw=False, thickness=1, draw_id=True, draw_bbox=True))

    if collection.images[baseline_image_filename].z is None:
        ax.set_title('{}'.format(collection.images[baseline_image_filename].filename))
    else:
        ax.set_title('z = {}'.format(np.round(collection.images[baseline_image_filename].z, 2)))

    return fig


def plot_single_particle_template_and_z(collection, particle_id, z=None, cmap='binary',
                                        draw_contours=False, fill_contours=False):

    if z is not None:
        if not (isinstance(z, list) or isinstance(z, tuple)):
            raise TypeError("Specify z range as a two-element list or tuple")

    z_trues = []
    zs = []
    templates = []
    for img in collection.images.values():
        for p in img.particles:
            if p.id == particle_id:
                yield p.z_true, p.z, p.template


def plot_single_particle_stack(collection, particle_id, z=None, draw_contours=True, fill_contours=False, imgs_per_row=20,
                               fig=None, axes=None, format_string=False):

    if z is not None:
        if not (isinstance(z, list) or isinstance(z, tuple)):
            raise TypeError("Specify z range as a two-element list or tuple")

    z_trues = []
    zs = []
    cms = []
    templates = []
    for img in collection.images.values():
        for p in img.particles:
            if p.id == particle_id:
                z_trues.append(p.z_true)
                zs.append(p.z)
                cms.append(p.cm)
                templates.append(p.get_template())

    zipped = list(zip(z_trues, zs, cms, templates))
    sorted_list = sorted(zipped, key=lambda x: x[0])

    n_images = len(templates)
    n_cols = min(imgs_per_row, n_images)
    n_rows = int(n_images / n_cols) + 1
    if fig is None:
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(2 * n_cols, n_rows * 2))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes]).reshape(-1, 1)

    if n_rows > 1 and n_cols > 1:
        for i in range(n_rows):
            for j in range(n_cols):
                n = i * n_cols + j
                if n > n_images - 1:
                    axes[i, j].imshow(np.zeros_like(template), cmap='viridis')
                    axes[i, j].set_title('None', fontsize=12)
                else:
                    z_true = sorted_list[n][0]
                    z = sorted_list[n][1]
                    cm = sorted_list[n][2]
                    template = sorted_list[n][3]

                    axes[i, j].imshow(template, cmap='viridis')
                    # axes[i, j].set_title('(z, true z) = ({}, {})'.format(np.round(z, 2), np.round(z_true, 2)), fontsize=10)
                    axes[i, j].set_title(r'$(\epsilon_{z}, c_{m}) =$' + ' ({}, {})'.format(np.round(z - z_true, 2),
                                                                                           np.round(cm, 2)),
                                         fontsize=8)
                    axes[i, j].get_xaxis().set_visible(False)
                    axes[i, j].get_yaxis().set_visible(False)

                axes[i, j].get_xaxis().set_visible(False)
                axes[i, j].get_yaxis().set_visible(False)

    fig.suptitle('Particle {} stack'.format(particle_id))
    fig.subplots_adjust(wspace=0.05, hspace=0.25)

    return fig

def plot_calib_stack(stack, z=None, draw_contours=True, fill_contours=False, imgs_per_row=5, fig=None, axes=None, format_string=False):
    # assert isinstance(stack, GdpytCalibratioStack)

    if z is not None:
        if not (isinstance(z, list) or isinstance(z, tuple)):
            raise TypeError("Specify z range as a two-element list or tuple")

    n_images = len(stack)
    n_cols = min(imgs_per_row, n_images)
    n_rows = int(n_images / n_cols) + 1
    if fig is None:
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(2 * n_cols, n_rows * 2))

    if n_images <= imgs_per_row:
        return fig

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes]).reshape(-1,1)

    for i in range(n_rows):
        for j in range(n_cols):
            n = i * n_cols + j
            if n > n_images - 1:
                axes[i, j].imshow(np.zeros_like(template), cmap='viridis')
                axes[i, j].set_title('None', fontsize=12)
            else:
                z, template = stack[n]
                axes[i, j].imshow(template, cmap='viridis', origin='lower')
                particle_z = [p for p in stack.particles if p.z == z]

                for index, p_z in enumerate(particle_z):
                    connected_contour = np.vstack([p_z.template_contour, p_z.template_contour[0, :]])
                    axes[i, j].plot(connected_contour[:, 0], connected_contour[:, 1], linewidth=0.5, color='red')

                    if fill_contours is True:
                        rr, cc = polygon(connected_contour[:, 1], connected_contour[:, 0], template.shape)
                        template_contour = template.copy()
                        template_contour[rr, cc] = np.max(template)
                        axes[i, j].imshow(template_contour, cmap='viridis', origin='lower')

                if format_string:
                    my_format = "{0:.3f}"
                    axes[i, j].set_title('z = {}'.format(my_format.format(np.round(z,2))), fontsize=12)
                else:
                    axes[i, j].set_title('z = {}'.format(np.round(z,2)), fontsize=12)
                axes[i, j].get_xaxis().set_visible(False)
                axes[i, j].get_yaxis().set_visible(False)

            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)

    fig.suptitle('Calibration stack (Particle ID {})'.format(stack.id))
    fig.subplots_adjust(wspace=0.05, hspace=0.25)

    return fig

def plot_calib_stack_3d(calib_stack, intensity_percentile=(10, 98.75), stepsize=5, aspect_ratio=3):

    temp = []
    z = []
    for p in calib_stack.particles:
        temp.append(p.template)
        z.append(p.z_true)
    zipped = zip(z, temp)
    z_stack = list(sorted(zipped, key=lambda x : x[0]))

    stack_3d = []
    for p in z_stack:
        x, y = p[1].shape
        yh = int(y // 2)
        half_temp = p[1][:, :yh]
        stack_3d.append(half_temp)

    stack_3d = np.array(stack_3d)
    vmin, vmax = np.percentile(stack_3d, intensity_percentile)
    stack_rescale = exposure.rescale_intensity(stack_3d, in_range=(vmin, vmax), out_range=(0, 1))

    X, Y = np.mgrid[0:x, 0:yh]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect(aspect=(1, 1, aspect_ratio))

    # ls = LightSource(azdeg=40, altdeg=50) if you want to shade the image slices

    for i, img in enumerate(stack_rescale):
        if i == 0 or (i + 1) % stepsize == 0:
            Z = np.zeros(X.shape) + z_stack[i][0]
            T = mpl.cm.viridis(img)
            # T = ls.shade(img, plt.cm.viridis) if you want to shade the image slices
            fig3d = ax.plot_surface(X, Y, Z, facecolors=T, linewidth=0, alpha=1, cstride=1, rstride=1, antialiased=False)

    ax.view_init(40, 50)

    ax.set_title(r'Calibration stack: $p_{ID}$ = ' + str(calib_stack.id))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'$z_{true}$ / h')

    cbar = fig.colorbar(fig3d)
    cbar.ax.set_yticklabels(np.arange(vmin, vmax, floor((vmax-vmin)/5), dtype=int))
    cbar.set_label(r'$I_{norm}$')

    return fig


def plot_calib_stack_self_similarity(calib_stack):
    fig, ax = plt.subplots()

    # plot the forward self-similarity
    ax.plot(calib_stack.self_similarity_forward[:, 0], calib_stack.self_similarity_forward[:, 1], color='cornflowerblue', alpha=0.125)
    ax.scatter(calib_stack.self_similarity_forward[:, 0], calib_stack.self_similarity_forward[:, 1],
               label=r'$\left(z_{i+1}\right)$', color='cornflowerblue', alpha=0.125)

    # plot the mean self-similarity of both adjacent images
    ax.plot(calib_stack.self_similarity[:, 0], calib_stack.self_similarity[:, 1], color='tab:blue')
    ax.scatter(calib_stack.self_similarity[:, 0], calib_stack.self_similarity[:, 1],
               label=r'$|\left(z_{i-1}, z_{i+1}\right)|$', color='tab:blue')

    # format figure
    ax.set_xlabel(r'$z$ / h')
    ax.set_ylabel(r'$S_i(z)$')
    #ax.set_ylim([np.min(calib_stack.self_similarity[:, 1])*0.95, 1.005])
    ax.legend()
    ax.grid(alpha=0.25)

    return fig


def plot_adjacent_self_similarity(calib_stack, index=[]):
    """
    Parameters
    ----------
    calib_stack
    index: the index of N_CAL images between z=0 and z=h. Note, these are different index units than similarity arrays.
    """
    fig = None

    if len(index) == 0:
        raise ValueError("Need to input z-value index to plot adjacent self similarities with images")
    else:
        stack_layers = list(calib_stack.layers.items())

        if index[0] > len(stack_layers) - 3:
            #index = [ind for ind in random.sample(set(np.arange(1, len(stack_layers) - 1)), 1)]
            index = [index[0] - 4]
        elif index[0] < 2:
            index = [2]

        for i in index:
            backward_image = stack_layers[i-1][1]
            backward_cm = calib_stack.self_similarity_forward[i-1][1]

            forward_image = stack_layers[i+1][1]
            forward_cm = calib_stack.self_similarity_forward[i][1]

            center_image = stack_layers[i][1]
            mean_adjacent_cm = calib_stack.self_similarity[i-1][1]

            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 6), sharey=True)
            ax1.imshow(backward_image, cmap='viridis')
            ax1.set_xlabel(r'$z_{true}$ = ' + str(calib_stack.self_similarity_forward[i-1][0]))
            ax1.set_title(r'$c_{m}(i-1, i)$' + '={}'.format(np.round(backward_cm, 4)))
            ax1.grid(color='gray', alpha=0.5)
            ax2.imshow(center_image, cmap='viridis')
            ax2.set_xlabel(r'$z_{true}$ = ' + str(calib_stack.self_similarity[i-1][0]))
            ax2.set_title(r'|$c_{m}(i-1, i), c_{m}(i, i+1)$|' + '={}'.format(np.round(mean_adjacent_cm, 4)))
            ax2.grid(color='gray', alpha=0.5)
            ax3.imshow(forward_image, cmap='viridis')
            ax3.set_xlabel(r'$z_{true}$ = ' + str(calib_stack.self_similarity_forward[i+1][0]))
            ax3.set_title(r'$c_{m}(i, i+1)$' + '={}'.format(np.round(forward_cm, 4)))
            ax3.grid(color='gray', alpha=0.5)

            plt.tight_layout()

        if fig is None:
            fig, ax = plt.subplots()

    return fig


def plot_img_collection(collection, raw=True, draw_particles=True, exclude=[], **kwargs):
    # Put this in there because this kwarg is used in GdpytImage.draw_particles
    kwargs.update({'raw': raw})

    images_to_plot = [img for name, img in collection.images.items() if name not in exclude]

    n_axes = len(images_to_plot)
    n_cols = min(10, n_axes)
    n_rows = int(n_axes / n_cols) + 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(2 * n_cols, n_rows * 2))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes]).reshape(-1, 1)

    vmin = []
    vmax = []
    for image in images_to_plot:
        if raw:
            img = image.raw
        else:
            img = image.filtered
        vmin.append(img.min())
        vmax.append(img.max())

    vmin = min(vmin)
    vmax = max(vmax)

    for i in range(n_rows):
        for j in range(n_cols):
            n = i * n_cols + j
            if n > len(images_to_plot) - 1:
                im = axes[i, j].imshow(np.zeros_like(canvas), cmap='gray', vmin=vmin, vmax=vmax)
                axes[i, j].set_title('None', fontsize=5)
            else:
                image = images_to_plot[n]
                if draw_particles:
                    canvas = image.draw_particles(**kwargs)
                else:
                    if raw:
                        canvas = image.raw
                    else:
                        canvas = image.filtered
                im = axes[i, j].imshow(canvas, cmap='gray', vmin=vmin, vmax=vmax)
                axes[i, j].set_title('{}'.format(image.filename), fontsize=5)
            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)

    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    if raw:
        img_type = 'raw'
    else:
        img_type = 'filtered'

    fig.suptitle('Image collection {}, Image type: {}'.format(collection.folder, img_type))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    return fig

def plot_particle_trajectories(collection, sort_images=None):
    coords = []
    if sort_images is None:
        for image in collection.images.values():
            coords.append(image.particle_coordinates())
    else:
        if not callable(sort_images):
            raise TypeError("sort_images must be a function that takes an image name as an argument and returns a value"
                            "that can be used to sort the images")
        # Get the particle coordinates from all the images
        for file in sorted(collection.files, key=sort_images):
            coords.append(collection.images[file].particle_coordinates())

    coords = pd.concat(coords)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for id_ in coords['id'].unique():
        thisid = coords[coords['id'] == id_]
        ax.scatter(thisid['x'], thisid['y'], thisid['z'], label='ID_{}'.format(id_))
    ax.legend()

    return fig

def plot_animated_surface(collection, sort_images=None, fps=10, save_as=None):
    coords = []

    # Get all the coordinates from all the images in a specific order
    if sort_images is None:
        for image in collection.images.values():
            coords.append(image.particle_coordinates())
    else:
        if not callable(sort_images):
            raise TypeError("sort_images must be a function that takes an image name as an argument and returns a value"
                            "that can be used to sort the images")
        # Get the particle coordinates from all the images
        for file in sorted(collection.files, key=sort_images):
            coords.append(collection.images[file].particle_coordinates())

    # Total number of images is the number of frames, get min and max z for axis limits
    n_frames = len(coords)
    coords_all = pd.concat(coords)
    max_z, min_z = coords_all['z'].max(), coords_all['z'].min()

    coords_0 = coords[0]
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    surf = ax.plot_trisurf(coords_0['x'].values, coords_0['y'].values,
                           coords_0['z'].values, cmap='magma', antialiased=False)
    ax.set_zlim([min_z, max_z])

    def update_surf(frame_idx):
        ax.clear()
        coord = coords[frame_idx]
        surf = ax.plot_trisurf(coord['x'].values, coord['y'].values, coord['z'].values, cmap='magma', antialiased=False)
        ax.set_zlim([min_z, max_z])

    ani = animation.FuncAnimation(fig, update_surf, n_frames, interval=1000 / fps, blit=False)
    if save_as is not None:
        root, ext = splitext(save_as)
        if ext not in ['.gif']:
            logger.error("In the current version animations can only be saves as a .gif. Received {}".format(ext))
        else:
            ani.save(save_as, writer='imagemagick', fps=fps)
    return fig


def plot_particle_coordinate(collection, coordinate='z', sort_images=None, particle_id=None):
    if particle_id is None:
        raise ValueError("Specify an integer or a list of integers corresponding to the particle IDs that should be plotted")
    else:
        if isinstance(particle_id, int):
            particle_id = [particle_id]
        elif isinstance(particle_id, list):
            pass
        else:
            raise TypeError("particle_id must be an integer or a list. Received type {}".format(type(particle_id)))

    if coordinate not in ['x', 'y', 'z']:
        raise ValueError("coordinate must be one of 'x', 'y' or 'z'. Received {}".format(coordinate))

    coords = []
    if sort_images is None:
        iter_images = collection.images.files
    else:
        if not callable(sort_images):
            raise TypeError("sort_images must be a function that takes an image name as an argument and returns a value"
                            "that can be used to sort the images")
        iter_images = sorted(collection.files, key=sort_images)

    # Get the particle coordinates from all the images
    for file in iter_images:
        coords_thisfile = collection.images[file].particle_coordinates(id_=particle_id)
        if coords_thisfile.empty:
            coords_thisfile = pd.DataFrame({'id': particle_id, coordinate: len(particle_id) * [np.nan]})
            coords_thisfile = coords_thisfile.set_index('id')[[coordinate]]
        elif not all([id_ in coords_thisfile['id'].tolist() for id_ in particle_id]):
            missing_ids = [id_ for id_ in particle_id if id_ not in coords_thisfile['id'].tolist()]
            coords_thisfile = coords_thisfile.set_index('id')[[coordinate]]
            coords_thisfile = coords_thisfile.append(pd.DataFrame({coordinate: len(missing_ids)*[np.nan]}, index=missing_ids))
        else:
            coords_thisfile = coords_thisfile.set_index('id')[[coordinate]]
        coords.append(coords_thisfile)

    coords = pd.concat(coords)
    fig, ax = plt.subplots(figsize=(11, 7))
    for id_ in particle_id:
        j = coords.loc[id_]
        ax.plot(coords.loc[id_].values, label='ID {}'.format(id_))
    ax.set_xlabel('Image #')
    ax.set_ylabel('{} position'.format(coordinate.upper()))
    #ax.legend()

    return fig

def plot_particle_coordinate_calibration(collection, measurement_quality, plot_type='errorbars',
                                         measurement_depth=None, true_xy=False, measurement_width=None):

    df = measurement_quality

    fig, ax = plt.subplots()
    ax.plot(df.true_z, df.true_z, color='black', linewidth=1, alpha=1, label='Ideal')

    if measurement_depth and true_xy:
        df['rmse_xy'] = (df['rmse_x'] ** 2 + df['rmse_y'] ** 2) ** 0.5
        df['rmse_vol_xy'] = df['rmse_xy'] / (measurement_width * 1e6)
        df['rmse_vol_z'] = df['rmse_z'] / 1
        if plot_type == 'scatter':
           ax.scatter(x=df.true_z, y=df.z)
        if plot_type == 'errorbars':
            ax.errorbar(x=df.true_z, y=df.z, yerr=df.rmse_vol_z, xerr=df.rmse_vol_xy, fmt='o', ecolor='gray', elinewidth=1, capsize=2, label='Measured')
        ax.set_xlabel(r'$z_{true}$ / h')
        ax.set_ylabel(r'$z_{measured}$ / h')

    elif measurement_depth is None and true_xy:
        if plot_type == 'scatter':
           ax.scatter(x=df.true_z, y=df.z)
        if plot_type == 'errorbars':
            ax.errorbar(x=df.true_z, y=df.z, yerr=df.rmse_z, xerr=df.rmse_xy, fmt='o', ecolor='gray', elinewidth=1, capsize=2, label='Measured')
        ax.set_xlabel(r'$\Delta z_{true} (\mu m)$')
        ax.set_ylabel(r'$z_{measured}$ $(\mu m)$')

    elif measurement_depth and true_xy is False:
        if plot_type == 'scatter':
           ax.scatter(x=df.true_z, y=df.z)
        if plot_type == 'errorbars':
            ax.errorbar(x=df.true_z, y=df.z, yerr=df.rmse_z, fmt='o', ecolor='gray', elinewidth=1, capsize=2, label='Measured')
        ax.set_xlabel(r'$z_{true}$ / h')
        ax.set_ylabel(r'$z_{measured}$ / h')

    ax.set_title('Measurement uncertainty: root mean squared error', fontsize=8)
    ax.legend()
    plt.tight_layout()

    return fig

def plot_images_and_similarity_curve(calib_set, test_col, particle_id=None, image_id=None, min_cm=0.5, error=5,
                                     sup_title=None, save_path=None):
    """
    Plot similarity curve with corresponding z-guess and z-true images.
    """
    margins = 0

    particles_to_plot = []
    z_trues = []

    if particle_id is not None and image_id is not None:
        # get a specific particle in a specific image
        for particle in test_col.images[image_id]:
            if particle.id == particle_id:
                if not np.isnan(particle.z):
                    if np.abs(particle.z_true - particle.z) > error:
                        particles_to_plot.append(particle)
                        z_trues.append(particle.z_true)

    elif particle_id is None and image_id is not None:
        # get every particle in a specific image with c_m > min_cm
        for particle in test_col.images[image_id]:
            if particle.cm > min_cm:
                if not np.isnan(particle.z):
                    if np.abs(particle.z_true - particle.z) > error:
                        particles_to_plot.append(particle)
                        z_trues.append(particle.z_true)

    elif particle_id is not None and image_id is None:
        # get a specific particle in every image
        for image in test_col.images.values():
            for particle in image.particles:
                if particle.id == particle_id:
                    if not np.isnan(particle.z):
                        if np.abs(particle.z_true - particle.z) > error:
                            particles_to_plot.append(particle)
                            z_trues.append(particle.z_true)
    else:
        raise ValueError("Need to specify a particle ID or image ID.")

    for p in particles_to_plot:

        z_calib, template_calib = get.corresponding_calibration_stack_particle_template(calib_set, p.z_true, p.id)

        # calculate error
        p_error = np.round(np.abs(p.z_true - p.z), 2)

        # particle template: z_true
        ax1 = plt.subplot(221)
        ax1.margins(margins)
        ax1.imshow(template_calib)
        ax1.axis('off')
        ax1.set_title(r'$z_{calib, nearest}=$ ' + '{}'.format(z_calib))

        # particle template: z
        ax2 = plt.subplot(222)
        ax2.margins(margins)  # Values in (-0.5, 0.0) zooms in to center
        ax2.imshow(p.template)
        ax2.axis('off')
        ax2.set_title(r'$z_{guess}=$ ' + '{}'.format(np.round(p.z, 3)))

        ax3 = plt.subplot(212)
        ax3.margins(margins)  # Default margin is 0.05, value 0 means fit

        # discrete c_m
        sim_z = p.similarity_curve.iloc[:, 0]
        sim_cm = p.similarity_curve.iloc[:, 1]
        ax3.plot(sim_z, sim_cm, color='tab:blue', label='c_m')
        # interpolated c_m
        interp_z = p.interpolation_curve.iloc[:, 0]
        interp_cm = p.interpolation_curve.iloc[:, 1]
        ax3.plot(interp_z, interp_cm, color='lightcoral', label='interp')
        # z_true
        ax3.scatter(p.z_true, p.cm, s=75, color='green', marker='*', label=r'$z_{true}$')

        ax3.set_ylim([np.min(sim_cm), 1.0125])
        ax3.set_xlabel(r'$z_{true}$')
        ax3.set_ylabel(r'$c_{m}$')
        ax3.set_title(r'$z_{true}$ = ' + str(np.round(p.z_true, 3)) + ', ' + r'$\epsilon_{z}$ = ' + str(p_error), fontsize=11)
        ax3.grid(b=True, which='major', alpha=0.25)
        ax3.grid(b=True, which='minor', alpha=0.125)

        """# determine the major and minor axis labels
        x_major = np.round((np.max(sim_z) - np.min(sim_z)) / 1, 1)
        x_minor = np.round((np.max(sim_z) - np.min(sim_z)) / 2, 1)

        ax3.xaxis.set_major_locator(MultipleLocator(x_major))
        ax3.xaxis.set_minor_locator(MultipleLocator(x_minor))
        ax3.yaxis.set_major_locator(MultipleLocator(0.05))
        ax3.yaxis.set_minor_locator(MultipleLocator(0.01))"""

        plt.suptitle(sup_title)
        plt.tight_layout()
        plt.savefig(fname=save_path, bbox_inches='tight')
        plt.close()


def plot_similarity_curve(collection, sub_image, method=None, min_cm=0, particle_id=None, image_id=None, imgs_per_row=9):
    """
    Plots the similarity curve for:
        1. a single particle (particle_id) across all images.
        2. every particle on a single image (image_id).
    Note - both particle_id and image_id cannot be valid. One must be None.
    Parameters
    ----------
    collection:     the test image collection with z-determined particles
    sub_image       if sub-image interpolation was used
    method          which inference method was applied
    min_cm          the minimum acceptable correlation value
    particle_id     which particle to plot
    image_id        which image to plot particles

    Returns
    -------

    """
    if particle_id is None and image_id is None:
        raise ValueError("Specify an integer for particle_id or filename for image_id")
    elif particle_id is not None and image_id is not None:
        raise ValueError("Cannot specify both image_id and particle_id")
    elif particle_id is not None:
        if isinstance(particle_id, int):
            pass
        elif isinstance(particle_id, list):
            particle_id = particle_id[0]
        else:
            raise TypeError("particle_id must be an integer or a list. Received type {}".format(type(particle_id)))

    img_list = sorted(collection.images.items(), key=lambda x: x[0], reverse=False)

    if particle_id is not None:
        n_images = len(collection.images)

        if n_images > 54: # TODO: make this programmatic to spread through the z-range
            n_images = 54

        n_cols = min(imgs_per_row, n_images)
        n_rows = int(n_images / n_cols) + 1

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(2 * n_cols, n_rows * 2))

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes]).reshape(-1, 1)

        k = 0
        for i in range(n_rows):
            for j in range(n_cols):
                n = i * n_cols + j
                if n > n_images - 1:
                    axes[i, j].axis('off')
                    axes[i, j].axis('off')
                else:
                    for p in img_list[k][1].particles:
                        if p.id != particle_id:
                            pass
                        else:
                            if np.isnan([p.z]):
                                logger.warning("particle z-coordinate is NaN")
                                axes[i, j].text(0.5, 0.5, str(np.round(p.cm, 3)) + r'$<c_{min}$', ha='center', va='center', color='red', fontsize=15)
                                axes[i, j].set_title(r'$z_{true}$ = ' + str(np.round(p.z_true, 3)), fontsize=11)
                                axes[i, j].set_yticklabels([])
                                axes[i, j].set_xticklabels([])
                                k += 1
                            else:
                                sim_z = p.similarity_curve.iloc[:, 0]
                                sim_cm = p.similarity_curve.iloc[:, 1]
                                axes[i, j].plot(sim_z, sim_cm, color='tab:blue', label='c_m')



                                if sub_image:
                                    interp_z = p.interpolation_curve.iloc[:, 0]
                                    interp_cm = p.interpolation_curve.iloc[:, 1]
                                    axes[i, j].plot(interp_z, interp_cm, color='lightcoral', label='interp')

                                axes[i, j].set_title(r'$z_{true}$ = ' + str(np.round(p.z_true, 3)), fontsize=11)
                                axes[i, j].grid(b=True, which='major', alpha=0.25)
                                axes[i, j].grid(b=True, which='minor', alpha=0.125)

                                # determine the major and minor axis labels
                                x_major = np.round((np.max(sim_z) - np.min(sim_z)) / 2, 1)
                                x_minor = np.round((np.max(sim_z) - np.min(sim_z)) / 4, 1)

                                axes[i, j].xaxis.set_major_locator(MultipleLocator(x_major))
                                axes[i, j].xaxis.set_minor_locator(MultipleLocator(x_minor))
                                axes[i, j].yaxis.set_major_locator(MultipleLocator(0.5))
                                axes[i, j].yaxis.set_minor_locator(MultipleLocator(0.1))

                                # only place labels on outer axes
                                if j != 0:
                                    axes[i, j].set_yticklabels([])
                                else:
                                    axes[i, j].set_ylabel(r'$c_m$')

                                if i != n_rows - 1:
                                    axes[i, j].set_xticklabels([])
                                else:
                                    axes[i, j].set_xlabel(r'$z_{cal}$')

                                # get 5% of the z-span for error calculation and red figure outlining
                                error_threshold = (np.max(sim_z) - np.min(sim_z)) * 0.05

                                # if measured z-coord > error threshold, change axes spines color to red
                                if np.abs(p.z_true - sim_z[np.argmax(sim_cm)]) > error_threshold:
                                    axes[i, j].text(0.5, 0.05, r'$\epsilon_{z}=$' + str(
                                        np.round(np.abs(p.z_true - sim_z[np.argmax(sim_cm)]), 3)),
                                                    ha='center', va='center', color='dimgray', fontsize=11)
                                    for side in ['bottom', 'left', 'right', 'top']:
                                        axes[i, j].spines[side].set_color('red')
                                else:
                                    axes[i, j].text(0.5, 0.05, r'$z=$' + str(
                                        np.round(sim_z[np.argmax(sim_cm)], 3)),
                                                    ha='center', va='center', color='dimgray', fontsize=11)

                                k += 1

    elif image_id is not None:

        n_images = len(img_list[image_id][1].particles)
        k = 0

        if n_images == 1:
            fig, axes = plt.subplots(nrows=2, figsize=(6,6))
            p = img_list[image_id][1].particles[0]
            if np.isnan([p.z]):
                logger.warning("particle z-coordinate is NaN")
                fig = None
                return fig
            else:
                sim_z = p.similarity_curve.iloc[:, 0]
                sim_cm = p.similarity_curve.iloc[:, 1]
                axes[0].scatter(sim_z, sim_cm, s=1, color='tab:blue', label=r'$c_m$')
                axes[0].plot(sim_z, sim_cm, color='tab:blue', alpha=0.15)
                axes[0].scatter(p.z_true, np.max(sim_cm), s=25, color='black', marker='*',
                                label=r'$z_{true}$')

                # get index of maximum correlation
                max_ind = np.argmax(sim_cm)

                # define left edge of interpolated plot
                max_ind_left = max_ind - 2
                if max_ind_left < 0:
                    max_ind_left = 0

                # define right edge of interpolated plot
                max_ind_right = max_ind + 3
                if max_ind_right > len(sim_z) - 1:
                    max_ind_right = len(sim_z) - 1

                # plot correlation near maximum
                sim_z_near_max = sim_z[max_ind_left:max_ind_right]
                sim_cm_near_max = sim_cm[max_ind_left:max_ind_right]
                axes[1].scatter(sim_z_near_max, sim_cm_near_max, s=6, color='tab:blue')
                axes[1].plot(sim_z_near_max, sim_cm_near_max, color='tab:blue', alpha=0.25)
                axes[1].scatter(sim_z[max_ind], sim_cm[max_ind], s=50, color='b', marker='o',
                                label=r'$\epsilon_{sub-image: off}=$' + str(np.round(np.abs(p.z_true-sim_z[max_ind]), 4)))

                if sub_image:
                    interp_z = p.interpolation_curve.iloc[:, 0]
                    interp_cm = p.interpolation_curve.iloc[:, 1]
                    axes[0].plot(interp_z, interp_cm, color='lightcoral', alpha=0.75, label=r'$interp.$')

                    interp_cm_max = np.max(interp_cm)
                    interp_z_max = interp_z[np.argmax(interp_cm)]
                    axes[1].plot(interp_z, interp_cm, color='lightcoral', alpha=0.75)
                    axes[1].scatter(interp_z_max, interp_cm_max, s=50, color='r', marker='*',
                                    label=r'$\epsilon_{sub-image: on}=$'+str(np.round(np.abs(p.z_true-interp_z_max), 4)))
                    axes[1].scatter(p.z_true, np.max(interp_cm_max), s=50, color='black', marker='*', label=r'$z_{true}$')

                # determine the major and minor axis labels
                x_major = np.round((np.max(sim_z_near_max) - np.min(sim_z_near_max)) / 2, 1)
                x_minor = np.round((np.max(sim_z_near_max) - np.min(sim_z_near_max)) / 4, 1)

                axes[0].set_ylim(bottom=0, top=1.05)
                axes[0].set_ylabel(r'$c_m$')
                axes[0].set_title(r'$z_{true}$ = ' + str(np.round(p.z_true, 2)), fontsize=11)
                axes[0].grid(b=True, which='major', alpha=0.25)
                axes[0].grid(b=True, which='minor', alpha=0.125)
                axes[0].xaxis.set_major_locator(MultipleLocator(x_major))
                axes[0].xaxis.set_minor_locator(MultipleLocator(x_minor))
                axes[0].yaxis.set_major_locator(MultipleLocator(0.25))
                axes[0].yaxis.set_minor_locator(MultipleLocator(0.05))
                axes[0].legend()

                axes[1].set_ylim(bottom=np.max([np.min(sim_cm_near_max), 0.8]), top=1.025)
                axes[1].set_ylabel(r'$c_m$')
                axes[1].set_xlabel(r'$z_{cal}$')
                axes[1].grid(b=True, which='major', alpha=0.25)
                axes[1].grid(b=True, which='minor', alpha=0.125)
                axes[1].xaxis.set_major_locator(MultipleLocator(x_major))
                axes[1].xaxis.set_minor_locator(MultipleLocator(x_minor))
                axes[1].yaxis.set_major_locator(MultipleLocator(0.025))
                axes[1].yaxis.set_minor_locator(MultipleLocator(0.005))
                axes[1].legend(fontsize=8, loc='upper right')

        else:
            n_images = len(img_list[image_id][1].particles)
            if n_images > 54:
                n_images = 54

            if n_images == 0:
                fig, axes = plt.subplots()
                return fig
            else:
                n_cols = min(imgs_per_row, n_images)
                n_rows = int(n_images / n_cols) + 1
                fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(2 * n_cols, n_rows * 2))

            if not isinstance(axes, np.ndarray):
                axes = np.array([axes]).reshape(-1, 1)

            k = 0
            for i in range(n_rows):
                for j in range(n_cols):
                    n = i * n_cols + j
                    if n > n_images - 1:
                        axes[i, j].axis('off')
                        axes[i, j].axis('off')
                    else:
                        for p in img_list[image_id][1].particles:
                            if p.id != k:
                                pass
                            else:
                                if np.isnan([p.z]):
                                    logger.warning("particle z-coordinate is NaN")
                                    axes[i, j].text(0.5, 0.5, str(np.round(p.cm, 3)) + r'$<c_{min}$', ha='center',
                                                    va='center', color='red', fontsize=15)
                                    axes[i, j].set_title(r'$p_{ID}$' + '= {}'.format(p.id), fontsize=11)
                                    axes[i, j].set_yticklabels([])
                                    axes[i, j].set_xticklabels([])
                                else:
                                    sim_z = p.similarity_curve.iloc[:, 0]
                                    sim_cm = p.similarity_curve.iloc[:, 1]
                                    axes[i, j].plot(sim_z, sim_cm, color='tab:blue', label='c_m')

                                    if sub_image:
                                        interp_z = p.interpolation_curve.iloc[:, 0]
                                        interp_cm = p.interpolation_curve.iloc[:, 1]
                                        axes[i, j].plot(interp_z, interp_cm, color='lightcoral', label='interp')

                                    axes[i, j].set_title(r'$p_{ID}$' + '= {}'.format(p.id), fontsize=11)
                                    axes[i, j].grid(b=True, which='major', alpha=0.25)
                                    axes[i, j].grid(b=True, which='minor', alpha=0.125)

                                    # determine the major and minor axis labels
                                    x_major = np.round((np.max(sim_z) - np.min(sim_z)) / 2, 1)
                                    x_minor = np.round((np.max(sim_z) - np.min(sim_z)) / 4, 1)

                                    axes[i, j].xaxis.set_major_locator(MultipleLocator(x_major))
                                    axes[i, j].xaxis.set_minor_locator(MultipleLocator(x_minor))
                                    axes[i, j].yaxis.set_major_locator(MultipleLocator(0.5))
                                    axes[i, j].yaxis.set_minor_locator(MultipleLocator(0.25))

                                    # only place labels on outer axes
                                    if j != 0:
                                        axes[i, j].set_yticklabels([])
                                    else:
                                        axes[i, j].set_ylabel(r'$c_m$')

                                    if i != n_rows - 1:
                                        axes[i, j].set_xticklabels([])
                                    else:
                                        axes[i, j].set_xlabel(r'$z_{cal}$')

                                    # get 5% of the z-span for error calculation and red figure outlining
                                    error_threshold = (np.max(sim_z) - np.min(sim_z)) * 0.05

                                    # if measured z-coord > error threshold, change axes spines color to red
                                    if np.abs(p.z_true - sim_z[np.argmax(sim_cm)]) > error_threshold:
                                        for side in ['bottom', 'left', 'right', 'top']:
                                            axes[i, j].text(0.5, 0.05, r'$\epsilon_{z}=$' + str(
                                                np.round(np.abs(p.z_true - sim_z[np.argmax(sim_cm)]), 2)),
                                                            ha='center', va='center', color='dimgray', fontsize=11)
                                            axes[i, j].spines[side].set_color('red')

                        k += 1

    return fig

def plot_every_image_particle_stack_similarity(test_col, calib_set, save_results_path, plot=False, infer_sub_image=False, min_cm=0.75,
                                               measurement_depth=1.0):
    same_id_most_correlated = []
    for img in test_col.images.values():

        for p in img.particles:
            if p.z_true is None:
                continue

            SAVE_TEST_ID = 'test_calibstacks_z{}_pid{}'.format(p.z_true, p.id)
            num_stacks = len(calib_set.calibration_stacks.values())
            color = iter(cm.nipy_spectral(np.linspace(0, 1, num_stacks)))

            if plot:
                fig, ax = plt.subplots(figsize=(10, 8))

            stack_cm_zs = []
            stack_interp_zs = []
            cms = []
            cmi = []
            for stack in calib_set.calibration_stacks.values():
                stack.infer_z(particle=p, function='sknccorr', min_cm=min_cm,
                              infer_sub_image=infer_sub_image)

                if p.z is not None and np.isnan(p.z) == False:
                    sim_z = p.similarity_curve.iloc[:, 0]
                    sim_cm = p.similarity_curve.iloc[:, 1]
                    stack_cm_zs.append(sim_z[np.argmax(sim_cm)])
                    cms.append(np.max(sim_cm))

                    if plot:
                        c = next(color)
                        ax.plot(sim_z, sim_cm, color=c, linewidth=0.5, alpha=0.5, label=r'$c_m$ ' + str(stack.id))

                    if infer_sub_image:
                        interp_z = p.interpolation_curve.iloc[:, 0]
                        interp_cm = p.interpolation_curve.iloc[:, 1]
                        cmi.append(np.max(interp_cm))
                        stack_interp_zs.append(interp_z[np.argmax(interp_cm)])
                        if plot:
                            ax.plot(interp_z, interp_cm, color=c, linewidth=2)  # , label='interp ' + str(stack.id)

                        same_id_most_correlated.append(
                            [p.z_true, p.id, stack.id, sim_z[np.argmax(sim_cm)], np.max(sim_cm),
                             interp_z[np.argmax(interp_cm)], np.max(interp_cm)])

            # if every inferred z-coordinate is NaN, must skip the loop
            if len(cms) == 0:
                continue

            if plot:
                # plot the true value
                ax.axvline(x=p.z_true, ymin=0, ymax=0.925, color='black', linestyle='--', alpha=0.85)
                ax.scatter(p.z_true, 1.00625, s=200, marker='*', color='magenta')

                # calculate stats
                best_guess_cm = np.round(stack_cm_zs[np.argmax(cms)], 3)
                best_guess_interp = np.round(stack_interp_zs[np.argmax(cmi)], 2)

                tp3cm = sorted(zip(cms, stack_cm_zs), reverse=True)[:3]
                tp3cm = [p[1] for p in tp3cm]
                top_three_cm = np.round(np.mean(tp3cm), 2)
                tp3ci = sorted(zip(cmi, stack_interp_zs), reverse=True)[:3]
                tp3ci = [p[1] for p in tp3ci]
                top_three_interp = np.round(np.mean(tp3ci), 2)

                mean_cm = np.round(np.mean(stack_cm_zs), 2)
                mean_interp = np.round(np.mean(stack_interp_zs), 2)
                std_cm = np.round(np.std(stack_cm_zs), 2)

                ax.set_title(r'$z_{true}$/h = ' + str(np.round(p.z_true, 2)) + r': $Stack_{ID}, p_{ID}$' + '= {}'.format(p.id), fontsize=13)
                ax.set_xlabel(r'$z$ / h', fontsize=12)
                #ax.set_xlim([mean_cm - 1.5*std_cm, mean_cm + 1.5*std_cm])
                ax.set_xlim([-0.01, 1.01])
                ax.set_ylim([0.795, 1.0125])
                ax.set_ylabel(r'$c_{m}$', fontsize=12)

                ax.grid(b=True, which='major', alpha=0.25)
                ax.grid(b=True, which='minor', alpha=0.125)

                textstr = '\n'.join((
                        r'$z_{true}$ / h = ' + str(np.round(p.z_true, 2)),
                        r'Best Guess $(z_{cc}, z_{interp.})$' + '= {}, {}'.format(best_guess_cm, best_guess_interp),
                        r'$Mean_{3}$: $(z_{cc}, z_{interp.})$' + '= {}, {}'.format(top_three_cm, top_three_interp),
                        r'$Mean_{all}$: $(z_{cc}, z_{interp.})$' + '= {}, {}'.format(mean_cm, mean_interp)
                ))

                if np.abs(p.z_true - top_three_cm) < 0.02:
                    boxcolor = 'springgreen'
                else:
                    boxcolor = 'lightcoral'
                props = dict(boxstyle='square', facecolor=boxcolor, alpha=0.25)
                ax.text(0.5, 0.19, textstr, verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes, bbox=props, fontsize=10)
                ax.legend(title=r'$CalibStack_{ID}$', loc='lower center', bbox_to_anchor=(0.5, 0.00625), ncol=int(np.round(num_stacks / 2)), fontsize=6)

                plt.tight_layout()
                savefigpath = join(save_results_path, SAVE_TEST_ID + '_similarity.png')
                fig.savefig(fname=savefigpath, bbox_inches='tight')
                plt.close()

    stack_results = np.array(same_id_most_correlated)
    dfstack = pd.DataFrame(data=stack_results, index=None,
                           columns=['img_id', 'p_id', 'stack_id', 'z_cm', 'max_c_cm', 'z_interp', 'max_z_interp'])
    savedata = join(save_results_path, 'gdpyt_every_stackid_results.xlsx')
    dfstack.to_excel(savedata)


def plot_bin_local_rmse_z(collection, measurement_quality, measurement_depth=1, second_plot=None):
    dfm = measurement_quality
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dfm.index, dfm.rmse_z / measurement_depth, color='tab:blue', linewidth=3)
    ax.scatter(dfm.index, dfm.rmse_z / measurement_depth, s=50, color='tab:blue', alpha=0.75)
    ax.set_xlabel(r'z ($\mu m$)', fontsize=15)
    ax.set_ylabel(r'$\sigma_{z}(z)$ / h', fontsize=15)
    ax.grid(alpha=0.25)
    ax.set_ylim([-0.0025, 0.1525])
    plt.tight_layout()
    return fig



def plot_local_rmse_uncertainty(collection, measurement_quality, measurement_depth=None, true_xy=False,
                                measurement_width=None, second_plot=None):
    """

    """
    # dataframe (data is already grouped by true_z coord)
    df = measurement_quality.copy()

    # get range of data
    z_start, z_stop = df.true_z.min(), df.true_z.max()
    z_range = z_stop - z_start
    z_step = z_range / (len(df.true_z) - 1)

    if true_xy is True:
        fig, ax = plt.subplots(nrows=2, sharex=True)

        if measurement_depth is not None:
            df['true_z'] = df['true_z'] / measurement_depth
            df['rmse_vol_z'] = df['rmse_z'] / measurement_depth
            ax[0].scatter(x=df.true_z, y=df.rmse_vol_z, color='tab:blue', zorder=2.5)
            ax[0].plot(df.true_z, df.rmse_vol_z, color='lightsteelblue')
            ax[0].set_ylabel(r'$\sigma_{z}(z)$ / h')
            #ax[0].set_ylim([0, 0.07])
            ax[0].grid(alpha=0.25)

            df['rmse_vol_xy'] = df['rmse_xy'] / measurement_width
            ax[1].scatter(x=df.true_z, y=df.rmse_vol_xy, color='tab:blue', zorder=2.5)
            ax[1].plot(df.true_z, df.rmse_vol_xy, color='lightsteelblue')
            ax[1].set_ylabel(r'$\sigma_{xy}$ / w')
            #ax[1].set_ylim([0, 0.05])
            ax[1].grid(alpha=0.25)
            ax[1].set_xlabel(r'$z$ / h')

        else:
            df['rmse_real_z'] = df['rmse_z']
            ax[0].scatter(x=df.true_z, y=df.rmse_real_z, color='tab:blue', zorder=2.5)
            ax[0].plot(df.true_z, df.rmse_real_z, color='lightsteelblue')
            ax[0].grid(alpha=0.25)
            ax[0].set_ylabel(r'$\sigma_{z}(z)$')

            ax[1].scatter(x=df.true_z, y=df.rmse_xy, color='tab:blue', zorder=2.5)
            ax[1].plot(df.true_z, df.rmse_xy, color='lightsteelblue')
            ax[1].grid(alpha=0.25)
            ax[1].set_ylabel(r'$\sigma_{xy}$')
            ax[1].set_xlabel(r'$z$')
    else:
        fig, ax = plt.subplots()
        if measurement_depth is not None:

            # normalize the z-coord from (0, 1) (but really, it's ~(0.05, 0.95) )
            df['true_z'] = (df.true_z - z_start + z_step / 2) / z_range

            # normalize the uncertainty
            df['rmse_vol_z'] = df['rmse_z'] / measurement_depth

            # plot
            ax.scatter(x=df.true_z, y=df.rmse_vol_z, color='tab:blue', zorder=2.5)
            ax.plot(df.true_z, df.rmse_vol_z, color='lightsteelblue')
            ax.set_ylabel(r'$\sigma_{z}(z)$ / h')
            ax.set_xlabel(r'$z$ / h')
            ax.grid(alpha=0.25)
        else:
            # plot
            df['rmse_vol_z'] = df['rmse_z']
            ax.scatter(x=df.true_z, y=df.rmse_vol_z, color='tab:blue', zorder=2.5)
            ax.plot(df.true_z, df.rmse_vol_z, color='lightsteelblue')
            ax.set_ylabel(r'$\sigma_{z}(z)$ $(\mu m)$')
            ax.set_xlabel(r'$z$ $(\mu m)$')
            ax.grid(alpha=0.25)

    if second_plot:
        ax2 = ax.twinx()
        if second_plot == 'cm' or second_plot == 'max_sim':
            ax2.errorbar(x=df.true_z, y=df.cm, fmt='x', color='chartreuse', ecolor='palegreen', elinewidth=1, capsize=1,
                         alpha=0.75, label=r'$c_{m}$')
            ax2.plot(df.true_z, df.cm, color='darkgreen', alpha=0.125)
            ax2.errorbar(x=df.true_z, y=df.max_sim, fmt='o', color='darkgreen', ecolor='tab:green', elinewidth=1,
                         capsize=2, label=r'$c_{interp.}$')
            ax2.plot(df.true_z, df.max_sim, color='darkgreen', alpha=0.125)
            ax2.set_ylabel(r'$C_{m}$ $(\%)$', color='darkgreen')
            ax2.set_ylim([np.min(df.cm) * 0.9875, 1.0125])
            ax2.legend()

        elif second_plot == 'num_valid_z_measure':
            ax2.errorbar(x=df.true_z, y=df.num_meas, fmt='o', color='darkgreen', ecolor='limegreen',
                         elinewidth=1, capsize=2)
            ax2.plot(df.true_z, df.num_meas, color='darkgreen', alpha=0.125)
            ax2.set_ylabel(r'$p_{num.}$', color='darkgreen')
            ax2.set_ylim([0, np.max(df.num_meas) * 1.125])

    plt.tight_layout()

    return fig

def plot_calib_col_image_stats(df):

    fig, ax = plt.subplots()
    ax.plot(df.z, df.snr_filtered, color='darkblue', alpha=1)

    ax.set_xlabel(r'$\Delta z_{true} (\mu m)$')
    ax.set_xlabel(r'$z_{true}$ / $h$')
    ax.set_ylabel(r'$SNR$ ( $\frac {\mu_p}{\sigma_I}$ )', color='darkblue')
    snrmax = int(np.round(np.max(df.snr_filtered) * 1.1, -1))
    ax.set_ylim([0, snrmax])

    ax2 = ax.twinx()
    ax2.plot(df.z, df.contour_area_mean, color='darkgreen', alpha=1)
    ax2.set_ylabel(r'$A_{p}$ $(pixels^2)$', color='darkgreen')
    ymax = int(np.round(np.max(df.contour_area_mean) * 1.1, -1))
    ax2.set_ylim([0, ymax])

    return fig

def plot_num_particles_per_image(collection):

    img_id = []
    num_particles = []
    for i in collection.images.values():
        if i.z and i.particles:
            img_id.append(i.z)
            num_particles.append(int(len(i.particles)))

    fig, ax = plt.subplots()

    # to make sure there are some images to particles or else method will error
    if len(num_particles) > 2:
        img_particles = list(zip(img_id, num_particles))
        img_particles = sorted(img_particles, key = lambda x: x[0])
        img_particles = np.vstack(img_particles)
        ax.plot(img_particles[:, 0], img_particles[:, 1], alpha=0.5)
        ax.scatter(img_particles[:, 0], img_particles[:, 1])

    ax.set_xlabel(r'$z_{true}$ / $h$')
    ax.set_ylabel(r'$p_{num}$')


def plot_particles_stats(collection, stat='area'):
    values = []
    particles_ids = []
    for img in collection.images.values():
        for p in img.particles:
            if p.z is not None and np.isnan(p.z) == False:
                values.append([p.id, p.z, p.z_true, p.area, p.solidity, p.snr, len(img.particles)])
                particles_ids.append(p.id)

    df = pd.DataFrame(data=values, columns=['id', 'z', 'true_z', 'area', 'solidity', 'snr', 'img_num_particles'])
    df = df.sort_values(by='true_z')

    particles_ids = np.unique(particles_ids)
    color = iter(cm.brg(np.linspace(0, 1, len(particles_ids))))

    # if the z-coordinate is already normalized
    if df.true_z.max() < 1.01:

        if stat == 'area':
            fig, ax = plt.subplots(figsize=(10, 8))

            min_areas_z = []
            min_areas = []
            for p in particles_ids:
                dfpp = df.loc[df['id'] == p]
                dfp = dfpp.sort_values(by='z', axis=0)

                min_areas.append([dfpp.area.min()])
                min_areas_index = dfpp.area.idxmin()
                min_areas_z.append([dfpp.loc[min_areas_index].z])

                c = next(color)
                ax.plot(dfp.z, dfp.area, color=c, alpha=0.25)
                ax.scatter(dfp.z, dfp.area, s=5, color=c, label=p)

            min_area = np.mean(min_areas)
            std_area = np.std(min_areas)
            min_area_z = np.mean(min_areas_z)
            ax.axvline(x=min_area_z, ymin=0, ymax=0.25, color='black', linestyle='--', alpha=0.75)
            ax.text(0.5, 0.3, r'$A_{p, min}$' + '(z={})= {} +/- {}'.format(np.round(min_area_z, 3), int(np.round(min_area, 0)), np.round(std_area, 1)),
                    verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes, color='black')

            ax.set_xlabel(r'$z_{true}$ / $h$')
            ax.set_ylabel(r'$A_{p}$ $(pixels^2)$')
            ymax = int(np.round(np.max(df.area) * 1.1, -1))
            ax.set_ylim([0, ymax])

            if len(particles_ids) < 40:
                ax.legend(title=r'$p_{ID}$', fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=8, fancybox=True, shadow=True)

        elif stat == 'img_num_particles':
            fig, ax = plt.subplots(figsize=(10, 8))

            ax.plot(df.true_z, df.img_num_particles, color='tab:blue', alpha=0.25)
            ax.scatter(df.true_z, df.img_num_particles, s=5, color='tab:blue')

            ax.set_xlabel(r'$z_{true}$ / $h$')
            ax.set_ylabel('num. particles')
            ymax = int(np.round(np.max(df.img_num_particles) * 1.1, -1))
            ax.set_ylim([0, ymax])

    else:
        if stat == 'area':
            fig, ax = plt.subplots(figsize=(10, 8))

            min_areas_z = []
            min_areas = []
            for p in particles_ids:
                dfpp = df.loc[df['id'] == p]
                dfp = dfpp.sort_values(by='z', axis=0)

                min_areas.append([dfpp.area.min()])
                min_areas_index = dfpp.area.idxmin()
                min_areas_z.append([dfpp.loc[min_areas_index].z])

                c = next(color)
                ax.plot(dfp.z, dfp.area, color=c, alpha=0.25)
                ax.scatter(dfp.z, dfp.area, s=5, color=c, label=p)

            min_area = np.mean(min_areas)
            std_area = np.std(min_areas)
            min_area_z = np.mean(min_areas_z)
            ax.axvline(x=min_area_z, ymin=0, ymax=0.25, color='black', linestyle='--', alpha=0.75)
            ax.text(0.5, 0.3,
                    r'$A_{p, min}$' + '(z={})= {} +/- {}'.format(np.round(min_area_z, 3), int(np.round(min_area, 0)),
                                                                 np.round(std_area, 1)),
                    verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes, color='black')

            ax.set_xlabel(r'$z_{true}$')
            ax.set_ylabel(r'$A_{p}$ $(pixels^2)$')
            ymax = int(np.round(np.max(df.area) * 1.1, -1))
            # ax.set_ylim([0, ymax])
            if len(particles_ids) < 40:
                ax.legend(title=r'$p_{ID}$', fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=8,
                          fancybox=True, shadow=True)

        elif stat == 'img_num_particles':
            fig, ax = plt.subplots(figsize=(8, 10))

            ax.plot(df.true_z, df.img_num_particles, color='tab:blue', alpha=0.25)
            ax.scatter(df.true_z, df.img_num_particles, s=5, color='tab:blue')

            ax.set_xlabel(r'$z_{true}$')
            ax.set_ylabel('num. particles')
            ymax = np.max([1.1, int(np.round(np.max(df.img_num_particles) * 1.1, -1))])
            ax.set_ylim([0, ymax])

    return fig

def plot_particle_snr_and(collection, second_plot='area', particle_id=None):

    values = []
    for img in collection.images.values():

        for p in img.particles:

            # if a particle_id is passed, only get data for that particle ID; else, get data for all particle ID's
            if particle_id:
                if p.id != particle_id:
                    continue

            # only get data for particles with a valid z-coordinate
            if p.z is not None and np.isnan(p.z) == False and p.snr:
                if collection.image_collection_type == 'test':
                    values.append([p.id, p.z, p.z_true, p.area, p.solidity, p.snr, len(img.particles), p.cm, p.max_sim])
                else:
                    values.append([p.id, p.z, p.z_true, p.area, p.solidity, p.snr, len(img.particles)])


    # if plotting a test collection, we can access the correlation values: cm (correlation) and max_sim (interpolated)
    if collection.image_collection_type == 'test':
        df = pd.DataFrame(data=values, columns=['id', 'z', 'true_z', 'area', 'solidity', 'snr', 'img_num_particles',
                                                'cm', 'max_sim'])
        if not particle_id:
            df = collection.bin_local_quantities(df, min_cm=0.9)
    else:
        df = pd.DataFrame(data=values, columns=['id', 'z', 'true_z', 'area', 'solidity', 'snr', 'img_num_particles'])

        if not particle_id:
            df = collection.bin_local_quantities(df, min_cm=None)

    # create the figure now so we can use modifiers downstream
    fig, ax = plt.subplots()

    # round the true_z value (which is not important for this plot so we do it early)
    # we have to correct to the correct decimal place to get at least 10-20 data points depending on our measurement
    # range (which can be 0-1 for normalized analyses or 0-NUM_TEST_IMAGES * Z_STEP_PER for meta analyses)
    # we can now round the z-value in order to perform an aggregate analysis

    # group by true_z to get aggregate values to plot
    df_z_count = df.groupby(['true_z']).count()
    df_z_mean = df.groupby(['true_z']).mean()
    df_z_std = df.groupby(['true_z']).std()

    ax.errorbar(x=df_z_mean.index, y=df_z_mean.snr, yerr=df_z_std.snr, fmt='o', color='darkblue', ecolor='lightblue', elinewidth=1, capsize=2)
    ax.plot(df_z_mean.index, df_z_mean.snr, color='darkblue', alpha=0.25)

    ax.set_ylabel(r'$SNR$ ( $\frac {\mu_p}{\sigma_I}$ )', color='darkblue')
    snrmax = int(np.round(np.max(df_z_mean.snr) * 1.35, -1))
    ax.set_ylim([0, snrmax])

    if second_plot == 'area':
        ax2 = ax.twinx()
        ax2.errorbar(x=df_z_mean.index, y=df_z_mean.area, yerr=df_z_std.area, fmt='o', color='darkgreen', ecolor='limegreen', elinewidth=1, capsize=2)
        ax2.plot(df_z_mean.index, df_z_mean.area, color='darkgreen', alpha=0.125)
        ax2.set_ylabel(r'$A_{p}$ $(pixels^2)$', color='darkgreen')
        ymax = int(np.round(np.max(df_z_mean.area) * 1.1, -2))
        if ymax < 25:
            ymax = df.area.max()
        ax2.set_ylim([0, ymax])

    elif second_plot == 'solidity':
        ax2 = ax.twinx()
        ax2.errorbar(x=df_z_mean.index, y=df_z_mean.solidity, yerr=df_z_std.solidity, fmt='o', color='darkgreen', ecolor='limegreen', elinewidth=1, capsize=2)
        ax2.plot(df_z_mean.index, df_z_mean.solidity, color='darkgreen', alpha=0.125)
        ax2.set_ylabel(r'$Solidity$ $(\%)$', color='darkgreen')
        ax2.set_ylim([0.8, 1.0125])

    elif second_plot == 'percent_measured':
        per_measured_particles = []
        for ind in df_z_count.index:
            per_measured_particles.append([ind, df_z_count['z'][ind] / df_z_mean['img_num_particles'][ind] * 100])
        percent_measured_particles = np.array(per_measured_particles, dtype=float)

        ax2 = ax.twinx()
        ax2.scatter(percent_measured_particles[:, 0], percent_measured_particles[:, 1], color='darkgreen')
        ax2.plot(percent_measured_particles[:, 0], percent_measured_particles[:, 1], color='darkgreen', alpha=0.25)
        ax2.set_ylabel(r'$Measured$ $Particles$ $(\%)$', color='darkgreen')
        ax2.set_ylim([49.9, 100.1])

    elif second_plot == 'cm' or second_plot == 'max_sim':
        ax2 = ax.twinx()
        ax2.errorbar(x=df_z_mean.index, y=df_z_mean.cm, yerr=df_z_std.cm, fmt='x', color='chartreuse',
                     ecolor='palegreen', elinewidth=1, capsize=1, alpha=0.75, label=r'$c_{m}$')
        ax2.plot(df_z_mean.index, df_z_mean.cm, color='darkgreen', alpha=0.125)
        ax2.errorbar(x=df_z_mean.index, y=df_z_mean.max_sim, yerr=df_z_std.max_sim, fmt='o', color='darkgreen',
                     ecolor='tab:green', elinewidth=1, capsize=2, label=r'$c_{interp.}$')
        ax2.plot(df_z_mean.index, df_z_mean.max_sim, color='darkgreen', alpha=0.125)
        ax2.set_ylabel(r'$C_{m}$ $(\%)$', color='darkgreen')
        ax2.set_ylim([np.min(df.cm) * 0.99, np.max(df_z_mean.max_sim) * 1.001])

    plt.tight_layout()

    return fig


def plot_particle_peak_intensity(collection, particle_id=None):

    values = []
    if collection.image_collection_type == 'calibration':
        if particle_id is not None:
            for img in collection.images.values():
                for p in img.particles:
                    if int(p.id) == particle_id and p.z_true is not None:
                        values.append([p.id, p.z_true, p.peak_intensity])
        else:
            for img in collection.images.values():
                for p in img.particles:
                    if p.z_true is not None:
                        values.append([p.id, p.z_true, p.peak_intensity])

        df = pd.DataFrame(data=values, columns=['id', 'true_z', 'peak_intensity'])

        # group by true_z to get aggregate values to plot
        df_z_count = df.groupby(['true_z']).count()
        df_z_mean = df.groupby(['true_z']).mean()
        df_z_std = df.groupby(['true_z']).std()
    else:
        if particle_id is not None:
            for img in collection.images.values():
                for p in img.particles:
                    if int(p.id) == particle_id:
                        if not np.isnan(p.z):
                            values.append([p.id, p.z, p.peak_intensity])
        else:
            for img in collection.images.values():
                for p in img.particles:
                    if not np.isnan(p.z):
                        values.append([p.id, p.z, p.peak_intensity])

        if len(values) == 0:
            return None

        df = pd.DataFrame(data=values, columns=['id', 'z', 'peak_intensity'])

        # group by true_z to get aggregate values to plot
        df_z_count = df.groupby(['z']).count()
        df_z_mean = df.groupby(['z']).mean()
        df_z_std = df.groupby(['z']).std()

    fig, ax = plt.subplots()

    ax.errorbar(x=df_z_mean.index, y=df_z_mean.peak_intensity, yerr=df_z_std.peak_intensity, fmt='o', color='darkblue', ecolor='lightblue',
                elinewidth=1, capsize=2)
    ax.plot(df_z_mean.index, df_z_mean.peak_intensity, color='darkblue', alpha=0.25)
    ax.set_ylabel(r'$I_{max}$ (A.U.)', color='darkblue')
    if collection.image_collection_type == 'calibration':
        ax.set_xlabel(r'$z_{true}$ ($\mu m$)')
    else:
        ax.set_xlabel(r'$z$ ($\mu m$)')

    ax2 = ax.twinx()
    ax2.scatter(df_z_mean.index, df_z_count.peak_intensity, color='darkgreen', alpha=0.25)
    ax2.plot(df_z_mean.index, df_z_count.peak_intensity, color='darkgreen', alpha=0.2)
    ax2.set_ylabel(r'$p_{num}$', color='darkgreen')

    plt.tight_layout()

    return fig


def plot_particle_signal(collection, optics, collection_image_stats, particle_id, intensity_max_or_mean='max'):

    # get particle data
    values = []
    for img in collection.images.values():
        for p in img.particles:
            if p.id == particle_id:
                    values.append([p.id, p.z_true, p.area, p.diameter, p.snr, p.peak_intensity, p.mean_signal,
                                   p.mean_background, p.std_background])

    dfp = pd.DataFrame(data=values, columns=['id', 'true_z', 'area', 'diameter', 'snr', 'peak_signal', 'mean_signal',
                                             'mean_background', 'std_background'])
    dfp = dfp.sort_values(by='true_z')

    # calculate background + noise
    dfp['noise_level'] = dfp['mean_background'] + dfp['std_background'] * 2

    # calculate the mean background intensity
    background_mean = dfp.mean_background.mean()

    # get value of maximum particle intensity (signal) and it's z-coordinate
    if intensity_max_or_mean == 'max':
        max_intensity_in_focus = dfp.peak_signal.max()
        z_max_intensity = dfp.true_z.iloc[dfp[['peak_signal']].idxmax().to_numpy()[0]]
    else:
        max_intensity_in_focus = dfp.mean_signal.max()
        z_max_intensity = dfp.true_z.iloc[dfp[['mean_signal']].idxmax().to_numpy()[0]]

    # get stigmatic intensity profile
    z_profile, stigmatic_intensity_profile = optics.stigmatic_maximum_intensity_z(z_space=dfp.true_z,
                                                                                  max_intensity_in_focus=max_intensity_in_focus,
                                                                                  z_zero=z_max_intensity,
                                                                                  background_intensity=background_mean)
    # plot
    fig, ax = plt.subplots()

    # theoretical intensity profile
    ax.plot(z_profile, stigmatic_intensity_profile, color='cornflowerblue', alpha=0.5, label=r'theory')

    # measured intensity (signal)
    if intensity_max_or_mean == 'max':
        ax.scatter(dfp.true_z, dfp.peak_signal, marker='s', color='mediumblue', label=r'signal', zorder=2.4)
        ax.plot(dfp.true_z, dfp.peak_signal, color='tab:blue', alpha=0.75)
    else:
        ax.scatter(dfp.true_z, dfp.mean_signal, marker='s', color='mediumblue', label=r'signal', zorder=2.4)
        ax.plot(dfp.true_z, dfp.mean_signal, color='tab:blue', alpha=0.75)

    # background + noise
    ax.plot(dfp.true_z, dfp.mean_background, color='dimgray', alpha=0.5, linestyle='--', label='background')
    ax.plot(dfp.true_z, dfp.noise_level, color='goldenrod', alpha=0.5, linestyle='--', label='peak noise')

    ax.set_xlabel(r'$z_{true}$')
    ax.set_ylabel(r'$Intensity$')
    ax.set_title(r'M{}, {}NA, f{}, pd{}'.format(optics.effective_magnification, optics.numerical_aperture,
                                                optics.focal_length, optics.particle_diameter))
    ax.legend()
    ax.grid(alpha=0.25)

    return fig


def plot_particle_diameter(collection, optics, collection_image_stats, particle_id):
    fig = None
    return fig


def plot_gaussian_ax_ay(collection, plot_type='one', p_inspect=[0]):

    if plot_type not in ['one', 'subset', 'all', 'mean']:
        raise ValueError("Not a valid plot type. Try: 'one', 'all', 'subset', 'mean'")

    values = []
    for img in collection.images.values():
        for p in img.particles:
            if p._fitted_gaussian_on_template is None:
                pass
            else:
                p_ax = p._fitted_gaussian_on_template['ax']
                p_ay = p._fitted_gaussian_on_template['ax']
                p_A = p._fitted_gaussian_on_template['A']
                values.append([p.id, p.z, p.z_true, p.area, p_ax, p_ay, p_A, p.snr])

    fig, ax = plt.subplots()

    # if no gaussian fitting
    if len(values) < 2:
        return fig

    df = pd.DataFrame(data=values, columns=['id', 'z', 'true_z', 'area', 'p_ax', 'p_ay', 'p_A', 'snr'])
    df = df.sort_values(by='true_z')
    z_steps = len(df.true_z.unique())

    ax2 = ax.twinx()

    if plot_type in ['one', 'subset', 'all']:
        if plot_type == 'all':
            p_inspect = df.id.unique()

        color_ax = iter(cm.nipy_spectral(np.linspace(0, 0.95, len(p_inspect))))
        color_ay = iter(cm.nipy_spectral(np.linspace(0.05, 1, len(p_inspect))))

        for pid in p_inspect:
            dfp = df.loc[df['id'] == pid]

            # filter out dataframes with a high standard deviation between z-steps
            dfp_snr = np.abs(dfp.p_ax.diff().mean()) / dfp.p_ax.diff().std()

            if dfp_snr > 0.01 and len(dfp) > z_steps * 0.5: # TODO: this needs some testing
                cax = next(color_ax)
                cay = next(color_ay)

                ax.scatter(dfp.true_z, dfp.p_ax, s=10, marker='o', color=cax, label=r'$\alpha_{x}$', zorder=2.5)
                ax.plot(dfp.true_z, dfp.p_ax, color=cax, alpha=0.5)

                ax.scatter(dfp.true_z, dfp.p_ay, s=10, marker='x', color=cay, label=r'$\alpha_{y}$', zorder=2.4)
                ax.plot(dfp.true_z, dfp.p_ay, color=cay, alpha=0.5)

                ax2.scatter(dfp.true_z, dfp.p_A, marker='.', color='tab:gray', alpha=0.25, label='A')
                ax2.plot(dfp.true_z, dfp.p_A, color='tab:gray', alpha=0.125)

    elif plot_type == 'mean':
        dfp = df.groupby(by='true_z').mean()

        ax.scatter(dfp.index, dfp.p_ax, marker='o', color='tab:blue', label=r'$\alpha_{x}$', zorder=2.5)
        ax.plot(dfp.index, dfp.p_ax, color='tab:blue', alpha=0.5)

        ax.scatter(dfp.index, dfp.p_ay, marker='s', color='mediumblue', label=r'$\alpha_{y}$', zorder=2.4)
        ax.plot(dfp.index, dfp.p_ay, color='mediumblue', alpha=0.5)

        ax2.scatter(dfp.index, dfp.p_A, marker='.', color='tab:gray', alpha=0.25, label='A')
        ax2.plot(dfp.index, dfp.p_A, color='tab:gray', alpha=0.125)


    ax.set_xlabel(r'$z_{true}$ / $h$')
    ax.set_ylabel(r'$\alpha_{x,y}$')
    ax.grid(alpha=0.25)

    ax2.set_ylabel('Amplitude', color='tab:gray')

    if plot_type not in ['subset', 'all']:
        ax.legend()

    plt.tight_layout()

    return fig


def plot_gaussian_fit_on_image_for_particle(collection, particle_ids, frame_step_size, path_figs):

    if isinstance(particle_ids, (int, float)):
        particle_ids = [int(particle_ids)]

    for img in collection.images.values():

        if img.frame % frame_step_size == 0:

            for p in img.particles:
                if p.id in particle_ids:

                    # get gaussian params
                    gauss_xc = p.gauss_xc
                    gauss_yc = p.gauss_yc
                    xc = gauss_xc - p.bbox[0]
                    yc = gauss_yc - p.bbox[1]
                    gauss_dia_x = p.gauss_dia_x
                    gauss_dia_y = p.gauss_dia_y
                    sigmax = p.gauss_sigma_x
                    sigmay = p.gauss_sigma_y

                    if all([gauss_yc, gauss_xc, gauss_dia_y, gauss_dia_x]) is False:
                        continue

                    # get images
                    img = p.image_raw.copy()
                    p_template = p.template.copy()

                    img = rescale_intensity(img, out_range=np.uint16)
                    p_template = rescale_intensity(p_template, out_range=np.uint16)

                    # generate ellipse
                    rr, cc = ellipse_perimeter(int(gauss_yc), int(gauss_xc), int(gauss_dia_y/2), int(gauss_dia_x/2))
                    rr, cc = filter_ellipse_points_on_image(img, rr, cc)
                    img[rr, cc] = np.max(img)

                    # plot
                    fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(10, 5))

                    # full image
                    ax1.imshow(img)
                    ax1.scatter(gauss_xc, gauss_yc, s=0.5, color='red', label=r'$p_{ID}$'
                                                                    + '={} (x={}, y={})'.format(p.id,
                                                                                                np.round(gauss_xc, 2),
                                                                                                np.round(gauss_yc, 2),
                                                                                                )
                                                                    + '\n'
                                                                    + r'$d_{e} \: (x, y)=$'
                                                                    + '({} x, {} y)'.format(np.round(gauss_dia_x, 2),
                                                                                            np.round(gauss_dia_y, 2),
                                                                                            )
                                )
                    ax1.legend()

                    # template
                    ax2.imshow(p_template)
                    ax2.scatter(xc, yc, color='red')
                    ax2.plot([xc - sigmax / 2, xc + sigmax / 2], [yc, yc], linewidth=2, color='red', label=r'$\sigma_{x}$')
                    ax2.plot([xc, xc], [yc - sigmay / 2, yc + sigmay / 2], linewidth=2, color='blue', label=r'$\sigma_{y}$')
                    ax2.legend()

                    plt.tight_layout()
                    plt.savefig(join(path_figs, 'pid{}_x{}_y{}_frame{}.png'.format(p.id,
                                                                                   np.round(gauss_xc, 2),
                                                                                   np.round(gauss_yc, 2),
                                                                                   p.frame,
                                                                                   )
                                     )
                                )
                    plt.close()


def plot_model_based_z_calibration(collection, plot_type='one', p_inspect=[0]):

    if plot_type not in ['one', 'subset', 'all', 'mean']:
        raise ValueError("Not a valid plot type. Try: 'one', 'all', 'subset', 'mean'")

    values = []
    for img in collection.images.values():
        for p in img.particles:
            p_ax = p._fitted_gaussian_on_template['ax']
            p_ay = p._fitted_gaussian_on_template['ax']
            p_A = p._fitted_gaussian_on_template['A']
            values.append([p.id, p.z, p.z_true, p.area, p_ax, p_ay, p_A, p.snr])

    df = pd.DataFrame(data=values, columns=['id', 'z', 'true_z', 'area', 'p_ax', 'p_ay', 'p_A', 'snr'])
    df = df.sort_values(by='true_z')
    z_steps = len(df.true_z.unique())

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    if plot_type in ['one', 'subset', 'all']:
        if plot_type == 'all':
            p_inspect = df.id.unique()

        color_ax = iter(cm.nipy_spectral(np.linspace(0, 0.95, len(p_inspect))))
        color_ay = iter(cm.nipy_spectral(np.linspace(0.05, 1, len(p_inspect))))

        for pid in p_inspect:
            dfp = df.loc[df['id'] == pid]

            if len(dfp.true_z) > 0.5 * z_steps:
                cax = next(color_ax)
                cay = next(color_ay)

                ax.scatter(dfp.true_z, dfp.p_ax, marker='o', color=cax, label=r'$\alpha_{x}$', zorder=2.5)
                ax.plot(dfp.true_z, dfp.p_ax, color=cax, alpha=0.5)

                ax.scatter(dfp.true_z, dfp.p_ay, marker='x', color=cay, label=r'$\alpha_{y}$', zorder=2.4)
                ax.plot(dfp.true_z, dfp.p_ay, color=cay, alpha=0.5)

                ax2.scatter(dfp.true_z, dfp.p_A, marker='.', color='tab:gray', alpha=0.25, label='A')
                ax2.plot(dfp.true_z, dfp.p_A, color='tab:gray', alpha=0.125)

    elif plot_type == 'mean':
        dfp = df.groupby(by='true_z').mean()

        ax.scatter(dfp.index, dfp.p_ax, marker='o', color='tab:blue', label=r'$\alpha_{x}$', zorder=2.5)
        ax.plot(dfp.index, dfp.p_ax, color='tab:blue', alpha=0.5)

        ax.scatter(dfp.index, dfp.p_ay, marker='s', color='mediumblue', label=r'$\alpha_{y}$', zorder=2.4)
        ax.plot(dfp.index, dfp.p_ay, color='mediumblue', alpha=0.5)

        ax2.scatter(dfp.index, dfp.p_A, marker='.', color='tab:gray', alpha=0.25, label='A')
        ax2.plot(dfp.index, dfp.p_A, color='tab:gray', alpha=0.125)


    ax.set_xlabel(r'$z_{true}$ / $h$')
    ax.set_ylabel(r'$\alpha_{x,y}$')
    ax.grid(alpha=0.25)

    ax2.set_ylabel('Amplitude', color='tab:gray')

    if plot_type not in ['subset', 'all']:
        ax.legend()

    plt.tight_layout()

    return fig





def plot_tensor_dset(dset, N):
    n_images = N
    n_cols = min(10, n_images)
    n_rows = int(n_images / n_cols) + 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(2 * n_cols, n_rows * 2))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes]).reshape(-1,1)

    idx_list = np.random.randint(0, len(dset), size=N)

    vmin = []
    vmax = []
    for idx in idx_list:
        img = dset[idx]['input']
        vmin.append(img.min())
        vmax.append(img.max())

    vmin = min(vmin)
    vmax = max(vmax)

    for i in range(n_rows):
        for j in range(n_cols):
            n = i * n_cols + j
            if n > n_images - 1:
                im = axes[i, j].imshow(np.zeros_like(sample['input'].numpy().squeeze()), cmap='gray',
                                       vmin=vmin, vmax=vmax)
                axes[i, j].set_title('None', fontsize=5)
            else:
                sample = dset[idx_list[n]]
                im = axes[i, j].imshow(sample['input'].numpy().squeeze(), cmap='gray',
                                       vmin=vmin, vmax=vmax)
                if dset.mode == 'train':
                    axes[i, j].set_title('z = {}'.format(sample['target']), fontsize=5)
            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)

    fig.subplots_adjust(right=0.91)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # fig.suptitle('Tensor dataset (Particle ID {})'.format(stack.id))
    fig.subplots_adjust(wspace=0.05, hspace=0.25)

    return fig

def plot_image(img, cmap='gray'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(img, cmap=cmap)

    return fig

def plot_stacks_self_similarity(calib_set, min_num_layers=0, save_string=None):

    color = iter(cm.gist_rainbow(np.linspace(0, 1, len(calib_set.calibration_stacks.values()))))
    min_cm = 0.5

    fig, ax = plt.subplots()

    num_plots = 0
    for stack in calib_set.calibration_stacks.values():
        if len(stack.layers) >= min_num_layers:

            if np.min(stack.self_similarity[:, 1]) < min_cm:
                min_cm = np.min(stack.self_similarity[:, 1]) * 0.95

            c = next(color)
            ax.plot(stack.self_similarity[:, 0], stack.self_similarity[:, 1], color=c, alpha=0.5)
            ax.scatter(stack.self_similarity[:, 0], stack.self_similarity[:, 1], s=3, color=c, label=stack.id)

            num_plots += 1

    ax.set_xlabel(r'$z$ / h')
    ax.set_ylabel(r'$S_i$ $\left(|z_{i-1}, z_{i+1}|\right)$')

    ax.set_ylim([min_cm, 1.005])
    ax.grid(alpha=0.25)

    if num_plots < 10:
        ax.legend(title=r'$S_i$')

    return fig

# Helper Functions

def filter_ellipse_points_on_image(img, rr, cc):
    rn = []
    cn = []

    for r, c in zip(rr, cc):
        if r < 0 or c < 0:
            continue
        elif r > img.shape[0] - 1 or c > img.shape[1] - 1:
            continue
        else:
            rn.append(r)
            cn.append(c)

    return rn, cn