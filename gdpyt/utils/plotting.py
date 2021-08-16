from os.path import splitext
import re
import logging

from math import floor
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.draw import polygon
from skimage import exposure
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib as mpl

logger = logging.getLogger(__name__)

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
                axes[i, j].imshow(template, cmap='viridis')
                particle_z = [p for p in stack.particles if p.z == z]

                for index, p_z in enumerate(particle_z):
                    connected_contour = np.vstack([p_z.template_contour, p_z.template_contour[0, :]])
                    axes[i, j].plot(connected_contour[:, 0], connected_contour[:, 1], linewidth=0.5, color='red')

                    if fill_contours is True:
                        rr, cc = polygon(connected_contour[:, 1], connected_contour[:, 0], template.shape)
                        template_contour = template.copy()
                        template_contour[rr, cc] = np.max(template)
                        axes[i, j].imshow(template_contour, cmap='viridis')

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
    ax.plot(calib_stack.self_similarity[:, 0], calib_stack.self_similarity[:, 1])
    ax.scatter(calib_stack.self_similarity[:, 0], calib_stack.self_similarity[:, 1])
    ax.set_xlabel(r'$z$ / h')
    ax.set_ylabel(r'$S_i$ $\left(|z_{i-1}, z_{i+1}|\right)$')
    ax.set_ylim([0.7475, 1.005])
    ax.grid(alpha=0.25)
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

    if measurement_depth is not None and true_xy is True:
        df['rmse_xy'] = (df['rmse_x'] ** 2 + df['rmse_y'] ** 2) ** 0.5
        df['rmse_vol_xy'] = df['rmse_xy'] / measurement_width
        df['rmse_vol_z'] = df['rmse_z'] / 1
        if plot_type == 'scatter':
           ax.scatter(x=df.true_z, y=df.z)
        if plot_type == 'errorbars':
            ax.errorbar(x=df.true_z, y=df.z, yerr=df.rmse_vol_z, xerr=df.rmse_vol_xy, fmt='o', ecolor='gray', elinewidth=1, capsize=2, label='Measured')
        ax.set_xlabel(r'$z_{true}$ / h')
        ax.set_ylabel(r'$z_{measured}$ / h')

    elif measurement_depth is None and true_xy is True:
        if plot_type == 'scatter':
           ax.scatter(x=df.true_z, y=df.z)
        if plot_type == 'errorbars':
            ax.errorbar(x=df.true_z, y=df.z, yerr=df.rmse_z, xerr=df.rmse_xy, fmt='o', ecolor='gray', elinewidth=1, capsize=2, label='Measured')
        ax.set_xlabel(r'$\Delta z_{true} (\mu m)$')
        ax.set_ylabel(r'$z_{measured}$ $(\mu m)$')

    elif measurement_depth is not None and true_xy is False:
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
                                axes[i, j].set_title(r'$z_{true}$/h = ' + str(np.round(p.z_true,2)), fontsize=11)
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

                                axes[i, j].set_title(r'$z_{true}$/h = ' + str(np.round(p.z_true,2)), fontsize=11)
                                axes[i, j].grid(b=True, which='major', alpha=0.25)
                                axes[i, j].grid(b=True, which='minor', alpha=0.125)
                                axes[i, j].xaxis.set_major_locator(MultipleLocator(0.5))
                                axes[i, j].xaxis.set_minor_locator(MultipleLocator(0.1))
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
                                    axes[i, j].set_xlabel(r'$z_{cal}$ / h')

                                # if measured z-coord > error threshold, change axes spines color to red
                                if np.abs(p.z_true - sim_z[np.argmax(sim_cm)]) > 0.05:
                                    for side in ['bottom', 'left', 'right', 'top']:
                                        axes[i, j].text(0.5, 0.05, r'$\epsilon_{z}=$'+str(np.round(np.abs(p.z_true - sim_z[np.argmax(sim_cm)]), 2)),
                                                        ha='center', va='center', color='dimgray', fontsize=11)
                                        axes[i, j].spines[side].set_color('red')

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

                max_ind = np.argmax(sim_cm)
                sim_z_near_max = sim_z[max_ind-2:max_ind+3]
                sim_cm_near_max = sim_cm[max_ind-2:max_ind+3]
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

                axes[0].set_ylim(bottom=0, top=1.05)
                axes[0].set_ylabel(r'$c_m$')
                axes[0].set_title(r'$z_{true}$/h = ' + str(np.round(p.z_true, 2)), fontsize=11)
                axes[0].grid(b=True, which='major', alpha=0.25)
                axes[0].grid(b=True, which='minor', alpha=0.125)
                axes[0].xaxis.set_major_locator(MultipleLocator(0.25))
                axes[0].xaxis.set_minor_locator(MultipleLocator(0.05))
                axes[0].yaxis.set_major_locator(MultipleLocator(0.25))
                axes[0].yaxis.set_minor_locator(MultipleLocator(0.05))
                axes[0].legend()

                axes[1].set_ylim(bottom=np.max([np.min(sim_cm_near_max), 0.8]), top=1.025)
                axes[1].set_ylabel(r'$c_m$')
                axes[1].set_xlabel(r'$z_{cal}$ / h')
                axes[1].grid(b=True, which='major', alpha=0.25)
                axes[1].grid(b=True, which='minor', alpha=0.125)
                axes[1].xaxis.set_major_locator(MultipleLocator(0.025))
                axes[1].xaxis.set_minor_locator(MultipleLocator(0.005))
                axes[1].yaxis.set_major_locator(MultipleLocator(0.025))
                axes[1].yaxis.set_minor_locator(MultipleLocator(0.005))
                axes[1].legend(fontsize=8, loc='upper right')

        else:
            n_images = len(img_list[image_id][1].particles)
            if n_images > 54:
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
                        inspectvar = img_list[image_id][1].particles
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
                                    axes[i, j].xaxis.set_major_locator(MultipleLocator(0.5))
                                    axes[i, j].xaxis.set_minor_locator(MultipleLocator(0.1))
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
                                        axes[i, j].set_xlabel(r'$z_{cal}$ / h')

                                    # if measured z-coord > error threshold, change axes spines color to red
                                    if np.abs(p.z_true - sim_z[np.argmax(sim_cm)]) > 0.05:
                                        for side in ['bottom', 'left', 'right', 'top']:
                                            axes[i, j].text(0.5, 0.05, r'$\epsilon_{z}=$' + str(
                                                np.round(np.abs(p.z_true - sim_z[np.argmax(sim_cm)]), 2)),
                                                            ha='center', va='center', color='dimgray', fontsize=11)
                                            axes[i, j].spines[side].set_color('red')

                        k += 1

    return fig


def plot_local_rmse_uncertainty(collection, measurement_quality, measurement_depth=None, true_xy=False, measurement_width=None):

    df = measurement_quality

    if true_xy is True:
        fig, ax = plt.subplots(nrows=2, sharex=True)

        if measurement_depth is not None:
            #df['true_z_vol'] = df['true_z'] / measurement_depth # TODO: fix measurement depth and volume plotting
            df['rmse_vol_z'] = df['rmse_z'] / 1
            ax[0].scatter(x=df.true_z, y=df.rmse_vol_z, color='tab:blue')
            ax[0].plot(df.true_z, df.rmse_vol_z, color='lightsteelblue')
            ax[0].set_ylabel(r'$\sigma_{z}(z)$ / h')
            #ax[0].set_ylim([0, 0.07])

            df['rmse_vol_xy'] = df['rmse_xy'] / measurement_width
            ax[1].scatter(x=df.true_z, y=df.rmse_vol_xy, color='tab:blue')
            ax[1].plot(df.true_z, df.rmse_vol_xy, color='lightsteelblue')
            ax[1].set_ylabel(r'$\sigma_{xy}$ / w')
            #ax[1].set_ylim([0, 0.05])
            ax[1].set_xlabel(r'$z$ / h')

        else:
            MEASUREMENT_DEPTH = 86 # TODO: resolve measurement depth in this plot
            df['rmse_real_z'] = df['rmse_z'] * 86
            ax[0].scatter(x=df.true_z, y=df.rmse_real_z, color='tab:blue')
            ax[0].plot(df.true_z, df.rmse_real_z, color='lightsteelblue')
            ax[0].set_ylabel(r'$\sigma_{z}(z)$')

            ax[1].scatter(x=df.true_z, y=df.rmse_xy, color='tab:blue')
            ax[1].plot(df.true_z, df.rmse_xy, color='lightsteelblue')
            ax[1].set_ylabel(r'$\sigma_{xy}$')
            ax[1].set_xlabel(r'$z$')
    else:
        fig, ax = plt.subplots()
        if measurement_depth is not None:
            df['rmse_vol_z'] = df['rmse_z'] / 1
            ax.scatter(x=df.true_z, y=df.rmse_vol_z, color='tab:blue')
            ax.plot(df.true_z, df.rmse_vol_z, color='lightsteelblue')
            ax.set_ylabel(r'$\sigma_{z}(z)$ / h')
            ax.set_xlabel(r'$z$ / h')
            # ax[0].set_ylim([0, 0.07])
        else:
            df['rmse_vol_z'] = df['rmse_z'] * 81
            ax.scatter(x=df.true_z, y=df.rmse_vol_z, color='tab:blue')
            ax.plot(df.true_z, df.rmse_vol_z, color='lightsteelblue')
            ax.set_ylabel(r'$\sigma_{z}(z)$')
            ax.set_xlabel(r'$z$ / h')

    #ax.set_title('Measurement uncertainty: root mean squared error', fontsize=8)
    #ax.legend()
    plt.tight_layout()

    return fig

def plot_calib_col_image_stats(df):

    fig, ax = plt.subplots()
    ax.plot(df.z, df.snr_filtered, color='darkblue', alpha=1)

    ax.set_xlabel(r'$\Delta z_{true} (\mu m)$')
    ax.set_xlabel(r'$z_{true}$ / $h$')
    ax.set_ylabel(r'$SNR$ ( $\frac {\mu_p}{\sigma_I}$ )', color='darkblue')
    ax.set_ylim([0, 150])

    ax2 = ax.twinx()
    ax2.plot(df.z, df.contour_area_mean, color='darkgreen', alpha=1)
    ax2.set_ylabel(r'$A_{p}$ $(pixels^2)$', color='darkgreen')
    ax2.set_ylim([0, 1500])

def plot_particles_stats(collection, stat='area'):
    values = []
    particles_ids = []
    for img in collection.images.values():
        for p in img.particles:
            if p.z is not None and np.isnan(p.z) == False:
                values.append([p.id, p.z, p.z_true, p.area, p.solidity, p.snr, len(img.particles)])
                particles_ids.append(p.id)

    df = pd.DataFrame(data=values, columns=['id', 'z', 'true_z', 'area', 'solidity', 'snr', 'img_num_particles'])

    particles_ids = np.unique(particles_ids)

    fig, ax = plt.subplots(figsize=(10, 8))
    for p in particles_ids:
        dfpp = df.loc[df['id'] == p]
        dfp = dfpp.sort_values(by='z', axis=0)
        ax.plot(dfp.z, dfp.area, alpha=0.5)
        ax.scatter(dfp.z, dfp.area, label=p)

    ax.set_xlabel(r'$z_{true}$ / $h$')
    ax.set_ylabel(r'$A_{p}$ $(pixels^2)$')
    ax.set_ylim([0, 1750])
    ax.legend(title=r'$p_{ID}$', fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=8, fancybox=True, shadow=True)
    plt.tight_layout()
    return fig

def plot_particle_snr_and(collection, second_plot='area'):
    values = []
    identified_particles_per_image = []
    for img in collection.images.values():
        for p in img.particles:
            if p.z is not None and np.isnan(p.z) == False:
                values.append([p.id, p.z, p.z_true, p.area, p.solidity, p.snr, len(img.particles)])

    df = pd.DataFrame(data=values, columns=['id', 'z', 'true_z', 'area', 'solidity', 'snr', 'img_num_particles'])

    df['true_z'] = df['true_z'].round(decimals=3)

    df_z_count = df.groupby(['true_z']).count()
    df_z_mean = df.groupby(['true_z']).mean()
    df_z_std = df.groupby(['true_z']).std()

    fig, ax = plt.subplots()
    ax.errorbar(x=df_z_mean.index, y=df_z_mean.snr, yerr=df_z_std.snr*2, fmt='o', color='darkblue', ecolor='lightblue', elinewidth=1, capsize=2)
    ax.plot(df_z_mean.index, df_z_mean.snr, color='darkblue', alpha=0.25)

    #ax.set_xlabel(r'$\Delta z_{true} (\mu m)$')
    ax.set_xlabel(r'$z_{true}$ / $h$')
    ax.set_ylabel(r'$SNR$ ( $\frac {\mu_p}{\sigma_I}$ )', color='darkblue')
    ax.set_ylim([0, 50])

    if second_plot == 'area':
        ax2 = ax.twinx()
        ax2.errorbar(x=df_z_mean.index, y=df_z_mean.area, yerr=df_z_std.area*2, fmt='o', color='darkgreen', ecolor='limegreen', elinewidth=1, capsize=2)
        ax2.plot(df_z_mean.index, df_z_mean.area, color='darkgreen', alpha=0.125)
        ax2.set_ylabel(r'$A_{p}$ $(pixels^2)$', color='darkgreen')
        ax2.set_ylim([0, 1700])

    elif second_plot == 'solidity':
        ax2 = ax.twinx()
        ax2.errorbar(x=df_z_mean.index, y=df_z_mean.solidity, yerr=df_z_std.solidity*2, fmt='o', color='darkgreen', ecolor='limegreen', elinewidth=1, capsize=2)
        ax2.plot(df_z_mean.index, df_z_mean.solidity, color='darkgreen', alpha=0.125)
        ax2.set_ylabel(r'$Solidity$ $(\%)$', color='darkgreen')
        ax2.set_ylim([0.8, 1.0125])

    elif second_plot == 'percent_measured':
        per_measured_particles = []
        for ind in df_z_count.index:
            per_measured_particles.append([ind, df_z_count['z'][ind] / df_z_mean['img_num_particles'][ind]])
        percent_measured_particles = np.array(per_measured_particles, dtype=float)

        ax2 = ax.twinx()
        ax2.scatter(percent_measured_particles[:, 0], percent_measured_particles[:, 1], color='darkgreen')
        ax2.plot(percent_measured_particles[:, 0], percent_measured_particles[:, 1], color='darkgreen', alpha=0.25)
        ax2.set_ylabel(r'$Measured$ $Particles$ $(\%)$', color='darkgreen')
        ax2.set_ylim([0.4, 1.0125])

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