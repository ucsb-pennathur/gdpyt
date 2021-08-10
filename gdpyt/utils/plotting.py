import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib as mpl
import numpy as np
import pandas as pd
from os.path import splitext
import re
import logging

logger = logging.getLogger(__name__)

def plot_calib_stack(stack, z=None, draw_contours=True, imgs_per_row=5, fig=None, axes=None):
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
                axes[i, j].imshow(np.zeros_like(template), cmap='gray')
                axes[i, j].set_title('None', fontsize=12)
            else:
                z, template = stack[n]
                axes[i, j].imshow(template, cmap='gray')

                for p in stack.particles:
                    if p.z == z:
                        jj = p.template_contour
                        jjj = jj[0, :]
                        jj = np.vstack([jj, jj[0, :]])
                        for contour in p.contour:
                            #axes[i, j].plot(jj[:, 0], jj[:, 1], linewidth=1, color='red')
                            axes[i, j].plot(jj[:, 1], jj[:, 0], linewidth=0.5, color='blue')

                """for particle in self.particles:
                    cv2.drawContours(canvas, [particle.contour], -1, color, thickness)"""
                j=1

                axes[i, j].set_title('z = {}'.format(z), fontsize=12)
            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)

    fig.suptitle('Calibration stack (Particle ID {})'.format(stack.id))
    fig.subplots_adjust(wspace=0.05, hspace=0.25)

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

def plot_particle_coordinate_calibration(collection, plot_type='errorbars', measurement_volume=None):

    coords = []
    identified_particles = 0
    for img in collection.images.values():
        # sum all particles that were identified in all images in the collection
        identified_particles += len(img.particles)

        # only append coordinates with a valid z-coordinate
        [coords.append([p.id, p.location[0], p.location[1], p.z, p.x_true, p.y_true, p.z_true]) for p in
         img.particles if p.z is not None and np.isnan(p.z) == False]

    df = pd.DataFrame(data=coords, columns=['id', 'x', 'y', 'z', 'true_x', 'true_y', 'true_z'])

    # number of particles with a valid z-coordinate measurement
    measured_z_particles = len(df.z)
    percent_measured_particles = measured_z_particles / identified_particles * 100

    # distance from true x-y position
    df['dist'] = np.sqrt((df['true_x'] - df['x']) ** 2 + (df['true_y'] - df['y']) ** 2)

    # measurement error
    df['square_error_x'] = (df['true_x'] - df['x']) ** 2
    df['square_error_y'] = (df['true_y'] - df['y']) ** 2
    df['square_error_z'] = (df['true_z'] - df['z']) ** 2

    # local z-uncertainty: root mean square error
    df_z_sum = df.groupby(['true_z']).sum()
    df_z_sum['rmse_x'] = np.sqrt(df_z_sum.square_error_x / measured_z_particles)
    df_z_sum['rmse_y'] = np.sqrt(df_z_sum.square_error_y / measured_z_particles)
    df_z_sum['rmse_z'] = np.sqrt(df_z_sum.square_error_z / measured_z_particles)
    df_z_sum['rmse_xy'] = np.sqrt(df_z_sum.rmse_x ** 2 + df_z_sum.rmse_y ** 2)

    if measurement_volume is not None:
        df_z_sum.rmse_xy = df_z_sum.rmse_xy / measurement_volume

    # mean z-coordinate for each true_z
    df_z_mean = df.groupby(['true_z']).mean()
    df_z_std = df.groupby(['true_z']).std()

    fig, ax = plt.subplots()
    ax.plot(df_z_sum.index, df_z_sum.index, color='black', linewidth=1, alpha=0.95, label='Ideal')
    if plot_type == 'scatter':
       ax.scatter(x=df.true_z, y=df.z)
    if plot_type == 'errorbars':
        ax.errorbar(x=df_z_sum.index, y=df_z_mean.z, yerr=df_z_sum.rmse_z, xerr=df_z_sum.rmse_xy, fmt='o', ecolor='gray', elinewidth=2, capsize=2, label='Measured')

    #ax.set_xlabel(r'$\Delta z_{true} (\mu m)$')
    ax.set_xlabel(r'$z_{true}$ / h')
    #ax.set_ylabel(r'$z_{measured}$ $(\mu m)$')
    ax.set_ylabel(r'$z_{measured}$ / h')
    ax.set_title('Measurement uncertainty: root mean squared error', fontsize=8)
    ax.legend()
    plt.tight_layout

    return fig

def plot_particle_snr_and(collection, second_plot='area'):
    values = []
    identified_particles_per_image = []
    for img in collection.images.values():
        for p in img.particles:
            if p.z is not None and np.isnan(p.z) == False:
                values.append([p.id, p.z, p.z_true, p.area, p.solidity, p.snr, len(img.particles)])

    df = pd.DataFrame(data=values, columns=['id', 'z', 'true_z', 'area', 'solidity', 'snr', 'img_num_particles'])

    df_z_count = df.groupby(['true_z']).count()
    df_z_mean = df.groupby(['true_z']).mean()
    df_z_std = df.groupby(['true_z']).std()

    fig, ax = plt.subplots()
    ax.errorbar(x=df_z_mean.index, y=df_z_mean.snr, yerr=df_z_std.snr*2, fmt='o', color='darkblue', ecolor='aliceblue', elinewidth=2, capsize=2)
    ax.plot(df_z_mean.index, df_z_mean.snr, color='darkblue', alpha=0.25)

    #ax.set_xlabel(r'$\Delta z_{true} (\mu m)$')
    ax.set_xlabel(r'$z_{true}$ / $h$')
    ax.set_ylabel(r'$SNR$ $(\frac {\mu_p}{\sigma_I})$', color='darkblue')
    ax.set_ylim([0, 50])

    if second_plot == 'area':
        ax2 = ax.twinx()
        ax2.errorbar(x=df_z_mean.index, y=df_z_mean.area, yerr=df_z_std.area*2, fmt='o', color='darkgreen', ecolor='limegreen', elinewidth=1, capsize=3)
        ax2.plot(df_z_mean.index, df_z_mean.area, color='darkgreen', alpha=0.25)
        ax2.set_ylabel(r'$A_{p}$ $(pixels^2)$', color='darkgreen')
        ax2.set_ylim([0, 1500])

    elif second_plot == 'solidity':
        ax2 = ax.twinx()
        ax2.errorbar(x=df_z_mean.index, y=df_z_mean.solidity, yerr=df_z_std.solidity*2, fmt='o', color='darkgreen', ecolor='limegreen', elinewidth=1, capsize=3)
        ax2.plot(df_z_mean.index, df_z_mean.solidity, color='darkgreen', alpha=0.25)
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
        ax2.set_ylim([0.8, 1.0125])

    plt.tight_layout

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