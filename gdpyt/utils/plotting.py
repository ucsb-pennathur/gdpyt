import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from os.path import splitext
import logging

logger = logging.getLogger(__name__)

def plot_calib_stack(stack, z=None, draw_contours=False):
    # assert isinstance(stack, GdpytCalibratioStack)

    if z is not None:
        if not (isinstance(z, list) or isinstance(z, tuple)):
            raise TypeError("Specify z range as a two-element list or tuple")

    n_images = len(stack)
    n_cols = min(10, n_images)
    n_rows = int(n_images / n_cols) + 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(2 * n_cols, n_rows * 2))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes]).reshape(-1,1)

    for i in range(n_rows):
        for j in range(n_cols):
            n = i * n_cols + j
            if n > n_images - 1:
                axes[i, j].imshow(np.zeros_like(template), cmap='gray')
                axes[i, j].set_title('None', fontsize=5)
            else:
                z, template = stack[n]
                axes[i, j].imshow(template, cmap='gray')
                axes[i, j].set_title('z = {}'.format(z), fontsize=5)
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
        ax.plot(coords.loc[id_].values, label='ID {}'.format(id_))
    ax.set_xlabel('Image #')
    ax.set_ylabel('{} position'.format(coordinate.upper()))
    ax.legend()

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
