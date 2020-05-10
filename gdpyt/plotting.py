import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
# from gdpyt import GdpytCalibratioStack
# from gdpyt import GdpytImageCollection

def plot_calib_stack(stack, z=None, draw_contours=False):
    # assert isinstance(stack, GdpytCalibratioStack)

    if z is not None:
        if not (isinstance(z, list) or isinstance(z, tuple)):
            raise TypeError("Specify z range as a two-element list or tuple")

    n_images = len(stack)
    n_cols = min(10, n_images)
    n_rows = int(n_images / n_cols) + 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_rows * 2, 2 * n_cols))

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
    fig.subplots_adjust(wspace=0.05, hspace=0.2)

    return fig

def plot_img_collection(collection, raw=True, draw_particles=True, exclude=[], **kwargs):
    # Put this in there because this kwarg is used in GdpytImage.draw_particles
    kwargs.update({'raw': raw})

    images_to_plot = [img for name, img in collection.images.items() if name not in exclude]

    n_axes = len(images_to_plot)
    n_cols = min(10, n_axes)
    n_rows = int(n_axes / n_cols) + 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_rows * 2, 2 * n_cols))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes]).reshape(-1, 1)

    for i in range(n_rows):
        for j in range(n_cols):
            n = i * n_cols + j
            if n > len(images_to_plot) - 1:
                axes[i, j].imshow(np.zeros_like(canvas), cmap='gray')
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
                axes[i, j].imshow(canvas, cmap='gray')
                axes[i, j].set_title('{}'.format(image.filename), fontsize=5)
            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)

    if raw:
        img_type = 'raw'
    else:
        img_type = 'filtered'

    fig.suptitle('Image collection {}, Image type: {}'.format(collection.folder, img_type))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    return fig

def plot_particle_trajectories(collection, sort_images=None, create_gif=False):
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
    if not create_gif:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for id_ in coords['id'].unique():
            thisid = coords[coords['id'] == id_]
            ax.scatter(thisid['x'], thisid['y'], thisid['z'], label='ID_{}'.format(id_))
        ax.legend()

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
        for image in collection.images.values():
            coords.append(image.particle_coordinates(id_=particle_id).set_index('id')[[coordinate]])
    else:
        if not callable(sort_images):
            raise TypeError("sort_images must be a function that takes an image name as an argument and returns a value"
                            "that can be used to sort the images")
        # Get the particle coordinates from all the images
        for file in sorted(collection.files, key=sort_images):
            coords.append(collection.images[file].particle_coordinates(id_=particle_id).set_index('id')[[coordinate]])

    coords = pd.concat(coords)
    fig, ax = plt.subplots(figsize=(11, 7))
    for id_ in particle_id:
        ax.plot(coords.loc[id_].values, label='ID {}'.format(id_))
    ax.set_xlabel('Image #')
    ax.set_ylabel('{} position'.format(coordinate.upper()))
    ax.legend()

    return fig

