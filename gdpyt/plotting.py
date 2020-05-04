import matplotlib.pyplot as plt
import numpy as np
# from gdpyt import GdpytCalibratioStack
# from gdpyt import GdpytImageCollection

def plot_calib_stack(stack, z=None, draw_contours=False):
    # assert isinstance(stack, GdpytCalibratioStack)

    if z is not None:
        if not (isinstance(z, list) or isinstance(z, tuple)):
            raise TypeError("Specify z range as a two-element list or tuple")

    n_images = len(stack)
    n_cols = min(10, n_images)
    n_rows = n_images % n_cols + 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_rows * 2, 2 * n_cols))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes]).reshape(-1,1)

    for i in range(n_rows):
        for j in range(n_cols):
            n = i * n_rows + j
            if n > n_images - 1:
                break
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
    n_rows = n_axes % n_cols + 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_rows * 2, 2 * n_cols))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes]).reshape(-1, 1)

    for i in range(n_rows):
        for j in range(n_cols):
            n = i * n_rows + j
            if n > len(images_to_plot) - 1:
                break
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