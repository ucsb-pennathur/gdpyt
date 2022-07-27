import numpy as np
from os.path import isdir, join, dirname
from os import mkdir
from pathlib import Path


DEFAULTS = dict(
    magnification=50,
    numerical_aperture=0.5,
    focal_length=350,
    ri_medium=1,
    ri_lens=1.5,
    pixel_size=6.5,
    pixel_dim_x=752,
    pixel_dim_y=376,
    background_mean=323,
    points_per_pixel=18,
    n_rays=500,
    gain=1,
    cyl_focal_length=0
)


def generate_sig_settings(settings, folder=None):

    assert isinstance(settings, dict)
    assert folder is not None

    settings_dict = {}
    settings_dict.update(DEFAULTS)
    settings_dict.update(settings)

    if not isdir(folder):
        mkdir(folder)

    # Generate settings.txt
    settings = ''
    for key, val in settings_dict.items():
        settings += '{} = {}\n'.format(key, val)

    with open(join(folder, 'settings.txt'), 'w') as file:
        file.write(settings)

    return settings_dict, join(folder, 'settings.txt')


def generate_grid_input(settings_file, n_images, grid, range_z=(-40, 40), particle_diameter=2,
                        linear_overlap=False):
    """
    Generates input images with particles arranged in a grid

    Options:
        * linearly spaced particle overlap
    """

    settings_path = Path(settings_file)
    txtfolder = join(settings_path.parent, 'input')

    if isdir(txtfolder):
        raise ValueError('Folder {} already exists. Specify a new one'.format(txtfolder))
    else:
        mkdir(txtfolder)

    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    img_shape = (settings_dict['pixel_dim_x'], settings_dict['pixel_dim_y'])

    for i in range(1, n_images+1):
        fname = 'B{0:04d}'.format(i)

        # option for particle overlap array
        if linear_overlap:
            coordinates = _generate_scaled_overlap_paired_random_z_grid_coordinates(grid, img_shape, z=range_z,
                                                                                    overlap_scale=linear_overlap)
        else:
            coordinates = _generate_grid_coordinates(grid, img_shape, z=range_z)

        output = _append_particle_diam(coordinates, particle_diameter)
        savepath = join(txtfolder, fname + '.txt')
        np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')


def generate_grid_input_from_function(settings_file, n_images, grid, function_z=None, particle_diameter=2):
    """
    Generate particle locations on a grid from a function
    """

    assert callable(function_z)

    settings_path = Path(settings_file)
    txtfolder = join(settings_path.parent, 'input')

    if isdir(txtfolder):
        raise ValueError('Folder {} already exists. Specify a new one'.format(txtfolder))
    else:
        mkdir(txtfolder)

    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    img_shape = (settings_dict['pixel_dim_x'], settings_dict['pixel_dim_y'])

    for i in range(n_images):
        fname = 'F{0:04d}'.format(i)
        xy_coords = _generate_grid_coordinates(grid, img_shape, z=None)
        z_coords = function_z(xy_coords, i)
        coordinates = np.hstack([xy_coords.reshape(-1, 2), z_coords.reshape(-1, 1)])
        output = _append_particle_diam(coordinates, particle_diameter)
        savepath = join(txtfolder, fname + '.txt')
        np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')


def _generate_xy_coords(settings_file, particle_density=None):
    """ Generates calibration images with particles arranged in a grid OR randomly. The difference between an input
    image is that all the particles in a calibration image are at the same height """

    settings_path = Path(settings_file)
    calib_path = join(settings_path.parent, 'calibration_input')

    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    xy_coordinates = _generate_random_xy_coordinates_by_density(particle_density=particle_density,
                                                                setup_params=settings_dict)

    return xy_coordinates


def _add_z_coord(xy_coords, z):
    if isinstance(z, int) or isinstance(z, float):
        z_coords = z * np.ones((len(xy_coords), 1))
    else:
        z_coords = np.random.uniform(z[0], z[1], size=(len(xy_coords), 1))

    coords = np.hstack([xy_coords, z_coords])

    return coords


def generate_identical_calibration_and_test(settings_file, z_levels_calib, z_levels_test,
                                            particle_diameter=2, particle_density=None):
    """ Generates calibration images with particles arranged in a grid OR randomly. The difference between an input
    image is that all the particles in a calibration image are at the same height """

    # read settings dictionary
    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    # create xy coordinates
    xy_coords = _generate_xy_coords(settings_file, particle_density=particle_density)

    # create calibration collection
    settings_path = Path(settings_file)
    calib_path = join(settings_path.parent, 'calibration_input')

    if isdir(calib_path):
        raise ValueError('Folder {} already exists. Specify a new one'.format(calib_path))
    else:
        mkdir(calib_path)

    for z in z_levels_calib:
        coordinates = _add_z_coord(xy_coords=xy_coords, z=z)
        output = _append_particle_diam(coordinates, particle_diameter)

        fname = 'calib_{}'.format(z)
        savepath = join(calib_path, fname + '.txt')
        np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')

    # create test collection
    test_path = join(settings_path.parent, 'test_input')

    if isdir(test_path):
        raise ValueError('Folder {} already exists. Specify a new one'.format(test_path))
    else:
        mkdir(test_path)

    for zt in z_levels_test:
        coordinates = _add_z_coord(xy_coords=xy_coords, z=zt)
        output = _append_particle_diam(coordinates, particle_diameter)

        fname = 'calib_{}'.format(zt)
        savepath = join(test_path, fname + '.txt')
        np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')


def generate_grid_calibration(settings_file, grid, z_levels, particle_diameter=2, create_multiple=None,
                              linear_overlap=None, particle_density=None):
    """ Generates calibration images with particles arranged in a grid OR randomly. The difference between an input
    image is that all the particles in a calibration image are at the same height """

    xy_coordinates = None

    settings_path = Path(settings_file)
    calib_path = join(settings_path.parent, 'calibration_input')

    if isdir(calib_path):
        raise ValueError('Folder {} already exists. Specify a new one'.format(calib_path))
    else:
        mkdir(calib_path)

    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    img_shape = (settings_dict['pixel_dim_x'], settings_dict['pixel_dim_y'])

    if particle_density:
        xy_coordinates = _generate_random_xy_coordinates_by_density(particle_density=particle_density,
                                                                    setup_params=settings_dict)
    for z in z_levels:

        if xy_coordinates is not None:
            coordinates = _add_z_coord(xy_coords=xy_coordinates, z=z)
        else:
            if linear_overlap:
                coordinates = _generate_scaled_overlap_grid_coordinates(grid, img_shape, z=z,
                                                                        overlap_scale=linear_overlap)
            else:
                coordinates = _generate_grid_coordinates(grid, img_shape, z=z)

        output = _append_particle_diam(coordinates, particle_diameter)

        if create_multiple is None:
            fname = 'calib_{}'.format(z)
            savepath = join(calib_path, fname + '.txt')
            np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')
        else:
            assert isinstance(create_multiple, int)
            for i in range(create_multiple):
                fname = 'calib{}_{}'.format(i, z)
                savepath = join(calib_path, fname + '.txt')
                np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')


def _generate_grid_coordinates(grid, imshape, z=None):
    assert len(imshape) == 2

    # Particle grid
    xtot, ytot = imshape
    n_x, n_y = grid
    n_particles = n_x * n_y
    edge_x = xtot / (n_x + 1)
    edge_y = ytot / (n_y + 1)

    # Make particle coordinates
    xy_coords = np.mgrid[edge_x:xtot - edge_x:np.complex(0, n_x),
                edge_y:ytot - edge_y:np.complex(0, n_y)].reshape(2, -1).T

    if z is None:
        return xy_coords

    if isinstance(z, int) or isinstance(z, float):
        z_coords = z * np.ones((n_particles, 1))
    else:
        z_coords = np.random.uniform(z[0], z[1], size=(n_particles, 1))
    coords = np.hstack([xy_coords, z_coords])

    return coords


def _generate_grid_coordinates_xy_translation(grid, imshape, x_disp, y_disp, image_number, z=None, z_baseline=None):
    assert len(imshape) == 2

    # Particle grid
    xtot, ytot = imshape
    n_x, n_y = grid
    n_particles = n_x * n_y
    edge_x = xtot / (n_x + 1)
    edge_y = ytot / (n_y + 1)

    # Make particle coordinates
    xy_coords = np.mgrid[edge_x:xtot - edge_x:np.complex(0, n_x),
                edge_y:ytot - edge_y:np.complex(0, n_y)].reshape(2, -1).T

    if z_baseline is None and image_number > 1:
        xy_coords = xy_coords + [x_disp, y_disp]

    if z is None:
        return xy_coords

    if isinstance(z, int) or isinstance(z, float):
        if z != z_baseline:
            xy_coords = xy_coords + [x_disp, y_disp]
        z_coords = z * np.ones((n_particles, 1))
    else:
        z_coords = np.random.uniform(z[0], z[1], size=(n_particles, 1))
    coords = np.hstack([xy_coords, z_coords])

    return coords


def _generate_scaled_overlap_grid_coordinates(grid, imshape, z=None, overlap_scale=5):
    assert len(imshape) == 2

    # Particle grid
    xtot, ytot = imshape
    n_x, n_y = grid
    n_particles = n_x * n_y + n_y * (n_x - 2)
    edge_x = xtot / (n_x + 1)
    edge_y = ytot / (n_y + 1)

    # Make particle coordinates
    xy_coords = np.mgrid[edge_x:xtot - edge_x:np.complex(0, n_x),
                edge_y:ytot - edge_y:np.complex(0, n_y)].reshape(2, -1).T

    # make linearly shifted particle coordinates

    # create cutoff limits (exclude first and last rows to not expand the image size)
    xl = n_x
    xh = n_x * (n_y - 1)

    # copy the x-coordinates from original grid
    xyc = xy_coords[:, 0].copy()

    # calculate the step size for linearly arrayed overlapping
    grid_step_size = (xyc[11] - xyc[0]) / xyc[0] * overlap_scale

    # add x-dependent spacing to particles
    xyc = xyc[:] + (xyc[:] - xyc[0]) / xyc[0] * overlap_scale

    # stack the new x-coordinates with the original y-coordinates
    xy_overlap_coords = np.vstack((xyc, xy_coords[:, 1])).T

    # grab only the inner columns
    xy_overlap_coords = xy_overlap_coords[xl:xh]

    # stack the arrays to overlap particles
    xyo_coords = np.vstack((xy_coords, xy_overlap_coords))

    if z is None:
        return xyo_coords

    if isinstance(z, int) or isinstance(z, float):
        z_coords = z * np.ones((n_particles, 1))
    else:
        z_coords = np.random.uniform(z[0], z[1], size=(n_particles, 1))
    coords = np.hstack([xyo_coords, z_coords])

    return coords


def _generate_scaled_overlap_paired_uniform_z_grid_coordinates(grid, imshape, z=None, overlap_scale=5):
    assert len(imshape) == 2

    # Particle grid
    xtot, ytot = imshape
    n_x, n_y = grid
    n_particles = n_x * n_y
    edge_x = xtot / (n_x + 1)
    edge_y = ytot / (n_y + 1)

    # Make particle coordinates
    xy_coords = np.mgrid[edge_x:xtot - edge_x:np.complex(0, n_x),
                edge_y:ytot - edge_y:np.complex(0, n_y)].reshape(2, -1).T

    # make linearly shifted particle coordinates

    # create cutoff limits (exclude first and last rows to not expand the image size)
    xl = n_x
    xh = n_x * (n_y - 1)

    # copy the x-coordinates from original grid
    xyc = xy_coords[:, 0].copy()

    # calculate the step size for linearly arrayed overlapping
    grid_step_size = (xyc[11] - xyc[0]) / xyc[0] * overlap_scale

    # add x-dependent spacing to particles
    xyc = xyc[:] + (xyc[:] - xyc[0]) / xyc[0] * overlap_scale

    # stack the new x-coordinates with the original y-coordinates
    xy_overlap_coords = np.vstack((xyc, xy_coords[:, 1])).T

    if z is None:
        return xy_coords

    if isinstance(z, int) or isinstance(z, float):
        z_coords = z * np.ones((n_particles, 1))
    else:
        raise ValueError("z must be an integer or float.")

    # stack the original particles with a random z-height
    coords = np.hstack([xy_coords, z_coords])

    # stack the paired overlapping particles with the corresponding z-height as the original particle
    overlap_coords = np.hstack([xy_overlap_coords, z_coords])

    # grab only the inner columns of the overlap particles
    overlap_coords = overlap_coords[xl:xh]

    # stack the original particles and overlapped particles with matching z-heights together
    xyo_coords = np.vstack((coords, overlap_coords))

    return xyo_coords


def _generate_scaled_overlap_paired_random_z_grid_coordinates(grid, imshape, z=None, overlap_scale=5):

    assert len(imshape) == 2

    # Particle grid
    xtot, ytot = imshape
    n_x, n_y = grid
    n_particles = n_x * n_y
    edge_x = xtot / (n_x + 1)
    edge_y = ytot / (n_y + 1)

    # Make particle coordinates
    xy_coords = np.mgrid[edge_x:xtot - edge_x:np.complex(0, n_x),
                edge_y:ytot - edge_y:np.complex(0, n_y)].reshape(2, -1).T

    # make linearly shifted particle coordinates

    # create cutoff limits (exclude first and last rows to not expand the image size)
    xl = n_x
    xh = n_x * (n_y - 1)

    # copy the x-coordinates from original grid
    xyc = xy_coords[:, 0].copy()

    # calculate the step size for linearly arrayed overlapping
    grid_step_size = (xyc[11] - xyc[0]) / xyc[0] * overlap_scale

    # add x-dependent spacing to particles
    xyc = xyc[:] + (xyc[:] - xyc[0]) / xyc[0] * overlap_scale

    # stack the new x-coordinates with the original y-coordinates
    xy_overlap_coords = np.vstack((xyc, xy_coords[:, 1])).T

    if z is None:
        return xy_coords

    z_coords_up_to_last_column = np.random.uniform(z[0], z[1], size=(n_particles - n_x, 1))
    z_second_to_last_column = z_coords_up_to_last_column[-n_x:]
    z_coords = np.vstack((z_coords_up_to_last_column, z_second_to_last_column))

    # stack the original particles with a random z-height
    coords = np.hstack([xy_coords, z_coords])

    # stack the paired overlapping particles with the corresponding z-height as the original particle
    overlap_coords = np.hstack([xy_overlap_coords, z_coords])

    # grab only the inner columns of the overlap particles
    overlap_coords = overlap_coords[xl:xh]

    # stack the original particles and overlapped particles with matching z-heights together
    xyo_coords = np.vstack((coords, overlap_coords))

    return xyo_coords


def max_overlap_spacing(image_size, max_diameter, boundary, overlap_scaling):
    xo = max_diameter + boundary
    xp = max_diameter + boundary
    xs = [[xo, np.nan]]
    ni = 4
    while xp < image_size - max_diameter - boundary:
        ni = ni + 1
        xo = xp + (max_diameter + boundary)
        xp = xo + overlap_scaling * ni

        if xp < image_size - max_diameter - boundary:
            xs.append([xo, xp])

    return np.array(xs)


def _generate_scaled_overlap_paired_random_z_plus_noise_grid_coordinates(grid, imshape, z=None, overlap_scale=5,
                                                                         particle_diameter=2):

    assert len(imshape) == 2

    # Particle grid
    xtot, ytot = imshape

    if grid is None:
        max_diameter = 32
        boundary = 7
        xs = max_overlap_spacing(xtot, max_diameter, boundary, overlap_scaling=overlap_scale)
        xo = xs[:, 0]
        xov = xs[1:, 1]
        ys = np.arange(1, np.floor(ytot / (max_diameter + boundary + 0.0))) * (max_diameter + boundary)

        Xo, Yo = np.meshgrid(xo, ys)
        Xov, Yov = np.meshgrid(xov, ys)

        xy_coords = np.vstack([Xo.flatten(), Yo.flatten()]).T
        xy_overlap_coords = np.vstack([Xov.flatten(), Yov.flatten()]).T

        xy_coords = xy_coords[np.lexsort((xy_coords[:, 1], xy_coords[:, 0]))]
        xy_overlap_coords = xy_overlap_coords[np.lexsort((xy_overlap_coords[:, 1], xy_overlap_coords[:, 0]))]

        n_particles = len(xy_coords)
        n_particles_overlapping = len(xy_overlap_coords)

    else:
        n_x, n_y = grid
        n_particles = n_x * n_y
        edge_x = xtot / (n_x + 1) - 25
        edge_y = ytot / (n_y + 1) - 25

        # Make particle coordinates
        xy_coords = np.mgrid[edge_x:xtot - 1.25 * edge_x:np.complex(0, n_x),
                    edge_y:ytot - edge_y:np.complex(0, n_y)].reshape(2, -1).T

        # copy the x-coordinates from original grid
        xyc = xy_coords[:, 0].copy()

        # add x-dependent spacing to particles
        xyc = xyc[:] + (xyc[:]) / xyc[0] * overlap_scale + 3

        # stack the new x-coordinates with the original y-coordinates
        xy_overlap_coords = np.vstack((xyc, xy_coords[:, 1])).T


    if z is None:
        return xy_coords

    if isinstance(z, (tuple, list, np.ndarray)):
        z_coords = np.random.uniform(z[0], z[1], size=(n_particles, 1))

        # stack the original particles with a random z-height
        coords = np.hstack([xy_coords, z_coords])

        # stack the paired overlapping particles with the corresponding z-height as the original particle
        overlap_coords = np.hstack([xy_overlap_coords, z_coords[len(ys):]])

        # z-noise
        dx = coords[len(ys):, 0] - overlap_coords[:, 0]
        z_rand_angle = 30 * np.random.uniform(low=-1.0, high=1.0, size=len(dx))
        print("Max angle: {}, Min angle: {}".format(np.max(z_rand_angle), np.min(z_rand_angle)))
        z_rand_noise = np.tan(np.deg2rad(z_rand_angle))  # np.zeros_like(dx)  #
        z_noise = dx * z_rand_noise * 1.6  # where 1.6X is the pixel-to-micron scaling factor

        # add z-noise to overlap coords
        # overlap_coords[:, 2] = overlap_coords[:, 2] + z_noise

        # correct any z-coordinates that are outside the range by reflecting them across the z-range limits
        # overlap_coords[:, 2] = reflect_outside_range(overlap_coords[:, 2], z)

        # create particle diameter array
        arr_p_diameter = np.ones_like(coords[:, 0]) * particle_diameter
        arr_p_diameter = arr_p_diameter[:, np.newaxis]

        arr_p_diameter_ov = np.ones_like(z_noise) * particle_diameter
        arr_p_diameter_ov = arr_p_diameter_ov[:, np.newaxis]

    elif isinstance(z, (int, float)):
        z_coords = np.ones(shape=(n_particles, 1)) * z
        arr_p_diameter = np.ones(shape=(n_particles, 1)) * particle_diameter

        # stack the original particles with a random z-height
        coords = np.hstack([xy_coords, z_coords])

        # stack the paired overlapping particles with the corresponding z-height as the original particle
        z_coords_ov = np.ones(shape=(n_particles_overlapping, 1)) * z
        arr_p_diameter_ov = np.ones(shape=(n_particles_overlapping, 1)) * particle_diameter
        overlap_coords = np.hstack([xy_overlap_coords, z_coords_ov])
    else:
        raise ValueError('z not understood.')

    # stack particle diameter before z-noise
    overlap_coords = np.hstack((overlap_coords, arr_p_diameter_ov))
    coords = np.hstack((coords, arr_p_diameter))

    # stack the original particles and overlapped particles with matching z-heights together
    xyo_coords = np.vstack((coords, overlap_coords))

    # sort
    xyo_coords = xyo_coords[np.lexsort((xyo_coords[:, 0], xyo_coords[:, 1]))]

    return xyo_coords


def _generate_scaled_overlap_random_z_grid_coordinates(grid, imshape, z=None, overlap_scale=5):
    assert len(imshape) == 2

    # Particle grid
    xtot, ytot = imshape
    n_x, n_y = grid
    n_particles = n_x * n_y
    edge_x = xtot / (n_x + 1)
    edge_y = ytot / (n_y + 1)

    # Make particle coordinates
    xy_coords = np.mgrid[edge_x:xtot - edge_x:np.complex(0, n_x),
                edge_y:ytot - edge_y:np.complex(0, n_y)].reshape(2, -1).T

    # make linearly shifted particle coordinates

    # create cutoff limits (exclude first and last rows to not expand the image size)
    xl = n_x
    xh = n_x * (n_y - 1)

    # copy the x-coordinates from original grid
    xyc = xy_coords[:, 0].copy()

    # calculate the step size for linearly arrayed overlapping
    grid_step_size = (xyc[11] - xyc[0]) / xyc[0] * overlap_scale

    # add x-dependent spacing to particles
    xyc = xyc[:] + (xyc[:] - xyc[0]) / xyc[0] * overlap_scale

    # stack the new x-coordinates with the original y-coordinates
    xy_overlap_coords = np.vstack((xyc, xy_coords[:, 1])).T

    if z is None:
        return xy_coords

    if isinstance(z, int) or isinstance(z, float):
        z_coords = z * np.ones((n_particles, 1))
    else:
        z_coords = np.random.uniform(z[0], z[1], size=(n_particles, 1))

    # stack the original particles with a random z-height
    coords = np.hstack([xy_coords, z_coords])

    # stack the paired overlapping particles with the corresponding z-height as the original particle
    overlap_coords = np.hstack([xy_overlap_coords, z_coords])

    # grab only the inner columns of the overlap particles
    overlap_coords = overlap_coords[xl:xh]

    # stack the original particles and overlapped particles with matching z-heights together
    xyo_coords = np.vstack((coords, overlap_coords))

    return xyo_coords


def _generate_random_xy_coordinates_by_density(particle_density, setup_params):
    """
    particle_density (particle density = # of particles / area of field of view): typical values ~1e-4 - 1e-3
    setup_params (dict): particle_diameter, magnification, numerical_aperture, pixel_size, pixel_dim_x, pixel_dim_y
    z (microns): if scalar, then all particles == z, if list then particles random in range: z[0] to z[1]
    """

    number_of_particles = int(particle_density * setup_params['pixel_dim_x'] * setup_params['pixel_dim_y'] / \
                              setup_params['magnification'])

    xy_coords = np.hstack([np.random.randint(0, setup_params['pixel_dim_x'], size=(number_of_particles, 1)),
                           np.random.randint(0, setup_params['pixel_dim_y'], size=(number_of_particles, 1))])

    return xy_coords


def _generate_random_coordinates_by_density(particle_density, setup_params, z=None):
    """
    particle_density (particle density = # of particles / area of field of view): typical values ~1e-4 - 1e-3
    setup_params (dict): particle_diameter, magnification, numerical_aperture, pixel_size, pixel_dim_x, pixel_dim_y
    z (microns): if scalar, then all particles == z, if list then particles random in range: z[0] to z[1]
    """

    number_of_particles = int(particle_density * setup_params['pixel_dim_x'] * setup_params['pixel_dim_y'] / \
                              setup_params['magnification'])

    xy_coords = np.hstack([np.random.randint(0, setup_params['pixel_dim_x'], size=(number_of_particles, 1)),
                           np.random.randint(0, setup_params['pixel_dim_y'], size=(number_of_particles, 1))])

    if isinstance(z, int) or isinstance(z, float):
        z_coords = z * np.ones((number_of_particles, 1))
    else:
        z_coords = np.random.uniform(z[0], z[1], size=(number_of_particles, 1))

    coords = np.hstack([xy_coords, z_coords])

    return coords


def _generate_random_coordinates_by_number(number_of_particles, setup_params, z=None):
    """
    number_of_particles: number of particles in the image
    """

    xy_coords = np.hstack([np.random.randint(0, setup_params['pixel_dim_x'], size=(number_of_particles, 1)),
                           np.random.randint(0, setup_params['pixel_dim_y'], size=(number_of_particles, 1))])

    if isinstance(z, int) or isinstance(z, float):
        z_coords = z * np.ones((number_of_particles, 1))
    else:
        z_coords = np.random.uniform(z[0], z[1], size=(number_of_particles, 1))

    coords = np.hstack([xy_coords, z_coords])

    return coords


def _append_particle_diam(coords, particle_diameter):
    n_particles = len(coords)

    if isinstance(particle_diameter, int) or isinstance(particle_diameter, float):
        out = np.append(coords, np.array(n_particles * [particle_diameter]).reshape(-1, 1), axis=1)
    else:
        out = np.append(coords, np.random.uniform(particle_diameter[0],
                                                  particle_diameter[1], size=(n_particles, 1)), axis=1)
    return out

# ------------------------- ------------------------- ------------------------- ------------------------- -------------


def generate_uniform_z_grid(settings_file, grid, z_levels, particle_diameter=2, create_multiple=None,
                            dataset='calibration'):
    """
    1. Grid: uniform z-coordinate
        * Generate images according to z-levels where all particles are at the same z-coordinate.
    """

    settings_path = Path(settings_file)

    if dataset == 'calibration':
        calib_path = join(settings_path.parent, 'calibration_input')
    else:
        calib_path = join(settings_path.parent, 'test-input')

    if isdir(calib_path):
        raise ValueError('Folder {} already exists. Specify a new one'.format(calib_path))
    else:
        mkdir(calib_path)

    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    img_shape = (settings_dict['pixel_dim_x'], settings_dict['pixel_dim_y'])

    for z in z_levels:

        coordinates = _generate_grid_coordinates(grid, img_shape, z=z)

        output = _append_particle_diam(coordinates, particle_diameter)

        if create_multiple is None:
            fname = 'calib_{}'.format(z)
            savepath = join(calib_path, fname + '.txt')
            np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')
        else:
            assert isinstance(create_multiple, int)
            for i in range(create_multiple):
                fname = 'calib{}_{}'.format(i, z)
                savepath = join(calib_path, fname + '.txt')
                np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')

# ------------------------- ------------------------- ------------------------- ------------------------- -------------


def generate_uniform_z_grid_xy_translate(settings_file, grid, z_levels, x_disp, y_disp, z_baseline, particle_diameter=2,
                                         create_multiple=None, dataset='calibration'):
    """
    1. Grid: uniform z-coordinate
        * Generate images according to z-levels where all particles are at the same z-coordinate for a specified image
        (by z-height) and all other images' particles are translated by x_disp and y_disp.
    """

    settings_path = Path(settings_file)

    if dataset == 'calibration':
        calib_path = join(settings_path.parent, 'calibration_input')
    else:
        calib_path = join(settings_path.parent, 'test_input')

    if isdir(calib_path):
        raise ValueError('Folder {} already exists. Specify a new one'.format(calib_path))
    else:
        mkdir(calib_path)

    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    img_shape = (settings_dict['pixel_dim_x'], settings_dict['pixel_dim_y'])

    for i, z in enumerate(z_levels):

        coordinates = _generate_grid_coordinates_xy_translation(grid, img_shape, x_disp, y_disp, image_number=i, z=z,
                                                                z_baseline=z_baseline)

        output = _append_particle_diam(coordinates, particle_diameter)

        if create_multiple is None:
            if dataset == 'calibartion':
                fname = 'calib_{}'.format(z)
            elif dataset == 'test':
                fname = 'B{0:04d}'.format(i+1)
            savepath = join(calib_path, fname + '.txt')
            np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')
        else:
            assert isinstance(create_multiple, int)
            for i in range(create_multiple):
                fname = 'calib{}_{}'.format(i, z)
                savepath = join(calib_path, fname + '.txt')
                np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')

# ------------------------- ------------------------- ------------------------- ------------------------- -------------


def generate_random_z_grid(settings_file, n_images, grid, range_z=(-40, 40), particle_diameter=2):
    """
    2. Grid: random z-coordinate
        * Generate images according to z-levels where all particles are at a random z-coordinate.
    """
    settings_path = Path(settings_file)
    txtfolder = join(settings_path.parent, 'test_input')

    if isdir(txtfolder):
        raise ValueError('Folder {} already exists. Specify a new one'.format(txtfolder))
    else:
        mkdir(txtfolder)

    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    img_shape = (settings_dict['pixel_dim_x'], settings_dict['pixel_dim_y'])

    for i in range(1, n_images + 1):
        fname = 'B{0:04d}'.format(i)

        coordinates = _generate_grid_coordinates(grid, img_shape, z=range_z)

        output = _append_particle_diam(coordinates, particle_diameter)
        savepath = join(txtfolder, fname + '.txt')
        np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')


# ------------------------- ------------------------- ------------------------- ------------------------- -------------


def generate_random_z_grid_xy_translate(settings_file, n_images, grid, x_disp, y_disp, range_z=(-40, 40), particle_diameter=2):
    """
    2. Grid: random z-coordinate
        * Generate images according to z-levels where all particles are at a random z-coordinate.
    """
    settings_path = Path(settings_file)
    txtfolder = join(settings_path.parent, 'test_input')

    if isdir(txtfolder):
        raise ValueError('Folder {} already exists. Specify a new one'.format(txtfolder))
    else:
        mkdir(txtfolder)

    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    img_shape = (settings_dict['pixel_dim_x'], settings_dict['pixel_dim_y'])

    for i in range(1, n_images + 1):
        fname = 'B{0:04d}'.format(i)

        coordinates = _generate_grid_coordinates_xy_translation(grid, img_shape, x_disp, y_disp, image_number=i,
                                                                z=range_z)

        output = _append_particle_diam(coordinates, particle_diameter)
        savepath = join(txtfolder, fname + '.txt')
        np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')



# ------------------------- ------------------------- ------------------------- ------------------------- -------------


def generate_uniform_z_overlap_grid(settings_file, grid, z_levels, particle_diameter=2, linear_overlap=5):
    """
    3. Grid overlap: uniform z-coordinate
        * Generate images according to z-levels with linearly-arrayed overlapped particles at the same z-coordinate.
    """

    settings_path = Path(settings_file)
    calib_path = join(settings_path.parent, 'calibration_input')

    if isdir(calib_path):
        raise ValueError('Folder {} already exists. Specify a new one'.format(calib_path))
    else:
        mkdir(calib_path)

    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    img_shape = (settings_dict['pixel_dim_x'], settings_dict['pixel_dim_y'])

    for z in z_levels:

        coordinates = _generate_scaled_overlap_paired_uniform_z_grid_coordinates(grid, img_shape, z=z,
                                                                                overlap_scale=linear_overlap)
        output = _append_particle_diam(coordinates, particle_diameter)

        fname = 'calib_{}'.format(z)
        savepath = join(calib_path, fname + '.txt')
        np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')

# ------------------------- ------------------------- ------------------------- ------------------------- -------------


def generate_paired_random_z_overlap_grid(settings_file, n_images, grid, range_z=(-40, 40), particle_diameter=2,
                                   linear_overlap=5):
    """
    4. Grid overlap: random z-coordinate
        * Generate images according to z-levels with linearly-arrayed overlapped particles at random z-coordinates.
    """
    settings_path = Path(settings_file)
    txtfolder = join(settings_path.parent, 'input')

    if isdir(txtfolder):
        raise ValueError('Folder {} already exists. Specify a new one'.format(txtfolder))
    else:
        mkdir(txtfolder)

    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    img_shape = (settings_dict['pixel_dim_x'], settings_dict['pixel_dim_y'])

    for i in range(1, n_images + 1):
        fname = 'B{0:04d}'.format(i)

        coordinates = _generate_scaled_overlap_paired_random_z_grid_coordinates(grid, img_shape, z=range_z,
                                                                                overlap_scale=linear_overlap)
        output = _append_particle_diam(coordinates, particle_diameter)

        savepath = join(txtfolder, fname + '.txt')
        np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')


def generate_paired_random_z_plus_noise_overlap_grid_calibration(settings_file, n_images, grid, range_z=(-40, 40),
                                                     particle_diameter=2, linear_overlap=5):
    """
    4. Grid overlap: random z-coordinate
        * Generate images according to z-levels with linearly-arrayed overlapped particles at random z-coordinates.
    """
    settings_path = Path(settings_file)
    txtfolder = join(settings_path.parent, 'calibration_input')

    if isdir(txtfolder):
        pass
        # raise ValueError('Folder {} already exists. Specify a new one'.format(txtfolder))
    else:
        mkdir(txtfolder)

    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    img_shape = (settings_dict['pixel_dim_x'], settings_dict['pixel_dim_y'])

    for i in range(1, n_images + 1):
        z = range_z[i - 1]
        fname = 'calib_{}'.format(z)

        coordinates = _generate_scaled_overlap_paired_random_z_plus_noise_grid_coordinates(grid,
                                                                                           img_shape,
                                                                                           z=z,
                                                                                           overlap_scale=linear_overlap,
                                                                                           particle_diameter=particle_diameter)
        # output = _append_particle_diam(coordinates, particle_diameter)
        output = coordinates

        savepath = join(txtfolder, fname + '.txt')
        np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')


def generate_paired_random_z_plus_noise_overlap_grid(settings_file, n_images, grid, range_z=(-40, 40),
                                                     particle_diameter=2, linear_overlap=5):
    """
    4. Grid overlap: random z-coordinate
        * Generate images according to z-levels with linearly-arrayed overlapped particles at random z-coordinates.
    """
    settings_path = Path(settings_file)
    txtfolder = join(settings_path.parent, 'test_input')

    if isdir(txtfolder):
        pass
        # raise ValueError('Folder {} already exists. Specify a new one'.format(txtfolder))
    else:
        mkdir(txtfolder)

    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    img_shape = (settings_dict['pixel_dim_x'], settings_dict['pixel_dim_y'])

    for i in range(1, n_images + 1):
        i = i + 1700
        fname = 'test_{0:03d}'.format(i)

        coordinates = _generate_scaled_overlap_paired_random_z_plus_noise_grid_coordinates(grid,
                                                                                           img_shape,
                                                                                           z=range_z,
                                                                                           overlap_scale=linear_overlap,
                                                                                           particle_diameter=particle_diameter)
        # output = _append_particle_diam(coordinates, particle_diameter)
        output = coordinates

        savepath = join(txtfolder, fname + '.txt')
        np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')


def generate_random_z_overlap_grid(settings_file, n_images, grid, range_z=(-40, 40), particle_diameter=2,
                                   linear_overlap=5):
    """
    4. Grid overlap: random z-coordinate
        * Generate images according to z-levels with linearly-arrayed overlapped particles at random z-coordinates.
    """
    settings_path = Path(settings_file)
    txtfolder = join(settings_path.parent, 'input')

    if isdir(txtfolder):
        raise ValueError('Folder {} already exists. Specify a new one'.format(txtfolder))
    else:
        mkdir(txtfolder)

    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    img_shape = (settings_dict['pixel_dim_x'], settings_dict['pixel_dim_y'])

    for i in range(1, n_images + 1):
        fname = 'B{0:04d}'.format(i)

        coordinates = _generate_scaled_overlap_random_z_grid_coordinates(grid, img_shape, z=range_z,
                                                                                overlap_scale=linear_overlap)
        output = _append_particle_diam(coordinates, particle_diameter)

        savepath = join(txtfolder, fname + '.txt')
        np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')

# ------------------------- ------------------------- ------------------------- ------------------------- -------------

def generate_uniform_z_density_distribution(settings_file, z_levels, particle_density, particle_diameter=2,
                                            create_multiple=None):
    """
    5. Random distribution by density: uniform z-coordinate
        * Generate images according to z-levels with randomly distributed particles at uniform z-coordinates.
    """

    settings_path = Path(settings_file)
    calib_path = join(settings_path.parent, 'calibration_input')

    if isdir(calib_path):
        raise ValueError('Folder {} already exists. Specify a new one'.format(calib_path))
    else:
        mkdir(calib_path)

    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    xy_coordinates = _generate_random_xy_coordinates_by_density(particle_density=particle_density,
                                                                setup_params=settings_dict)
    for z in z_levels:

        coordinates = _add_z_coord(xy_coords=xy_coordinates, z=z)

        output = _append_particle_diam(coordinates, particle_diameter)

        if create_multiple is None:
            fname = 'calib_{}'.format(z)
            savepath = join(calib_path, fname + '.txt')
            np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')
        else:
            assert isinstance(create_multiple, int)
            for i in range(create_multiple):
                fname = 'calib{}_{}'.format(i, z)
                savepath = join(calib_path, fname + '.txt')
                np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')

# ------------------------- ------------------------- ------------------------- ------------------------- -------------

def generate_uniform_z_density_distribution_collection(settings_file, z_levels, zt_levels, particle_densities, particle_diameter=2,
                                            create_multiple=None):
    """
    5. Random distribution by density: uniform z-coordinate
        * Generate images according to z-levels with randomly distributed particles at uniform z-coordinates.
    """

    settings_path = Path(settings_file)

    max_particle_density = np.max(particle_densities)

    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    xy_coordinates = _generate_random_xy_coordinates_by_density(particle_density=max_particle_density,
                                                                setup_params=settings_dict)
    num_particles = len(xy_coordinates[:, 0])

    # start loop
    for pd in particle_densities:

        percent_particles = int(pd / max_particle_density * num_particles)
        xy_coordinates_pd = xy_coordinates[:percent_particles, :]

        calib_path = join(settings_path.parent, 'calibration_{}_input'.format(pd))
        test_path = join(settings_path.parent, 'test_{}_input'.format(pd))

        if isdir(calib_path):
            raise ValueError('Folder {} already exists. Specify a new one'.format(calib_path))
        else:
            mkdir(calib_path)
            mkdir(test_path)

        for z in z_levels:

            coordinates = _add_z_coord(xy_coords=xy_coordinates_pd, z=z)

            output = _append_particle_diam(coordinates, particle_diameter)

            if create_multiple is None:
                fname = 'calib_{}'.format(z)
                savepath = join(calib_path, fname + '.txt')
                np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')
            else:
                assert isinstance(create_multiple, int)
                for i in range(create_multiple):
                    fname = 'calib{}_{}'.format(i, z)
                    savepath = join(calib_path, fname + '.txt')
                    np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')

        for zt in zt_levels:

            tcoordinates = _add_z_coord(xy_coords=xy_coordinates_pd, z=zt)

            toutput = _append_particle_diam(tcoordinates, particle_diameter)

            if create_multiple is None:
                fname = 'calib_{}'.format(zt)
                savepath = join(test_path, fname + '.txt')
                np.savetxt(savepath, toutput, fmt='%.6f', delimiter=' ')
            else:
                assert isinstance(create_multiple, int)
                for i in range(create_multiple):
                    fname = 'calib{}_{}'.format(i, zt)
                    savepath = join(test_path, fname + '.txt')
                    np.savetxt(savepath, toutput, fmt='%.6f', delimiter=' ')

# ------------------------- ------------------------- ------------------------- ------------------------- -------------


def generate_random_z_density_distribution(settings_file, n_images, particle_density, range_z=(-40, 40),
                                           particle_diameter=2):
    """
    6. Random distribution by density: random z-coordinate
        * Generate images according to z-levels with randomly distributed particles at random z-coordinates.
    """

    settings_path = Path(settings_file)
    txtfolder = join(settings_path.parent, 'input')

    if isdir(txtfolder):
        raise ValueError('Folder {} already exists. Specify a new one'.format(txtfolder))
    else:
        mkdir(txtfolder)

    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    xy_coordinates = _generate_random_xy_coordinates_by_density(particle_density=particle_density,
                                                                setup_params=settings_dict)
    for i in range(1, n_images + 1):
        fname = 'B{0:04d}'.format(i)

        coordinates = _add_z_coord(xy_coords=xy_coordinates, z=range_z)

        output = _append_particle_diam(coordinates, particle_diameter)

        savepath = join(txtfolder, fname + '.txt')
        np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')


# ------------------------- ------------------ HELPER FUNCTIONS ---------------- ------------------------- -------------

def reflect_outside_range(arr, z_range):
    low_limit = z_range[0]
    up_limit = z_range[1]

    # lower limit
    idx_underlimit = np.argwhere(arr < low_limit)

    # reflect value across limit
    for i in idx_underlimit:
        arr[i] = low_limit - (arr[i] - low_limit)

    # upper limit
    idx_overlimit = np.argwhere(arr > up_limit)

    # reflect value across limit
    for i in idx_overlimit:
        arr[i] = up_limit - (arr[i] - up_limit)

    return arr


if __name__ == '__main__':
    pass