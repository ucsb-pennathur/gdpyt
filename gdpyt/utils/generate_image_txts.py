import numpy as np
from os.path import isdir, join, dirname
from os import mkdir
from pathlib import Path

DEFAULTS = dict(
    magnification = 50,
    numerical_aperture = 0.5,
    focal_length = 350,
    ri_medium = 1,
    ri_lens = 1.5,
    pixel_size = 6.5,
    pixel_dim_x = 752,
    pixel_dim_y = 376,
    background_mean = 323,
    points_per_pixel = 18,
    n_rays = 500,
    gain = 3.2,
    cyl_focal_length = 0
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

def generate_grid_input(settings_file, n_images, grid, range_z=(-43, 43), background_noise=0,
                        img_shape=(1000, 1000), particle_diameter=2, folder=None):
    """ Generates input images with particles arranged in a grid"""
    settings_path = Path(settings_file)
    txtfolder = join(settings_path.parent, 'input')

    if isdir(txtfolder):
        raise ValueError('Folder {} already exists. Specify a new one'.format(txt))
    else:
        mkdir(txtfolder)

    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    img_shape = (settings_dict['pixel_dim_x'], settings_dict['pixel_dim_y'])

    for i in range(n_images):
        fname = 'B{0:04d}'.format(i)
        coordinates = _generate_grid_coordinates(grid, img_shape, z=range_z)
        output = _append_particle_diam(coordinates, particle_diameter)
        savepath = join(txtfolder, fname + '.txt')
        np.savetxt(savepath, output, fmt='%.6f', delimiter=' ')

def generate_grid_input_from_function(settings_file, n_images, grid, function_z=None, particle_diameter=2, folder=None):
    assert callable(function_z)

    settings_path = Path(settings_file)
    txtfolder = join(settings_path.parent, 'input')

    if isdir(txtfolder):
        raise ValueError('Folder {} already exists. Specify a new one'.format(txt))
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

def generate_grid_calibration(settings_file, grid, z_levels, particle_diameter=2, create_multiple=None):
    """ Generates calibration images with particles arranged in a grid. The difference between an input image is that
    all the particles in a calibration image are at the same height """

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

    # Particle gird
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

def _generate_random_coordinates(density, imshape, particle_d, dpm, z=None):
    m_per_px = 1 / dpm
    a_total = imshape[0] * imshape[1] * m_per_px** 2
    a_particle = (particle_d / 2) ** 2 * np.pi

    n_particles = int(a_total * density / a_particle)
    xy_coords = np.hstack([np.random.randint(0, imshape[0], size=(n_particles, 1)),
                           np.random.randint(0, imshape[1], size=(n_particles, 1))])
    if isinstance(z, int) or isinstance(z, float):
        z_coords = z * np.ones((n_particles, 1))
    else:
        z_coords = np.random.uniform(z[0], z[1], size=(n_particles, 1))

    coords = np.hstack([xy_coords, z_coords])

    return coords

def _append_particle_diam(coords, particle_diameter):
    n_particles = len(coords)

    if isinstance(particle_diameter, int) or isinstance(particle_diameter, float):
        out = np.append(coords, np.array(n_particles * [particle_diameter]).reshape(-1, 1), axis=1)
    else:
        out = np.append(coords, np.random.randint(particle_diameter[0],
                                                  particle_diameter[1], size=(n_particles, 1)), axis=1)
    return out

if __name__ == '__main__':
   pass








