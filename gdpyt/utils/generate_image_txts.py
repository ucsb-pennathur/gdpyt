import numpy as np
from os.path import isdir, join
from os import mkdir
from pathlib import Path

DEFAULTS = dict(
    magnification = 10,
    numerical_aperture = 0.3,
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

def generate_sig_settings(background_noise=0, img_shape=(1000, 1000), particle_density=0.1, particle_diameter=2,
                          folder=None):
    assert isinstance(background_noise, int) or isinstance(background_noise, float)
    assert isinstance(img_shape, tuple)
    assert isinstance(particle_density, float) or isinstance(particle_density, tuple)
    assert isinstance(particle_diameter, int) or isinstance(particle_diameter, tuple)
    assert folder is not None

    if isinstance(particle_density, float):
        assert 0 < particle_density < 1
    else:
        assert len(particle_density) == 2
    if isinstance(particle_diameter, int):
        assert particle_diameter > 0
    else:
        assert len(particle_diameter) == 2

    settings_dict = {}
    settings_dict.update(DEFAULTS)
    settings_dict.update({'particle_density': particle_density})
    settings_dict.update({'particle_diameter': particle_diameter})
    settings_dict.update({'background_noise': background_noise})
    settings_dict.update({'pixel_dim_x': int(img_shape[0])})
    settings_dict.update({'pixel_dim_y': int(img_shape[1])})

    # Generate settings.txt
    settings = ''
    for key, val in settings_dict.items():
        settings += '{} = {}\n'.format(key, val)

    with open(join(folder, 'settings.txt'), 'w') as file:
        file.write(settings)

    return settings_dict

def generate_sig_image_source(fname, settings=None, range_z=None, folder=None):
    assert folder is not None

    d_per_px = settings['pixel_size']
    a_total = settings['pixel_dim_x'] * settings['pixel_dim_y'] * d_per_px ** 2
    a_particle = (settings['particle_diameter'] / 2) ** 2 * np.pi

    if isinstance(settings['particle_density'], float):
        n_particles = int(a_total * settings['particle_density'] / a_particle)
        xy_coords = np.hstack([np.random.randint(0, settings['pixel_dim_x'], size=(n_particles, 1)),
                            np.random.randint(0, settings['pixel_dim_y'], size=(n_particles, 1))])
    else:
        n_x = settings['particle_density'][0]
        n_y = settings['particle_density'][1]
        n_particles = n_x * n_y
        edge_x = settings['pixel_dim_x']/ (n_x + 1)
        edge_y = settings['pixel_dim_y'] / (n_y + 1)

        xy_coords = np.mgrid[edge_x:settings['pixel_dim_x'] - edge_x:np.complex(0, n_x),
                            edge_y:settings['pixel_dim_y'] - edge_y:np.complex(0, n_y)].reshape(2, -1).T

    if isinstance(range_z, int) or isinstance(range_z, float):
        z_coords = range_z * np.ones((n_particles, 1))
    else:
        z_coords = np.random.uniform(range_z[0], range_z[1], size=(n_particles, 1))

    coords = np.hstack([xy_coords, z_coords])
    # Add the particle diameter column
    if isinstance(settings['particle_diameter'], int):
        out = np.append(coords, np.array(n_particles * [settings['particle_diameter']]).reshape(-1, 1), axis=1)
    else:
        out = np.append(coords,  np.random.randint(settings['particle_diameter'][0],
                                                   settings['particle_diameter'][1], size=(n_particles, 1)),
                        axis=1)
    savepath = join(folder, fname + '.txt')
    np.savetxt(savepath, out, fmt='%.6f', delimiter=' ')


def generate_sig_input(n_images=100, background_noise=0, particle_density=0.1, range_z=(-43, 43),
                        img_shape=(1000, 1000), particle_diameter=2, folder=None):

    if isdir(folder):
        raise ValueError('Folder {} already exists. Specify a new one'.format(folder))
    else:
        mkdir(folder)

    settings_dict = generate_sig_settings(background_noise=background_noise, particle_density=particle_density,
                                          particle_diameter=particle_diameter, img_shape=img_shape, folder=folder)

    # In folder, create a subfolder for the raw txts
    txtfolder = join(folder, 'input')
    mkdir(txtfolder)

    for i in range(n_images):
        fname = 'B{0:04d}'.format(i)
        generate_sig_image_source(fname, settings=settings_dict, range_z=range_z, folder=txtfolder)


def generate_sig_calibration(settings_file, z_levels, folder=None):
    if folder is None:
        settings_path = Path(settings_file)
        calib_path = join(settings_path.parent, 'calibration_input')
    else:
        calib_path = folder

    if isdir(calib_path):
        raise ValueError('Folder {} already exists. Specify a new one'.format(calib_path))
    else:
        mkdir(calib_path)

    settings_dict = {}
    with open(settings_file) as file:
        for line in file:
            thisline = line.split('=')
            settings_dict.update({thisline[0].strip(): eval(thisline[1].strip())})

    for z in z_levels:
        generate_sig_image_source('calib_{}'.format(z), settings=settings_dict, range_z=z, folder=calib_path)

if __name__ == '__main__':
   pass








