import numpy as np
from os.path import isdir, join
from os import mkdir

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

def generate_sig_input(n_images=100, background_noise=0, particle_density=0.1, range_z=(-43, 43), uniform_xy=None,
                        img_shape=(1000, 1000), particle_diameter=2, folder=None):
    assert isinstance(background_noise, int) or isinstance(background_noise, float)
    assert isinstance(img_shape, tuple)
    assert isinstance(range_z, tuple) or isinstance(range_z, list)
    assert 0 < particle_density < 1
    assert isinstance(n_images, int) and n_images > 0
    assert particle_diameter > 0
    assert folder is not None

    if isdir(folder):
        raise ValueError('Folder {} already exists. Specify a new one'.format(folder))
    else:
        mkdir(folder)

    settings_dict = {}
    settings_dict.update(DEFAULTS)
    settings_dict.update({'background_noise': background_noise})
    settings_dict.update({'pixel_dim_x': int(img_shape[0])})
    settings_dict.update({'pixel_dim_y': int(img_shape[1])})

    # Generate settings.txt
    settings = ''
    for key, val in settings_dict.items():
        settings += '{} = {}\n'.format(key, val)

    with open(join(folder, 'settings.txt'), 'w') as file:
        file.write(settings)

    d_per_px = DEFAULTS['pixel_size']
    a_total = img_shape[0] * img_shape[1] * d_per_px**2
    a_particle = (particle_diameter / 2)**2 * np.pi

    # In folder, create a subfolder for the raw txts
    txtfolder = join(folder, 'rawtxt')
    mkdir(txtfolder)

    for i in range(n_images):
        if uniform_xy is None:
            n_particles = int(a_total / a_particle)
            coords = np.hstack([np.random.randint(0, img_shape[0], size=(n_particles, 1)),
                                np.random.randint(0, img_shape[1], size=(n_particles, 1)),
                                np.random.uniform(range_z[0], range_z[1], size=(n_particles, 1))])
        else:
            n_x = uniform_xy[0]
            n_y = uniform_xy[1]
            n_particles = n_x * n_y
            edge_x = img_shape[0] / (n_x + 1)
            edge_y = img_shape[1] / (n_y + 1)

            coords = np.hstack([np.mgrid[edge_x:img_shape[0] - edge_x:np.complex(0, n_x),
                                         edge_y:img_shape[1] - edge_y:np.complex(0, n_y)].reshape(2, -1).T,
                                np.random.uniform(range_z[0], range_z[1], size=(n_particles, 1))])

        # Add the particle diameter column
        out = np.append(coords, np.array(n_particles * [particle_diameter]).reshape(-1, 1), axis=1)

        savepath = join(txtfolder, 'B{0:04d}.txt'.format(i))
        np.savetxt(savepath, out, fmt='%.6f', delimiter=' ')

if __name__ == '__main__':
    generate_sig_inpupt(50, background_noise=20, uniform_xy=(20, 20),
                        folder=r'C:\Users\silus\UCSB\master_thesis\python_stuff\gdpyt\tests\test_synthetic\testset')








