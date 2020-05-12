import numpy as np

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

def generate_sig_inpupt(n_images=100, background_noise=0, particle_density=0.1, range_z=(-43, 43),
                        img_shape=(1000, 1000), particle_diameter=2):
    assert isinstance(background_noise, int) or isinstance(background_noise, float)
    assert isinstance(img_shape, tuple)
    assert isinstance(range_z, tuple) or isinstance(range_z, list)
    assert 0 < particle_density < 1
    assert isinstance(n_images, int) and n_images > 0
    assert particle_diameter > 0

    settings_dict = {}
    settings_dict.update(DEFAULTS)
    settings_dict.update({'background_noise' : background_noise})

    # Generate settings.txt
    settings = ''
    for key, val in settings_dict.items():
        settings += '{} = {}\n'.format(key, val)

    with open('si_settings.txt', 'w') as file:
        file.write(settings)

    for i in range(n_images):
        pass