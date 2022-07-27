import numpy as np
import imageio
import os


# file paths
path_read = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/synthetic_overlap_noise-level2/grid-dz/test_images_nl5'
path_write = path_read + '_nl15'

# setup
noise = 14
subset = [1, 1500]
pixel_dim_x = 1024  # number of pixels in x-direction
pixel_dim_y = 512  # number of pixels in y-direction

files = os.listdir(path_read)
nums = [float(f.split('test_')[-1].split('.tif')[0]) for f in files]

for f, n in zip(files, nums):
    if subset[0] <= n <= subset[1]:
        # read image
        I = imageio.imread(os.path.join(path_read, f))

        # generate noise
        Irand = np.random.normal(0, noise, (pixel_dim_y, pixel_dim_x))

        # add noise to image
        I = I + np.round(Irand)

        # save image
        imageio.imwrite(os.path.join(path_write, f), np.uint16(I))
        j = 1