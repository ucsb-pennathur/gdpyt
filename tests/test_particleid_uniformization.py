from gdpyt import GdpytImageCollection
import matplotlib.pyplot as plt
from skimage.morphology import disk
from scipy import ndimage as ndi
from skimage import (
    color, feature, filters, io, measure, morphology, segmentation, util
)
import cv2
import numpy as np

folder = r'/Users/mackenzie/Box Sync/2019-2020/Research/BPE/Data/Experiments/Elastic ' \
         r'Modulus/bulgeTesting_11.16.20_Elastosil/calib/calib_5umSteps_1imgPerStep_z0at50_zfat70'
filetype = '.tif'
processing = {
    #'none': {},
    'median': {'args': [disk(1.2)]},
    'gaussian': {'args': [], 'kwargs': dict(sigma=0.85, preserve_range=True)},
    'white_tophat': {'args': [disk(5)]},
        }

collection = GdpytImageCollection(folder, filetype, processing_specs=processing,
                                  min_particle_size=3)
collection.uniformize_particle_ids(threshold=60, uv=[[0,0]])

n_images = len(collection)
n_cols = 3 #5
n_rows = 3 #n_images % n_cols + 1

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_rows*6, 2 * n_cols))
for i in range(n_rows):
    for j in range(n_cols):
        n = i * n_rows + j * n_cols
        axes[i, j].imshow(collection[n].draw_particles(), cmap='gray')
        axes[i, j].set_title(collection[n].filename)

plt.show()


# Plot particle ID 0 to 19
fig = collection.plot_particle_trajectories(sort_images=None)
fig.show()
print('hey')