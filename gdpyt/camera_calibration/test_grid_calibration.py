# test grid calibration

# imports
from os.path import join
import numpy as np
import pandas as pd
from skimage import data, util, io
from skimage.feature import corner_harris, corner_subpix, corner_peaks

from scipy import signal

from matplotlib import pyplot as plt


# grid path
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/images/grid/'
fp = 'grid_10umLines_50umSpacing_20X_0.5Xdemag.tif'

# read image
img = io.imread(join(base_dir, fp))
img = np.mean(img, axis=0)

# skimage
coords = corner_peaks(corner_harris(img), min_distance=5, threshold_rel=0.02)

min_corner_distance = pd.DataFrame(coords, columns=['y', 'x'])
min_corner_distance = min_corner_distance.sort_values(by=['y', 'x'])
mcd_diff_y = min_corner_distance.diff().abs().sort_values('x')

min_corner_distance = min_corner_distance.sort_values(by=['x', 'y'])
mcd_diff_x = min_corner_distance.diff().abs().sort_values('y')

coords_subpix = corner_subpix(img, coords, window_size=7)
fig, ax = plt.subplots()
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
        linestyle='None', markersize=6)
ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
ax.axis((0, 310, 200, 0))
plt.show()

w = signal.windows.gaussian(7, )

j=1