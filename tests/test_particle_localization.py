"""
Test particle localization methods
"""
import os

import pandas as pd
from matplotlib import pyplot as plt
from skimage.io import imread
import scipy.optimize as opt
import numpy as np

# fit 2D Gaussian
"""
# imports
from gdpyt.subpixel_localization.gaussian import gaussian2D, _gaussian2D, fit, fit_results
from gdpyt.subpixel_localization.gaussian import *

# read image to disk
fp = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Calibration/' \
     'Calibration-noise-level2/Calib-0050/B00019.tif'
img = imread(fname=fp, as_gray=True)

# guess Gaussian parameters
guess_prms = [(29, 29, 0.5, 0.5, 1000)]

# fit
popt, pcov, X, Y = fit(image=img, guess_params=guess_prms, fx=1, fy=1, print_results=True)

# results
fitted, rms = fit_results(img, X, Y, popt)

# plot
show_plots = True
if show_plots:
     fig0 = plot_2D_image(img)
     fig1 = plot_3D_image(X, Y, img)
     fig2 = plot_3D_fit(X, Y, img, fitted)
     fig3 = plot_2D_image_contours(img, fitted, X, Y)
     plt.show()

fdir = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Calibration/Calibration-noise-level2/Calib-0050/'
#fdir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/particle4_2.1umNR_HighInt/calib_fc2/'
file_list = os.listdir(fdir)
files = []
[files.append(f) for f in file_list if f.endswith('.tif')]

guess_prms = [(29, 29, 50, 50, 500)]
img_index = []
fitted_params = []
residuals = []

for f in files:
     i = f[-6:-4]
     img = imread(fname=os.path.join(fdir, f), as_gray=True)
     popt, pcov, X, Y = fit(image=img, guess_params=guess_prms, fx=1, fy=1, print_results=True)
     fitted, rms = fit_results(img, X, Y, popt)

     # store data
     img_index.append(i)
     fitted_params.append(popt)
     residuals.append(rms)

fitted_params = np.array(fitted_params)
temp = np.stack((img_index, residuals), axis=-1)
calib_fit_data = np.hstack([temp, fitted_params])

df = pd.DataFrame(data=calib_fit_data, columns=['i', 'rms', 'x', 'y', 'xa', 'ya', 'A'])
df = df.astype(dtype=float)
df['i'] = df['i'] / df['i'].max()
df = df.set_index('i')
df = df.sort_index()

fig, ax = plt.subplots()
ax.plot(df.index, df.xa)
ax.plot(df.index, df.ya)

ax2 = ax.twinx()
ax2.plot(df.index, df.A, color='gray', alpha=0.5)

plt.show()

j=1

# trackpy's centroid-based iterative refinement (working perfectly)
"""
# imports
from gdpyt.subpixel_localization.centroid_based_iterative import refine_coords_via_centroid, bandpass, grey_dilation, validate_tuple

# read image to disk
fp = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Calibration/' \
     'Calibration-noise-level2/Calib-0050/B00013.tif'
img = imread(fname=fp, as_gray=True)

# particle localization
raw_image = img
shape = raw_image.shape
ndim = len(shape)
diameter = 31
diameter = validate_tuple(diameter, ndim)
diameter = tuple([int(x) for x in diameter])
radius = tuple([x//2 for x in diameter])
separation = tuple([x + 1 for x in diameter])
smoothing_size = diameter # size of sides of the square kernel in boxcar (rolling average) smoothing
noise_size = 1 # width of gaussian blurring kernel
noise_size = validate_tuple(noise_size, ndim)
threshold = 1 # clip bandpass result below this value. Thresholding is done on background subtracted image.
percentile = 98 # features must have peak brighter than pixels in this percentile.
max_iterations = 10



image = bandpass(image=img, lshort=noise_size, llong=smoothing_size, threshold=threshold, truncate=4)
margin = tuple([max(rad, sep // 2 - 1, sm // 2) for (rad, sep, sm) in zip(radius, separation, smoothing_size)])
coords = grey_dilation(image, separation, percentile, margin, precise=False)

refined_coords = refine_coords_via_centroid(raw_image=raw_image, image=image, radius=radius, coords=coords,
                                            max_iterations=max_iterations, show_plot=True)
j=1


# radial variance transform method
"""
from gdpyt.subpixel_localization.particle_localization_sandbox import rvt, twoD_Gaussian

# read image to disk
fp = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/JP-EXF01-20/Calibration/' \
     'Calibration-noise-level2/Calib-0050/B00013.tif'
img = imread(fname=fp, as_gray=True)

# radial variance transform
rmin = 6
rmax = 17
img_rvt = rvt(img=img, rmin=rmin, rmax=rmax)

# fitting a 2D gaussian
# Create x and y indices
jjj=np.shape(img_rvt)[0]
x = np.linspace(0, np.shape(img_rvt)[0]-1, np.shape(img_rvt)[0])
y = np.linspace(0, np.shape(img_rvt)[1]-1, np.shape(img_rvt)[1])
xdata_tuple = np.meshgrid(x, y)
# amplitude, xo, yo, sigma_x, sigma_y, theta, offset
initial_guess = (1, 28, 28, 1, 1, 0, 0)
sqz = img_rvt.flatten()
popt, pcov = opt.curve_fit(twoD_Gaussian, xdata=xdata_tuple, ydata=sqz, p0=initial_guess)


data_fitted = twoD_Gaussian(xdata_tuple, *popt)
df = np.array(data_fitted.reshape(59, 59))

fig, ax = plt.subplots(1, 1)
ax.imshow(img_rvt, cmap=plt.cm.jet, extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(df, 8, colors='black')
plt.show()
"""

# lmfit method
"""
import lmfit
from lmfit.lineshapes import gaussian2d

npoints = 10000
x = np.random.rand(npoints)*10 - 4
y = np.random.rand(npoints)*5 - 3
z = gaussian2d(x, y, amplitude=30, centerx=2, centery=-.5, sigmax=.6, sigmay=.8)
z += 2*(np.random.rand(*z.shape)-.5)
error = np.sqrt(z+1)


x = np.linspace(0, np.shape(img_rvt)[0]-1, np.shape(img_rvt)[0])
y = np.linspace(0, np.shape(img_rvt)[1]-1, np.shape(img_rvt)[1])
x, y = np.meshgrid(x, y)
x = x.flatten()
y = y.flatten()
error = np.sqrt(sqz+1)

model = lmfit.models.Gaussian2dModel()
params = model.guess(sqz, x, y)
result = model.fit(sqz, x=x, y=y, params=params, weights=1/error)
lmfit.report_fit(result)

"""


j=1