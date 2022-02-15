"""
This script measures image distortion and provides tools for distortion correction.
Process:
1. Read image
2. Find intersections center
3. Center and align the grid image
4. Create an image of the "true" grid
5.


Functions:
1. find_grid_corners
2. center_and_align_grid
2. create_true_grid
3. fit_distortion_model
"""

# below imports from blog
from __future__ import print_function, division
import os
import glob
import sys, argparse
import pprint
import numpy as np
import cv2
from scipy import optimize as opt

# below imports from sean
from skimage import io
from skimage.morphology import square
from skimage.filters import median, threshold_otsu, threshold_local, gaussian, median
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.exposure import equalize_adapthist, rescale_intensity
from matplotlib import pyplot as plt


np.set_printoptions(suppress=True)
puts = pprint.pprint

DATA_DIR = "/Users/mackenzie/Desktop/calibrateCamera/data_manual/"
BF_DIR = "/Users/mackenzie/Desktop/calibrateCamera/brightfield/"
DEBUG_DIR = "/Users/mackenzie/Desktop/calibrateCamera/data/debug/"
PATTERN_SIZE = (6, 5)
SQUARE_SIZE = 1.0

# read checkerboard
images = [each for each in glob.glob(DATA_DIR + "*.tif")]
img = io.imread(images[0])
img_mean = np.rint(np.mean(img, axis=0)).astype(np.uint16)
img_mean = rescale_intensity(img_mean, out_range=np.uint8)

# read brightfield
"""
images = [each for each in glob.glob(BF_DIR + "*.tif")]
img = io.imread(images[0])
bf_mean = np.rint(np.mean(img, axis=0)).astype(np.uint16)
bf_inv = np.invert(bf_mean)
bf_inv_rescale = rescale_intensity(bf_inv, out_range=float)

# correct checkerboard using brightfield
img_corr = np.rint(img_mean * bf_inv_rescale).astype(np.uint16)
"""

# make boundaries white
mask = img_mean < 5
img_corr2 = img_mean.copy()
img_corr2[mask] = 255

fig, [ax1, ax2] = plt.subplots(ncols=2)
ax1.imshow(img_mean)
ax2.imshow(img_corr2)
plt.show()

# find corners - OpenCV
xr = np.arange(3, 10)
yr = np.arange(3, 10)
for xx in xr:
    for yy in yr:
        psize = (xx, yy)
        ret, corners = cv2.findChessboardCorners(img_mean, patternSize=psize)
        if ret == True:
            print(xx, yy)
            j = 1
        else:
            print(xx, yy)



criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
if ret == True:

    corners2 = cv2.cornerSubPix(img_mean, corners, PATTERN_SIZE, (-1, -1), criteria)

    # Draw and display the corners
    cv2.drawChessboardCorners(img_mean, PATTERN_SIZE, corners2, ret)
    cv2.imshow('ha', img_mean)
    cv2.waitKey(500)

"""
def show_image(string, image):
    cv2.imshow(string, image)
    cv2.waitKey()

def get_camera_images():
    images = [each for each in glob.glob(DATA_DIR + "*.tif")]
    images = sorted(images)
    for each in images:
        img = io.imread(each)
        img = np.rint(np.mean(img, axis=0)).astype(np.uint8)
        yield (each, img)
    # yield [(each, cv2.imread(each, 0)) for each in images]

def getChessboardCorners(images = None, visualize=False):
    objp = np.zeros((PATTERN_SIZE[1]*PATTERN_SIZE[0], 3), dtype=np.float64)
    # objp[:,:2] = np.mgrid[0:PATTERN_SIZE[1], 0:PATTERN_SIZE[0]].T.reshape(-1, 2)
    objp[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    chessboard_corners = []
    image_points = []
    object_points = []
    correspondences = []
    ctr = 0
    for (path, each) in get_camera_images(): #images:
        print("Processing Image : ", path)
        ret, corners = cv2.findChessboardCorners(each, patternSize=PATTERN_SIZE)
        if ret:
            print("Chessboard Detected ")
            corners = corners.reshape(-1, 2)
            # corners = corners.astype(np.int)
            # corners = corners.astype(np.float64)
            if corners.shape[0] == objp.shape[0] :
                # print(objp[:,:-1].shape)
                image_points.append(corners)
                object_points.append(objp[:,:-1]) #append only World_X, World_Y. Because World_Z is ZERO. Just a simple modification for get_normalization_matrix
                assert corners.shape == objp[:, :-1].shape, "mismatch shape corners and objp[:,:-1]"
                correspondences.append([corners.astype(np.int), objp[:, :-1].astype(np.int)])
            if visualize:
                # Draw and display the corners
                ec = cv2.cvtColor(each, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(ec, PATTERN_SIZE, corners, ret)
                cv2.imwrite(DEBUG_DIR + str(ctr)+".png", ec)
                # show_image("mgri", ec)
                #
        else:
            print("Error in detection points", ctr)

        ctr += 1

    # sys.exit(1)
    return correspondences

chessboard_correspondences = getChessboardCorners(images=None, visualize = True)
"""
j = 1