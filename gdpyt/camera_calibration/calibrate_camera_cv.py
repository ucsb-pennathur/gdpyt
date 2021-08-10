import numpy as np
import cv2 as cv
import glob
import re
import os
from skimage import io
from skimage.exposure import rescale_intensity

import matplotlib.pyplot as plt

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((14 * 14, 3), np.float32)
objp[:, :2] = np.mgrid[0:14, 0:14].T.reshape(-1, 2)
objp = objp * 25

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


# program to find all files in a folder
def find_images_in_dir(folderpath):
    files = []

    for file in os.listdir(folderpath):
        if file.endswith(".tif"):
            files.append(file)

    print("{} files in ...{}".format(len(files), file))

    return files


folderpath = '/Users/mackenzie/Desktop/gdpyt-characterization/checkerboard/07.15.21 - checkerboard/grid_25umSquares/20X/'

# ----- find all files and sort according to index -----
files = find_images_in_dir(folderpath)
base = 'grid_'
filetype = '.tif'

file_indices = []
for file in files:
    file_index = re.search(base + '(.*)' + filetype, file)
    file_indices.append(file_index.group(1))

files_and_indices = np.stack((file_indices, files), axis=1)
print(np.shape(files_and_indices))

sorted_files_indices = sorted(files_and_indices, key=lambda x: int(x[0]))

files = np.array(sorted_files_indices)[:, 1]
indices = np.array(sorted_files_indices)[:, 0]

grays = []
for file in files:
    fname = folderpath + file
    img_stack = io.imread(fname)
    img = np.mean(img_stack, axis=0)
    img = rescale_intensity(img, in_range=img, out_range=np.uint8)

    #img = cv.imread(fname)
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grays.append(img)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(img, (14, 14), None)
    print(ret)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print("object points: {}".format(len(objp)))
        print("image points: {}".format(len(corners)))
        j=1

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (14, 14), corners2, ret)
        cv.imshow(file, img)
        cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

ii = 0
for gray in grays:

    h, w = gray.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w, h))

    # undistort
    dst = cv.undistort(gray, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.imwrite('/Users/mackenzie/Desktop/gdpyt-characterization/checkerboard/07.15.21 - checkerboard/grid_25umSquares/write/calib' + str(ii) +'result.png', dst)
    ii += 1