"""
This script evaluates the corrleation value 'c_m' across different calib stacks.

"""

import os
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data about this stack
"""
# dataset information
N_CAL = 81
MEASUREMENT_DEPTH = 81.0
MEASUREMENT_WIDTH = 412.9
TRUE_NUM_PARTICLES_PER_IMAGE = 13

# synthetic particle generator data
MAGNIFCATION = 20
NA = 0.45
FOCAL_LENGTH = 350
REF_INDEX_MEDIUM = 1
REF_INDEX_LENS = 1.5
PIXEL_SIZE = 16
PIXEL_DIM_X = 512
PIXEL_DIM_Y = 512
BKG_MEAN = 120
BKG_NOISES = 7
GAIN = 5
CYL_FOCAL_LENGTH = 0

# optics
PIXEL_TO_MICRON_SCALING = 0.8064516129  # units: microns per pixel for 20X objective measured from 25-um checkerboard
WAVELENGTH = 600e-9
LATERAL_RESOLLUTION = 16E-6
depth_of_focus = WAVELENGTH * REF_INDEX_MEDIUM / NA ** 2 + REF_INDEX_MEDIUM / (MAGNIFCATION * NA) * LATERAL_RESOLLUTION

# image pre-processing
SHAPE_TOL = 0.95  # None == take any shape; 1 == take perfectly circular shape only.
MIN_P_AREA = 6  # minimum particle size (area: units are in pixels) (recommended: 5)
MAX_P_AREA = 2500  # maximum particle size (area: units are in pixels) (recommended: 200)
SAME_ID_THRESH = 12  # maximum distance=sqrt(x**2 + y**2) for particle to have the same ID between images
MEDIAN_DISK = 5  # size of median disk filter - [1, 1.5, 2, 2.5, 3, 4,]
CALIB_PROCESSING = {'median': {'args': [disk(MEDIAN_DISK), None, 'wrap']}}
PROCESSING = {'median': {'args': [disk(MEDIAN_DISK), None, 'wrap']}}
MEDIAN_PERCENT = 0.1  # percent additional threshold value from median value
THRESHOLD = {'otsu': []}  # {'median_percent': [MEDIAN_PERCENT]} #

# similarity
ZERO_CALIB_STACKS = False
ZERO_STACKS_OFFSET = 0.5
INFER_METHODS = 'bccorr'
MIN_CM = 0.5
SUB_IMAGE_INTERPOLATION = True
"""

# goals of this script
"""
Goals:
    1. compare uncertainty evaluated from identical stack compared to "other stacks".
        Other stacks:
            1. the second best stack
            2. a single other stack (chosen that is generally the best)
            3. the average of all other stacks
"""

# file paths
fdir = '/Users/mackenzie/Desktop/BPE.Gen2/settings/optics/calib/particle3_5.61umPinkHighInt/characterize_gdpyt/testing which calib stack is best per particle and iamge'
fname = 'gdpyt_every_stackid_results.xlsx'

# read to dataframe
df = pd.read_excel(io=join(fdir, fname), index_col=0)

# Goal #1 - evaluate the uncertainty of a particle ID evaluated by the identical stack ID.

# calculate the error
df['error_z_cm'] = np.abs(df['img_id'] - df['z_cm'])



j=1