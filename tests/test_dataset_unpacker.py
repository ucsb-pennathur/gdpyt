"""
This program tests the GDPyT measurement accuracy on... DataSet I
"""

from gdpyt import GdpytImageCollection, GdpytSetup, GdpytCharacterize
from gdpyt.utils.datasets import dataset_unpacker
from os.path import join
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.morphology import disk, square

# ----- ----- ----- ----- TEST DATASET UNPACKER ----- ----- ----- ----- ----- ----- -----
test_dataset = 'JP-EXF01-20'

if test_dataset == 'JP-EXF01-20':
    calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=2,
                                      number_of_images=50).unpack()
    test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test', noise_level=1,
                                     number_of_images=50).unpack()
elif test_dataset == 'synthetic_overlap_noise-level':
    calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=1).unpack()
    test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test', noise_level=1).unpack()
elif test_dataset == '20X_1Xmag_5.61umPink_HighInt':
    calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration').unpack()
    test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test').unpack()
elif test_dataset == '10X_0.5Xmag_2.15umNR_HighInt_0.03XHg':
    calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration').unpack()
    test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test').unpack()
elif test_dataset == '10X_1Xmag_5.1umNR_HighInt_0.06XHg':
    calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration').unpack()
    test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test').unpack()
elif test_dataset == '10X_1Xmag_2.15umNR_HighInt_0.12XHg':
    calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration').unpack()
    test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test').unpack()


# ----- ----- ----- ----- TEST ----- ----- ----- ----- ----- ----- -----
simple_test = False
if simple_test:
    simple_z = GdpytSetup.z_assessment(infer_method='sknccorr', min_cm=0.5, sub_image_interpolation=True)
    simple_test_settings = GdpytSetup.GdpytSetup(inputs=None, outputs=None, processing=None, z_assessment=simple_z, optics=None)
    calib_col, calib_set, calib_col_image_stats, calib_stack_data = GdpytCharacterize.test(calib_settings,
                                                                                           simple_test_settings,
                                                                                           return_variables='calibration_plots')
else:
    calib_col, calib_set, calib_col_image_stats, calib_stack_data, test_col, test_col_stats, \
    test_col_local_meas_quality, test_col_global_meas_quality = GdpytCharacterize.test(calib_settings,
                                                                                           test_settings,
                                                                                           return_variables=None)