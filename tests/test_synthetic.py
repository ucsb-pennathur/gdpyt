from gdpyt import GdpytImageCollection, GdpytCalibrationSet
from gdpyt.utils import generate_sig_input, generate_sig_calibration, generate_synthetic_images
from gdpyt.utils.evaluation import GdpytPerformanceEvaluation
import matplotlib.pyplot as plt
from os.path import join
import pandas as pd
import numpy as np

# Properties of the synthetic images
n_images = 40
background_noise = 20
grid = (10, 10)
particle_diameter = 2
range_z = (-40, 40)
shape = (512, 512)
folder = r'C:\Users\silus\UCSB\master_thesis\python_stuff\gdpyt\tests\test_synthetic\testset'

setting_file = join(folder, 'settings.txt')
testtxt_folder = join(folder, 'input')
calibtxt_folder = join(folder, 'calibration_input')
testimg_folder = join(folder, 'images')
calibimg_folder = join(folder, 'calibration_images')

# Generate input files for synthetic image generator
# generate_sig_input(n_images=n_images, background_noise=background_noise, particle_density=grid, img_shape=shape,
#                    range_z=range_z, particle_diameter=particle_diameter, folder=folder)
# generate_sig_calibration(setting_file, np.linspace(-40, 40, 81))
#
# # Generate .tif
# generate_synthetic_images(setting_file, testtxt_folder, testimg_folder)
# generate_synthetic_images(setting_file, calibtxt_folder, calibimg_folder)

filetype = '.tif'
processing = {
    'cv2.medianBlur': {'args': [5]}}#,
    #'cv2.bilateralFilter': {'args': [9, 10, 10]}}

calib_collection = GdpytImageCollection(calibimg_folder, filetype, processing_specs=processing,
                                        min_particle_size=20, max_particle_size=2000)
calib_collection.uniformize_particle_ids()
name_to_z = {}
for image in calib_collection.images.values():
    name_to_z.update({image.filename: float(image.filename.split('_')[-1].split('.')[0])})

calib_set = calib_collection.create_calibration(name_to_z)

collection = GdpytImageCollection(testimg_folder, filetype, processing_specs=processing,
                                  min_particle_size=20, max_particle_size=2000)
collection.uniformize_particle_ids(baseline=calib_set)

collection.infer_z(calib_set)
collection.image_stats

perf_eval = GdpytPerformanceEvaluation(collection, testtxt_folder)
perf_eval.sigma_z()
perf_eval.sigma_z_local()