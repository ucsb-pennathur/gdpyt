from gdpyt import GdpytImageCollection
import matplotlib.pyplot as plt
import cv2
import numpy as np

folder = r'C:\Users\silus\UCSB\master_thesis\python_stuff\gdpyt\tests\test_synthetic\DS_Grid_Gaussian_N50_Sigma20\calibration_images'
filetype = '.tif'
processing = {
    'cv2.GaussianBlur': {'args': [(11, 11), 5]}}
   # 'cv2.medianBlur': {'args': [9]},
   # 'cv2.bilateralFilter': {'args': [9, 13, 15]}}

collection = GdpytImageCollection(folder, filetype, processing_specs=processing,
                                  min_particle_size=20)
collection.uniformize_particle_ids(threshold=20)

name_to_z = {}
for image in collection.images.values():
    name_to_z.update({image.filename: float(image.filename.split('_')[-1].split('.')[0])})

calib_set = collection.create_calibration(name_to_z)

fig = calib_set.calibration_stacks[3].plot(draw_contours=True)
fig.show()

assert True
