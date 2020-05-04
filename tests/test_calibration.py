from gdpyt import GdpytImageCollection
import matplotlib.pyplot as plt
import cv2
import numpy as np

folder = r'C:\Users\silus\UCSB\master_thesis\python_stuff\gdpyt\tests\test_data\chip2test1\50X\calib'
filetype = '.tif'
processing = {
    'cv2.medianBlur': {'args': [9]},
    'cv2.bilateralFilter': {'args': [9, 10, 10]}}

collection = GdpytImageCollection(folder, filetype, processing_specs=processing,
                                  min_particle_size=500)
collection.uniformize_particle_ids()

name_to_z = {}
for image in collection.images.values():
    name_to_z.update({image.filename: float(image.filename.split('_')[-1].split('.')[0])})

calib_set = collection.create_calibration(name_to_z)

fig = calib_set.calibration_stacks[3].plot(draw_contours=True)
fig.show()

assert True
