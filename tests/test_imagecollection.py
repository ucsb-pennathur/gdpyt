from gdpyt import GdptImageCollection
import matplotlib.pyplot as plt
import cv2
import numpy as np

folder = r'C:\Users\silus\UCSB\master_thesis\python_stuff\gdpyt\tests\test_data\chip2test1\50X\calib'
filetype = '.tif'
processing = {
    'cv2.medianBlur': {'args': [9]},
    'cv2.bilateralFilter': {'args': [9, 13, 15]}}

collection = GdptImageCollection(folder, filetype, processing_specs=processing,
                                 min_particle_size=500)
img = collection.images['calib_53.tif']

fig, ax = plt.subplots(nrows=3)
ax[0].imshow(img.raw, cmap='gray')
ax[1].imshow(img.filtered, cmap='gray')
ax[2].imshow(img.draw_particles())
plt.show()

