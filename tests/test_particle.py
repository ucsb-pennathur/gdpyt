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
n_particles = len(img.particles.keys())
fig, ax = plt.subplots(nrows=n_particles)

for id, a in zip(img.particles.keys(), ax):
    a.imshow(img.particles[id].template)
    a.set_title('Particle ID: {}'.format(id))

plt.show()