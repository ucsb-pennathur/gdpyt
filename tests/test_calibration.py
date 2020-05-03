from gdpyt import GdpytImageCollection
import matplotlib.pyplot as plt
import cv2
import numpy as np

folder = r'C:\Users\silus\UCSB\master_thesis\python_stuff\gdpyt\tests\test_data\chip2test1\50X\calib'
filetype = '.tif'
processing = {
    'cv2.medianBlur': {'args': [9]},
    'cv2.bilateralFilter': {'args': [9, 13, 15]}}

collection = GdpytImageCollection(folder, filetype, processing_specs=processing,
                                  min_particle_size=500)
collection.uniformize_particle_ids()

name_to_z = {}
for image in collection.images.values():
    name_to_z.update({image.filename: float(image.filename.split('_')[-1].split('.')[0])})

calib_set = collection.create_calibration(name_to_z)

stack = calib_set.calibration_stacks[3]
n_images = len(stack)
n_cols = min(8, n_images)
n_rows = n_images % n_cols + 1
print(n_rows, n_cols)
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_rows*2, 2 * n_cols))
for i in range(n_rows):
    for j in range(n_cols):
        n = i * n_rows + j
        if n > n_images -1:
            break
        axes[i, j].imshow(stack[n][1], cmap='gray')
        axes[i, j].set_title('z = {}'.format(stack[n][0]))

plt.show()