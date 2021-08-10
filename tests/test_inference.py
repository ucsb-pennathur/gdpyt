from gdpyt import GdpytImageCollection
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.morphology import disk


#folder = r'C:\Users\silus\UCSB\master_thesis\python_stuff\gdpyt\tests\test_data\chip2test1\50X\calib'
"""
filetype = '.tif'
processing = {
    'cv2.medianBlur': {'args': [9]},
    'cv2.bilateralFilter': {'args': [9, 13, 15]}}
"""
folder = r'/Users/mackenzie/Desktop/gdpyt-tests/synthetic/calibration_images'
filetype = '.tif'
processing = {
    #'median': {'args': [disk(2)]},
    'gaussian': {'args': [], 'kwargs': dict(sigma=0.5, preserve_range=True)}
}


collection = GdpytImageCollection(folder, filetype, processing_specs=processing,
                                  min_particle_size=5)
collection.uniformize_particle_ids()

name_to_z = {}
for image in collection.images.values():
    name_to_z.update({image.filename: float(image.filename.split('_')[-1].split('.')[0])})
exclude = [img_name for img_name, z in name_to_z.items() if z > 45.0]

calib_set = collection.create_calibration(name_to_z, exclude=exclude)

#test_folder = r'C:\Users\silus\UCSB\master_thesis\python_stuff\gdpyt\tests\test_data\chip2test1\50X\test\chip1_E5_P2_f1_Obj50X_FPNR5_run_4'
test_folder = r'/Users/mackenzie/Desktop/gdpyt-tests/synthetic/images'
filetype = '.tif'
test_collection = GdpytImageCollection(test_folder, filetype, processing_specs=processing,
                                  min_particle_size=5)
test_collection.uniformize_particle_ids(baseline=calib_set)
test_collection.infer_z(calib_set).znccorr(min_cm=0.9)

test_collection.image_stats
sort_imgs = lambda x: int(x.split('B00')[-1].split('.')[0])
#test_collection.plot_particle_coordinate(particle_ids=[1], sort_images=sort_imgs)
# Plot particle ID 0 to 25
fig = test_collection.plot_particle_coordinate(particle_ids=[i for i in range(25)], sort_images=sort_imgs)
fig.get_axes()[0].get_legend().remove()
fig.show()

pass