from gdpyt import GdpytImageCollection
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters.rank import median
from skimage.filters import gaussian, median
from skimage.exposure import equalize_adapthist
from scipy import ndimage as ndi
from skimage import (
    color, feature, filters, io, measure, morphology, segmentation, util
)

import numpy as np

# test only "test collection"

test_col = GdpytImageCollection(folder='/Users/mackenzie/Downloads/spt/',
                                filetype='.tif',
                                image_collection_type='test',
                                file_basestring='test_',
                                calibration_stack_z_step=1.0,
                                subset=[0, 3, 1],
                                true_num_particles=1,
                                folder_ground_truth=None,
                                stacks_use_raw=False,
                                crop_specs=None,
                                background_subtraction=None,
                                processing_specs=None,
                                thresholding_specs={'manual': [3000]},
                                min_particle_size=3,
                                max_particle_size=12,
                                shape_tol=0.5,
                                overlap_threshold=0.5,
                                same_id_threshold=8,
                                measurement_depth=10,
                                template_padding=5,
                                if_img_stack_take='mean',
                                take_subset_mean=[0, 3],
                                inspect_contours_for_every_image=False,
                                baseline='test_000.tif',
                                hard_baseline=False,
                                static_templates=False,
                                particle_id_image='test_000.tif',
                                overlapping_particles=False,
                                xydisplacement=[[0, 0]],
                                )


# old test script
"""folder = r"/Users/mackenzie/Desktop/BPE-ICEO/06.08.21 - BPE-ICEO actuator/calibration/1um_step_1imageperstep_50X/"
filetype = '.tif'
base_string = 'calib_'


cropping = {
    'xmin': 225,
    'xmax': 400,
    'ymin': 150,
    'ymax': 512
}

processing = {
    #'none': 0
    'median': {'args': [disk(2)]},
    #'gaussian': {'args': [], 'kwargs': dict(sigma=0.65, preserve_range=True)},
    'white_tophat': {'args': [disk(3)]}, # returns bright spots smaller than the structuring element.
    #'equalize_adapthist': {'args': [41], 'kwargs': dict(clip_limit=0.005)},
    }

threshold = {
    #'none': [],
    #'manual': [manual_initial_guess],
    'manual_smoothing': [2, 20, 0.5],
    #'triangle': [],
    #'otsu': [],
    #'multiotsu': {'classes': 3},
    #'local': {'block_size': 35, 'offset': 10, 'method': 'gaussian'},
    #'li': [],
    #'niblack': {'window_size': 11, 'k': 0.2},
    #'sauvola': {'window_size': 11, 'k': 0.2}
}

collection = GdpytImageCollection(folder, filetype,
                                  crop_specs=cropping,
                                  background_subtraction='min',
                                  processing_specs=processing,
                                  thresholding_specs=threshold,
                                  min_particle_size=4, shape_tol=0.4, overlap_threshold=0.8,
                                  exclude=[])
#img = collection.images[0]
img = collection.images['calib_5.tif']

print(np.mean(img.subbg))
print(np.mean(img.original))

img_clipped = img.raw.copy()
img_clipped = np.where(img_clipped<2700,img_clipped,2700)          # clip upper percentile
img_clipped = np.where(img_clipped>760,img_clipped,760)          # clip lower percentile
img_min_clipped = np.where(img_clipped - collection.background_img<0,0,img_clipped - collection.background_img)

fig, axes = plt.subplots(nrows=2,ncols=3, figsize=(12, 8))
ax = axes.ravel()
ax[0].set_title('Background Subtracted')
ax[0].imshow(img.subbg*10, cmap='viridis')
ax[1].set_title('Filtered')
ax[1].imshow(img.filtered, cmap='gray')
ax[2].set_title('Masked')
ax[2].imshow(img.masked, cmap='gray')
ax[3].set_title('Original')
linecolor=int(np.round(img.original.max(),0))
cv2.rectangle(img.original, (cropping['xmin'], cropping['ymin']), (cropping['xmax'], cropping['ymax']), linecolor, 2)
ax[3].imshow(img.original*10, cmap='viridis')
ax[5].set_title('Identified Particles')
ax[5].imshow(img.draw_particles(draw_id=False, thickness=1))
ax[4].set_title('Clipped + Background Subtraction')
ax[4].imshow(img_min_clipped, cmap='gray')

plt.tight_layout()
plt.show()

# image segmentation
dividing = img.masked
distance = ndi.distance_transform_edt(dividing)
local_maxi = feature.peak_local_max(distance, indices=False, min_distance=7)
markers = measure.label(local_maxi)
segmented_particles = segmentation.watershed(-distance, markers, mask=dividing)

# -- Overlay the images --
color_labels = color.label2rgb(segmented_particles, img.raw*50, alpha=0.2, bg_label=0)

fig, ax = plt.subplots(figsize=(8,6))
ax.imshow(color_labels)
ax.set_title('Segmentation result over raw image')
plt.show()"""