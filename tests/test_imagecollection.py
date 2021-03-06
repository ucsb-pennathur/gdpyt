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
import cv2
import numpy as np

folder = r"/Users/mackenzie/Box Sync/2019-2020/Research/BPE/Data/Experiments/Elastic " \
         r"Modulus/bulgeTesting_11.16.20_Elastosil/calib/calib_min/"
filetype = '.tif'


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
                                  min_particle_size=4, shape_tol=0.4, overlap_threshold=0.8)
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
plt.show()
