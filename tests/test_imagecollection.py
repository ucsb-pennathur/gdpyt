from gdpyt import GdpytImageCollection
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters.rank import median
from skimage.filters import gaussian, median
from scipy import ndimage as ndi
from skimage import (
    color, feature, filters, io, measure, morphology, segmentation, util
)
import cv2
import numpy as np

folder = r'/Users/mackenzie/Desktop/test_img'
filetype = '.tif'

cropping = {
    'xmin': 200,
    'xmax': 386,
    'ymin': 212,
    'ymax': 512
}

processing = {
    #'none': 0
    'median': {'args': [disk(1.25)]},
    'gaussian': {'args': [], 'kwargs': dict(sigma=0.65, preserve_range=True)},
    'white_tophat': {'args': [disk(3)]},
    #'equalize_adapthist': {'args': [], 'kwargs': dict(clip_limit=0.03)},
    }

threshold = {
    #'none': [],
    #'manual': [manual_initial_guess],
    'manual_smoothing': [2, 30],
    #'triangle': [],
    #'otsu': [],
    #'multiotsu': {'classes': 3},
    #'local': {'block_size': 13, 'offset': 15, 'method': 'gaussian'},
    #'li': [],
    #'niblack': {'window_size': 11, 'k': 0.2},
    #'sauvola': {'window_size': 11, 'k': 0.2}
}

collection = GdpytImageCollection(folder, filetype,
                                  crop_specs=cropping,
                                  processing_specs=processing,
                                  thresholding_specs=threshold,
                                  min_particle_size=4, shape_tol=0.4)
img = collection.images['test_8_X07.tif']

fig, ax = plt.subplots(ncols=5, figsize=(14, 7))
ax[0].imshow(img.raw, cmap='viridis')
ax[1].imshow(img.filtered, cmap='gray')
ax[2].imshow(img.masked, cmap='gray')
ax[3].imshow(img.draw_particles())
linecolor=int(np.round(img.original.max(),0))
cv2.rectangle(img.original, (cropping['xmin'], cropping['ymin']), (cropping['xmax'], cropping['ymax']), linecolor, 2)
ax[4].imshow(img.original, cmap='gray')
plt.show()

# image segmentation
dividing = img.masked
distance = ndi.distance_transform_edt(dividing)
local_maxi = feature.peak_local_max(distance, indices=False, min_distance=7)
markers = measure.label(local_maxi)
segmented_particles = segmentation.watershed(-distance, markers, mask=dividing)

# -- Overlay the images --
color_labels = color.label2rgb(segmented_particles, img.raw*50, alpha=0.2, bg_label=0)

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(color_labels)
ax.set_title('Segmentation result over raw image')
plt.show()
