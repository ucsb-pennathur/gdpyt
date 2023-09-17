import numpy as np
from skimage.io import imread
from skimage.filters import median, gaussian
from skimage.morphology import disk
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

import cv2


for font in [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX]:
    fp = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/synthetic_example/calibration_images/calib_-35.0.tif'
    img = imread(fp)

    canvas = rescale_intensity(img, in_range='image', out_range=np.uint8)
    max_val = int(canvas.max())
    print(max_val)
    color = (255)

    bbox = [100, 100, 50, 50]
    x, y, w, h = [100, 100, 50, 50]

    start_p = (x, y)  # (100, 100)
    end_p = (x + w, y + h)  # (200, 200)
    cv2.rectangle(canvas, start_p, end_p, color, 1)

    coords = (int(bbox[0] - 0.2 * bbox[2]), int(bbox[1] - 0.2 * bbox[3]))
    cv2.putText(canvas, "ID", coords, font, 0.5, color, 2)

    fig, ax = plt.subplots()
    ax.imshow(canvas)
    plt.show()

"""if raw:
    canvas = self.raw.copy()
else:
    canvas = self.filtered.copy()

canvas = rescale_intensity(canvas, in_range='image', out_range=np.uint8)

max_val = int(canvas.max())
color = (max_val, max_val, max_val)
for particle in self.particles:
    cv2.drawContours(canvas, [particle.contour], -1, color, thickness)
    if draw_id:
        bbox = particle.bbox
        coords = (int(bbox[0] - 0.2 * bbox[2]), int(bbox[1] - 0.2 * bbox[3]))
        cv2.putText(canvas, "ID: {}".format(particle.id), coords, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

return canvas"""

add_distortion = False
if add_distortion:
    pad = 40
    img_distortion = np.pad(np.ones_like(img) * np.min(img), pad)
    k = -5e-6

    size_x, size_y = img.shape
    xc, yc = size_x // 2 + pad, size_y // 2 + pad
    xco, yco = 0, 0

    for x in range(size_x):
        xp = x + pad
        for y in range(size_y):
            i = img[x, y]

            yp = y + pad

            # radial distance from image center
            r = np.sqrt((xp - xc) ** 2 + (yp - yc) ** 2)

            # corrected x and y coordinates
            xo = xp - xc
            yo = yp - yc

            xdo = xco + (xo - xco) / (1 + k * r ** 2)
            ydo = yco + (yo - yco) / (1 + k * r ** 2)

            xdo = xdo + xc
            ydo = ydo + yc

            xdo = int(np.round(xdo))
            ydo = int(np.round(ydo))

            """
            xd = xc + (x - xc) / (1 + k * r ** 2)
            yd = yc + (y - yc) / (1 + k * r ** 2)
    
            xd = int(np.round(xd))
            yd = int(np.round(yd))
            """

            img_distortion[xdo, ydo] = i

    # image smoothing
    img_d = median(img_distortion, disk(3))
    img_d = gaussian(img_d, sigma=0.25, preserve_range=True)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.imshow(img)
    ax2.imshow(img_d)
    plt.show()