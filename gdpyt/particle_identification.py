import cv2
import imutils
import numpy as np

def apply_threshold(img, invert=False):
    if invert:
        thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh_img

def identify_contours(particle_mask):
    contours = cv2.findContours(particle_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    bboxes = [cv2.boundingRect(contour) for contour in contours]

    return contours, bboxes

def identify_circles(image):
    h, w = image.shape
    min_radius = 3
    circles = cv2.HoughCircles(image.copy(), cv2.HOUGH_GRADIENT, dp=1.1, minDist=min_radius * 2,
                               param1=20, param2=5, minRadius=min_radius, maxRadius=int(np.minimum(h, w) / 2))
    # Create contours
    n_pts = 360
    contours = []
    d_angle = 2 * np.pi / n_pts

    if circles is not None:
        for circle in circles[0]:
            center = np.array([[circle[0], circle[1]]])
            contour = [center + circle[2]*np.array([[np.cos(i * -d_angle), np.sin(i * -d_angle)]])
                       for i in range(n_pts)]
            contour = np.unique(np.array(contour).astype(np.int), axis=0)
            contours.append(contour)

        bboxes = [cv2.boundingRect(contour) for contour in contours]
    else:
        bboxes = []

    return contours, bboxes

