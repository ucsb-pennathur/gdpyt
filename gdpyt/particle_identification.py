import cv2
import imutils
import numpy as np

def apply_threshold(img, parameter, invert=False):
    if not len(parameter) == 1:
        raise ValueError("Thresholding parameter must be specified as a dictionary with one key and a list containing"
                         "supplementary arguments for that function as a value")

    method = list(parameter.keys())[0]
    if method not in ['otsu','adaptive_mean', 'adaptive_gaussian']:
        raise ValueError("method must be one of ['otsu','adaptive_mean', 'adaptive_gaussian']")
    if invert:
        threshold_type = cv2.THRESH_BINARY_INV
    else:
        threshold_type = cv2.THRESH_BINARY

    if method == 'otsu':
        _, thresh_img = cv2.threshold(img, 0, 255, threshold_type | cv2.THRESH_OTSU)
    elif method == 'adaptive_mean':
        args = parameter[method]
        thresh_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, threshold_type, *args)
    elif method == 'adaptive_gaussian':
        args = parameter[method]
        thresh_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, threshold_type, *args)

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

def merge_particles(particles):
    id_ = None
    merged_contour = None
    for particle in particles:
        if id_ is None:
            id_ = particle.id
        else:
            if not particle.id == id_:
                raise ValueError("Only particles with duplicate IDs can be merged")

        if merged_contour is None:
            merged_contour = particle.contour.copy()
        else:
            merged_contour = np.vstack([merged_contour, particle.contour.copy()])

    new_contour = cv2.convexHull(merged_contour)
    new_bbox = cv2.boundingRect(new_contour)

    return new_contour, new_bbox


