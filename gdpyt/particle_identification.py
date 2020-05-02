import cv2
import imutils

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