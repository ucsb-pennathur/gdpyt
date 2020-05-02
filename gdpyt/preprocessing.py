import cv2

def calculate_histogram(img):
    return cv2.calcHist([img], [0], None, [256], [0, 256])

def apply_filter(img, func, *args, **kwargs):
    assert callable(func)
    return func(img, *args, **kwargs)


