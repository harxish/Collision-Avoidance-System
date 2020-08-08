import numpy as np
import cv2

def get_Horizon(img):
    '''
    Find co-ordinates of the horizon in the image.

    Input : BGR Image.
    Output : (x1, y1) and (x2, y2) of the Horizon.
    '''
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, th3 = cv2.threshold(blurred_image, 40, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    edges = cv2.Canny(th3, 400, 450)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=len)

    vx, vy, x, y = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    right = int((-x * vy / vx) + y)
    left = int(((gray.shape[1] - x) * vy / vx) + y)

    return ((gray.shape[1]-1,  left), (0, right))

def get_border(left, right, shape):
    '''
    Find border of the horizon in the image.

    Input : Left and right co-ordinate of the border && Shape(h, w) of the image.
    Output : List of length w containing co-ordinates of the border.
    '''

    height, width = shape
    border = [0]*(width)
    m = (right[1] - left[1]) / (right[0] - left[0])

    for i in range(width):
        y = int((m * (i - left[0])) + left[1])
        if y < 0:
            y = 0
        elif y >= height:
            y = height-1
        border[i] = y

    return border

def CMO(img, border):
    I_op = cv2.erode(cv2.dilate(img, (5, 5)), (5, 5))
    I_cls = cv2.dilate(cv2.erode(img, (5, 5)), (5, 5))
    return I_op - I_cls

def temporal_filter(img):
    pass