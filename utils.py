import numpy as np
from tqdm import tqdm
from scipy import ndimage
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


def EMWA(border, prev_border, t):

    if prev_border == []:
        prev_border = [0]*len(border)

    border, prev_border = np.array(border), np.array(prev_border)
    beta = 0.945
    n =  beta*prev_border + (1-beta)*border
    d = 1 - beta**t
    return n / d if t == 1 else n

def detect(image):
    '''
    Find keypts small objects in the image.

    Input : Image.
    Output : List of length w containing co-ordinates of the border.
    '''

    I_op = cv2.erode(cv2.dilate(image, (5, 5)), (5, 5))
    I_cls = cv2.dilate(cv2.erode(image, (5, 5)), (5, 5))
    img = I_op - I_cls

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.where(img <= 50, img, 255)
    kernel = np.ones((10, 10), np.uint8)
    img = cv2.dilate(img, kernel)

    params = cv2.SimpleBlobDetector_Params()
    params.blobColor = 255.0
    detector = cv2.SimpleBlobDetector_create(params)
    keypts = detector.detect(img)

    return keypts


def is_obstacle(pt, left, right):
    '''
    Remove noise from potentail objects.

    Input : List of potential obstacles, border-left, border-right.
    Output : Keypts (x, y) of potential obstacles without noise.
    '''
    
    a, b = right[1] - left[1], left[0] - right[0] 
    c = a*(left[0]) + b*(left[1])
    return True if(a * pt[0] + b * pt[1] - c < 0) else False