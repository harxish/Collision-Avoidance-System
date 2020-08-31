
import numpy as np
import math
from tqdm import tqdm
from scipy import ndimage
import cv2

def preprocessing(img):
    clahe = cv2.createCLAHE(clipLimit = 40)
    contrasted_image = clahe.apply(img)
    blurred = cv2.GaussianBlur(contrasted_image,(5,5),0)
    #cv2.imshow("blurred",blurred)
    return blurred

def detect_fast_features(img):
    fast = cv2.FastFeatureDetector_create()
    keypts  = fast.detect(img)
    objects = []
    for i in keypts:
        objects.append((i.pt[0],i.pt[1]))
    objects = np.array(objects)
    return objects
    
def tracking(old_frame,new_frame,prev_kps):
    
    flow = cv2.calcOpticalFlowPyrLK(old_frame,new_frame,
                                prev_kps, None)
    
    matched_keypoints = match_keypoints(flow, prev_kps)
    return matched_keypoints

def match_keypoints(optical_flow, prev_kps):
    """Match optical flow keypoints

    :param optical_flow: output of cv2.calcOpticalFlowPyrLK
    :param prev_kps: keypoints that were passed to cv2.calcOpticalFlowPyrLK to create optical_flow
    :return: tuple of (cur_matched_kp, prev_matched_kp)
    """
    cur_kps, status, err = optical_flow

    # storage for keypoints with status 1
    prev_matched_kp = []
    cur_matched_kp = []

    if status is None:
        return cur_matched_kp, prev_matched_kp

    for i, matched in enumerate(status):
        # store coords of keypoints that appear in both
        if matched:
            prev_matched_kp.append(prev_kps[i])
            cur_matched_kp.append(cur_kps[i])

    return cur_matched_kp, prev_matched_kp

def get_Horizon_fitLine(img):
    '''
    Find co-ordinates of the horizon in the image.
    Using fit line method.
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

def get_Horizon_Hough(img):
    '''
    Find co-ordinates of the horizon in the image.
    Using Hough Lines.
    Input : BGR Image.
    Output : (x1, y1) and (x2, y2) of the Horizon.
    '''
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray, (3, 3), 0)
    v, sigma = np.median(blurred_image), .33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(blurred_image, lower, upper)

    dilated = cv2.dilate(edged, np.ones((3, 3), dtype=np.uint8))
    lines = cv2.HoughLines(dilated, 1, np.pi / 100, threshold = 200, min_theta=np.pi / 3, max_theta=2 * np.pi / 3)

    if lines is not None:
        rho = np.mean([line[0][0] for line in lines])
        theta = np.mean([line[0][1] for line in lines])
        
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 10000 * (-b)), int(y0 + 10000 * (a)))
        pt2 = (int(x0 - 10000 * (-b)), int(y0 - 10000 * (a)))

        return pt1, pt2
        
    else:
        return ((0, 0), (0, 0))

def get_border(left, right, shape):
    '''
    Find border of the horizon in the image.
    Input : Left and right co-ordinate of the border && Shape(h, w) of the image.
    Output : List of length w containing co-ordinates of the border.
    '''

    height, width = shape
    border = [0]*(width)
    if left == right:
        return border
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
    '''
    Calculates Exponential Moving Weighted Average of the border.
    Input : Current location of the border, previous values of the border, time.
    Output : EMWA on the border with respect to prev_border values.
    '''

    if prev_border == []:
        prev_border = [0]*len(border)

    border, prev_border = np.array(border), np.array(prev_border)
    beta = 0.98
    n =  beta*prev_border + (1-beta)*border
    d = 1 - beta**t
    return n / d if t == 1 else n


def detect(image, left, right):
    '''
    Find coordinates for small obstacles in the image.
    Input : Image.
    Output : Coordinates of potential obstacles.
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
    # detector = cv2.SimpleBlobDetector_create(params)
    # keypts = detector.detect(img)
    fast = cv2.FastFeatureDetector_create() 

# Detect keypoints with non max suppression
    keypts= fast.detect(img, None)

    

    objects = []
    a, b = right[1] - left[1], left[0] - right[0] 
    c = a*(left[0]) + b*(left[1])

    for i in keypts:
        if(a * (i.pt[0]+10) + b * (i.pt[1]+10) - c < 0):
            objects.append(i.pt)

    return objects


def objects_to_track(img_size, objects, tracker):
    '''
    Removes noise from the obstacle array.
    Input : Image Dimension, Object array, tracker array.
    Output : Object array, tracker array.
    '''
    
    partition = 50
    ret_obj = []
    grid = np.zeros((partition, partition))
    sol = np.zeros((partition, partition))

    for obj in objects:
        x, y = map(int, [obj[0]*partition/img_size[1], obj[1]*partition/img_size[0]])
        grid[x, y] = 1.

    if len(tracker) < 20 : 
        tracker.append(grid)
    else:
        tracker.pop(0)
        tracker.append(grid)

    for i in tracker:
        sol += i

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    sol = ndimage.convolve(sol, kernel, mode='constant')

    for obj in objects:
        x, y = map(int, [obj[0]*partition/img_size[1], obj[1]*partition/img_size[0]])
        if sol[x, y] >= 10:
            ret_obj.append(obj)

    return ret_obj, tracker
