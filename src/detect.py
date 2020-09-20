import numpy as np
import cv2
from scipy import ndimage

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
    detector = cv2.SimpleBlobDetector_create(params)
    keypts = detector.detect(img)

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
        if sol[x, y] >= 3:
            ret_obj.append(obj)

    return ret_obj, tracker