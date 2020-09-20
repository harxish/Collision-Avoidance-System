import re 
import numpy as np

IOU_THRESHOLD =0.80

def read_annotation(path):

    file = open(path,'r')
    data = file.readlines()
    Annotations = []
    actual_present = 0
    
    for i in data:
        datum = list(map(int , re.findall(r'\d+', i)))[1:]
        if datum:
            datum.insert(0,1)
            Annotations.append(datum)
            actual_present += 1
        else:
            Annotations.append([0,9,9,9,9])
    #Annotations = list(filter(None ,Annotations)) 
    #   We don't filter the empty lists coz 
    #   it might tamper with the detections performance measurement 
    # print(len(Annotations))
    #Annotations = np.array(Annotations)
    # print(Annotations.shape)
    return Annotations, actual_present
    

def IOU(bb1,bb2):
    # assert bb1['x1'] < bb1['x2']
    # assert bb1['y1'] < bb1['y2']
    # assert bb2['x1'] < bb2['x2']
    # assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    print(bb1, bb2)
    x_left   = max(bb1[0], bb2[0])
    y_top    = max(bb1[1], bb2[1])
    x_right  = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left:
        print('X')

    if y_bottom < y_top:
        print('Y')

    print('END')

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
    

def check_Detection(detections,annotation):

    if (IOU(annotation,detections) > IOU_THRESHOLD):
        return True
    return False