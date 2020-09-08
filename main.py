import numpy as np
import cv2

from src.border import Horizon
from src.detect import *
from src.SORT import Sort

sort = Sort()

def video():
    path = "/home/harish/Documents/Okulo Aerosapce/Dataset/Videos/Clip_1.mov"
    cap = cv2.VideoCapture(path)
    tracker = []
    horizon = Horizon()
    
    while True:
        ret, img = cap.read()
        if ret is False:
            break

        img = cv2.resize(img, (700, 400))
        left, right = horizon(img)
        
        objects = detect(img, left, right)
        objects, tracker = objects_to_track((400, 700), objects, tracker)
        objs = np.empty((0, 5))
        
        for i in objects:
            start_point = ( int(i[0] - 10), int(i[1] -10))
            end_point = (int(i[0] + 10), int(i[1] + 10))
            x = list(end_point + start_point + (1, ))
            objs = np.insert(objs, 0, [x])
            img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 1)
            
        if len(objects):
            print(objs)
            print(objs.shape)
            print(sort.update(objs))
                    
        cv2.line(img, left, right, 255, 2)

        cv2.imshow('Input', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    video()