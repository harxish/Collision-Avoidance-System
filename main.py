import numpy as np
import cv2

from src.border import Horizon
from src.detect import Detector
from src.SORT import Sort


def video():
    path = "/home/harish/Documents/Okulo Aerosapce/Dataset/Videos/Clip_1.mov"
    cap = cv2.VideoCapture(path)
    detector = Detector()
    horizon = Horizon()
    sort = Sort()
    
    while True:
        ret, img = cap.read()
        if ret is False:
            break

        img = cv2.resize(img, (700, 400))
        left, right = horizon(img)
        
        objects = detector(img, left, right)
        objs = []
        
        for i in objects:
            start_point = ( int(i[0] - 10), int(i[1] -10))
            end_point = (int(i[0] + 10), int(i[1] + 10))
            objs.append(list(end_point + start_point + (1, )))
            img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 1)

        if len(objects):
            objs = np.array(objs)
            print(objs.shape)
            print(sort.update(objs))
            
        else:
            print(sort.update(np.empty((0, 5))))
                    
        cv2.line(img, left, right, 255, 2)

        cv2.imshow('Input', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    video()