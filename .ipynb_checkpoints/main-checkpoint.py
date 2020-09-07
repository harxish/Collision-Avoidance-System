import numpy as np
import cv2
from utils import *
from border import *


def video():
    path = "/home/harish/Documents/Okulo Aerosapce/Dataset/Videos/Clip_5.mov"
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

        for i in objects:
            start_point = ( int(i[0] - 10), int(i[1] -10))
            end_point = (int(i[0] + 10), int(i[1] + 10))
            img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 1)
        
        cv2.line(img, left, right, 255, 2)

        cv2.imshow('Input', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    video()