import numpy as np
import cv2

from src.border import Horizon
from src.detect import Detector
from src.validation import *

horizon = Horizon()
detector = Detector()


def video():
    path = "/home/harish/Documents/Okulo Aerosapce/Dataset/Videos/Clip_1.mov"
    cap = cv2.VideoCapture(path)
    annotations, annotations_time = read_annotation('/home/harish/Documents/Okulo Aerosapce/Dataset/Annotation/Clip_1_gt.txt')
    detector_time, frame_count = 0, 0
    
    while True:
        ret, img = cap.read()
        if ret is False:
            break

        # img = cv2.resize(img, (700, 400))
        left, right = horizon(img)
        
        objects = detector(img, left, right)
        
        for i in objects:
            start_point = ( int(i[0] - 10), int(i[1] -10))
            end_point = (int(i[0] + 10), int(i[1] + 10))
            img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 1)

            if (annotations[frame_count][0] == 1):
                det = [start_point[0], start_point[1], end_point[0], end_point[1]]
                status = check_Detection(det, annotations[frame_count][1:])

            if status:
                detector_time += 1

        frame_count += 1
                    
        cv2.line(img, left, right, 255, 2)

        cv2.imshow('Input', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(frame_count, annotations_time)

if __name__ == "__main__":
    video()