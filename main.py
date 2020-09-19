import numpy as np
import cv2
from utils import *
from perf_utils import *

def video():
    path = "Clip_1.mov"
    cap = cv2.VideoCapture(path)
    t = 1
    prev_border = []
    prev_border_ = []
    tracker = []
    count = 0
    detector_time = 0
    annotation,time = read_annotation("Video_Annotation\Clip_1_gt.txt")
    # print(annotation[4][1:])
   # print(annotation)
    while True:
        ret, img = cap.read()
        if ret is False:
            break

        img = cv2.resize(img, (700, 400))
        left_, right_ = get_Horizon_Hough(img)
        border_ = get_border(left_, right_, img.shape[0:2])
        border_ = EMWA(border_, prev_border_, t)
        left_, right_ = np.array([len(border_)-1, int(border_[-1])]), np.array([0, int(border_[0])])

        left, right = get_Horizon_fitLine(img)
        if left != right:
            border = get_border(left, right, img.shape[0:2])
            border = EMWA(border, prev_border, t)
            left, right = np.array([len(border)-1, int(border[-1])]), np.array([0, int(border[0])])
        else:
            border = prev_border
            left, right = np.array([len(border)-1, int(border[-1])]), np.array([0, int(border[0])])

        left, right = (left + left_) / 2, (right + right_) / 2
        left, right = tuple(left.astype(int)), tuple(right.astype(int))

        objects = detect(img, left, right)
        
        objects, tracker = objects_to_track((400, 700), objects, tracker)
   
        for i in objects:
            start_point = ( int(i[0] - 10),int(i[1] -10))
            end_point = (int(i[0] + 10),int(i[1] + 10))

            if (annotation[count][0] == 1):
                det = [start_point[0],start_point[1],end_point[0],end_point[1]]
                status = check_Detection(det,annotation[count][1:])
            img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 1)
        
            if status == True:
                detector_time = detector_time + 1
           
        count += 1

        cv2.line(img, left, right, 255, 2)

        prev_border = border if border != [] else prev_border
        prev_border_ = border_
        t += 1

        cv2.imshow('Input', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    print(detector_time/time)

if __name__ == "__main__":
    video()
