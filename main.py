import numpy as np
import cv2
from utils import *

def image():
    path = "1.png"
    img = cv2.imread(path)
    img = cv2.resize(img, (700, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Result", img)
    preprocessed = preprocessing(gray)
    keypts = detect_fast_features(preprocessed);
    print(len(keypts))
    # left, right = get_Horizon_fitLine(img)
    # border = get_border(left, right, img.shape[0:2])
    # objects = detect(img, left, right)

    # for i in objects:
    #     start_point = ( int(i[0] - 70), int(i[1] - 70))
    #     end_point = (int(i[0] + 70), int(i[1] + 70))
    #     img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 1)

    # objects_to_track((400, 700), objects, None)

    # cv2.line(img, left, right, 255, 2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video():
    path = "Clip_1.mov"
    cap = cv2.VideoCapture(path)
    t = 1
    prev_border = []
    prev_border_ = []
    tracker = []

    while True:
        ret, img = cap.read()
        if ret is False:
            break

        img = cv2.resize(img, (700, 400))
        # left_, right_ = get_Horizon_Hough(img)
        # border_ = get_border(left_, right_, img.shape[0:2])
        # border_ = EMWA(border_, prev_border_, t)
        # left_, right_ = np.array([len(border_)-1, int(border_[-1])]), np.array([0, int(border_[0])])

        # left, right = get_Horizon_fitLine(img)
        # if left != right:
        #     border = get_border(left, right, img.shape[0:2])
        #     border = EMWA(border, prev_border, t)
        #     left, right = np.array([len(border)-1, int(border[-1])]), np.array([0, int(border[0])])
        # else:
        #     border = prev_border
        #     left, right = np.array([len(border)-1, int(border[-1])]), np.array([0, int(border[0])])

        # left, right = (left + left_) / 2, (right + right_) / 2
        # left, right = tuple(left.astype(int)), tuple(right.astype(int))
        
        # objects = detect(img, left, right)
        # objects, tracker = objects_to_track((400, 700), objects, tracker)

        # for i in objects:
        #     start_point = ( int(i[0] - 10),int(i[1] -10))
        #     end_point = (int(i[0] + 10),int(i[1] + 10))
        #     img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 1)
        
        # cv2.line(img, left, right, 255, 2)

        # prev_border = border if border != [] else prev_border
        # prev_border_ = border_
        # t += 1
        preprocessed = preprocessing(img)
        detect_fast_features(img);
        cv2.imshow('Input', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    image()
