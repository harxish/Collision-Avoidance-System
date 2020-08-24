import numpy as np
import cv2
from utils import *

def image():
    path = "/home/harish/Documents/Okulo Aerosapce/Collision-Avoidance-System/Dataset/Images/1.png"
    img = cv2.imread(path)
    img = cv2.resize(img, (700, 400))

    left, right = get_Horizon(img)
    border = get_border(left, right, img.shape[0:2])
    objects = detect(img, left, right)

    for i in objects:
        start_point = ( int(i[0] - 70), int(i[1] - 70))
        end_point = (int(i[0] + 70), int(i[1] + 70))
        img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 1)

    objects_to_track((400, 700), objects, None)

    cv2.line(img, left, right, 255, 2)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video():
    path = "/home/harish/Documents/Okulo Aerosapce/Collision-Avoidance-System/Dataset/Videos/Clip_1.mov"
    cap = cv2.VideoCapture(path)
    t = 1
    prev_border = []
    tracker = []

    while True:
        ret, img = cap.read()
        if ret is False:
            break

        img = cv2.resize(img, (700, 400))

        left, right = get_Horizon(img)
        border = get_border(left, right, img.shape[0:2])
        border = EMWA(border, prev_border, t)
        left, right = (len(border)-1, int(border[-1])), (0, int(border[0]))
        objects = detect(img, left, right)
        objects, tracker = objects_to_track((400, 700), objects, tracker)

        for i in objects:
            start_point = ( int(i[0] - 10),int(i[1] -10))
            end_point = (int(i[0] + 10),int(i[1] + 10))
            img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 1)
        cv2.line(img, left, right, 255, 2)

        prev_border = border
        t += 1

        cv2.imshow('Input', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    video()