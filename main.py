import numpy as np
import cv2
from utils import *

def image():
    path = "/home/harish/Documents/Okulo Aerosapce/Collision-Avoidance-System/Dataset/Images/1.png"
    img = cv2.imread(path)
    img = cv2.resize(img, (700, 400))

    left, right = get_Horizon(img)
    border = get_border(left, right, img.shape[0:2])
    keypts = detect(img)
    objects = []

    for i in keypts:
        if(is_obstacle(i.pt, left ,right) == True):
            objects.append(i.pt)

    for i in objects:
        start_point = ( int(i[0] - 70), int(i[1] - 70))
        end_point = (int(i[0] + 70), int(i[1] + 70))
        img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 1)

    # cv2.imwrite('/home/harish/Pictures/Input.png', img)
    # cv2.imwrite('/home/harish/Pictures/CMO.png', im2)

    cv2.line(img, left, right, 255, 2)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video():
    path = "/home/harish/Documents/Okulo Aerosapce/Collision-Avoidance-System/Dataset/Videos/Clip_3.mov"
    cap = cv2.VideoCapture(path)
    t = 1
    prev_border = []

    while True:
        ret, img = cap.read()
        if ret is False:
            break

        img = cv2.resize(img, (700, 400))

        left, right = get_Horizon(img)
        print(f'Acutal : {left, right}')
        border = get_border(left, right, img.shape[0:2])
        border = EMWA(border, prev_border, t)
        left, right = (len(border)-1, int(border[-1])), (0, int(border[0]))
        print(f'EMWA : {left, right}')
        keypts = detect(img)
        objects = []

        for i in keypts:
            if(is_obstacle(i.pt, left ,right) == True):
                objects.append(i.pt)

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