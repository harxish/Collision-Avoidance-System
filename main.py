import numpy as np
import cv2
from utils import *

path = "/home/harish/Documents/Okulo Aerosapce/Collision-Avoidance-System/Dataset/Images/1.png"
img = cv2.imread(path)
img = cv2.resize(img, (700, 400))

left, right = get_Horizon(img)
border = get_border(left, right, img.shape[0:2])
C = CMO(img, border)

cv2.imshow("Input", img)
cv2.imshow("CMO", C)
cv2.waitKey(0)
cv2.destroyAllWindows()