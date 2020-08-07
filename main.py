import numpy as np
import cv2
from utils import *

path = "/home/harish/Documents/Okulo Aerosapce/Collision-Avoidance-System/Dataset/Images/1.png"
img = cv2.imread(path)
img = cv2.resize(img, (700, 400))
img = cv2.flip(img, 1)

left, right = get_Horizon(img)
print(left, right)

cv2.line(img, left, right, 255, 2)
cv2.imshow("Horizon", img)
cv2.waitKey(0)
cv2.destroyAllWindows()