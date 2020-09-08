import numpy as np
import math
import cv2

# Use horizon(img) to get left and right of the horizon in the frame.
class Horizon():
    
    def __init__(self):
        
        self.t = 1
        self.prev_border = []
        self.prev_border_ = []

        
    def __call__(self, img):
        '''
        Find co-ordinates of the horizon in the image.
        By calcluating two horizons (fitline and hough transforms) and then passing it to EMWA.
    
        Input : BGR Image.
        Output : (x1, y1) and (x2, y2) of the Horizon.
        '''
        
        left_, right_ = self.get_Horizon_Hough(img)
        border_ = self.get_border(left_, right_, img.shape[0:2])
        border_ = self.EMWA(border_, self.prev_border_, self.t)
        left_, right_ = np.array([len(border_)-1, int(border_[-1])]), np.array([0, int(border_[0])])
    
        left, right = self.get_Horizon_fitLine(img)
        if left != right:
            border = self.get_border(left, right, img.shape[0:2])
            border = self.EMWA(border, self.prev_border, self.t)
            left, right = np.array([len(border)-1, int(border[-1])]), np.array([0, int(border[0])])
        else:
            border = self.prev_border
            left, right = np.array([len(border)-1, int(border[-1])]), np.array([0, int(border[0])])
    
        left, right = (left + left_) / 2, (right + right_) / 2
        left, right = tuple(left.astype(int)), tuple(right.astype(int))
        
        self.prev_border = border if border != [] else self.prev_border
        self.prev_border_ = border_
        self.t += 1
        
        return left, right
    
    
    def get_Horizon_fitLine(self, img):
        '''
        Find co-ordinates of the horizon in the image.
        Using fit line method.
    
        Input : BGR Image.
        Output : (x1, y1) and (x2, y2) of the Horizon.
        '''
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, th3 = cv2.threshold(blurred_image, 40, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        edges = cv2.Canny(th3, 400, 450)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=len)
    
        vx, vy, x, y = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        right = int((-x * vy / vx) + y)
        left = int(((gray.shape[1] - x) * vy / vx) + y)
    
        return ((gray.shape[1]-1,  left), (0, right))
    
    
    def get_Horizon_Hough(self, img):
        '''
        Find co-ordinates of the horizon in the image.
        Using Hough Lines.
    
        Input : BGR Image.
        Output : (x1, y1) and (x2, y2) of the Horizon.
        '''
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray, (3, 3), 0)
        v, sigma = np.median(blurred_image), .33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(blurred_image, lower, upper)
    
        dilated = cv2.dilate(edged, np.ones((3, 3), dtype=np.uint8))
        lines = cv2.HoughLines(dilated, 1, np.pi / 100, threshold = 200, min_theta=np.pi / 3, max_theta=2 * np.pi / 3)
    
        if lines is not None:
            rho = np.mean([line[0][0] for line in lines])
            theta = np.mean([line[0][1] for line in lines])
            
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 10000 * (-b)), int(y0 + 10000 * (a)))
            pt2 = (int(x0 - 10000 * (-b)), int(y0 - 10000 * (a)))
    
            return pt1, pt2
            
        else:
            return ((0, 0), (0, 0))
    
        
    def get_border(self, left, right, shape):
        '''
        Find border of the horizon in the image.
    
        Input : Left and right co-ordinate of the border && Shape(h, w) of the image.
        Output : List of length w containing co-ordinates of the border.
        '''
    
        height, width = shape
        border = [0]*(width)
        if left == right:
            return border
        m = (right[1] - left[1]) / (right[0] - left[0])
    
        for i in range(width):
            y = int((m * (i - left[0])) + left[1])
            if y < 0:
                y = 0
            elif y >= height:
                y = height-1
            border[i] = y
    
        return border
    
    
    def EMWA(self, border, prev_border, t):
        '''
        Calculates Exponential Moving Weighted Average of the border.
    
        Input : Current location of the border, previous values of the border, time.
        Output : EMWA on the border with respect to prev_border values.
        '''
    
        if prev_border == []:
            prev_border = [0]*len(border)
    
        border, prev_border = np.array(border), np.array(prev_border)
        beta = 0.98
        n =  beta*prev_border + (1-beta)*border
        d = 1 - beta**t
        return n / d if t == 1 else n