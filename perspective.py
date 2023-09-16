import cv2
import numpy as np

width_ratio = 1
height_ratio = 1
width = 888
height = 500


def init_transform(pts):
    global width, height, width_ratio, height_ratio

    selected_width = abs(pts[1][0] - pts[0][0])  # Calculate the width
    selected_height = abs(pts[2][1] - pts[0][1])  # Calculate the height
    # Calculate destination points to match the selected portion's size
    pts2 = np.float32([[0, 0], [selected_width, 0], [0, selected_height], [selected_width, selected_height]])

    # The four points of the A4 paper in the image
    pts1 = np.float32([ \
        [pts[0][0], pts[0][1]], \
        [pts[1][0], pts[1][1]], \
        [pts[2][0], pts[2][1]], \
        [pts[3][0], pts[3][1]]])

    width_ratio = (max(abs(pts[0][0] - pts[1][0]), abs(pts[2][0] - pts[3][0])))/width
    height_ratio = (max(abs(pts[0][1] - pts[2][1]), abs(pts[1][1] - pts[3][1])))/height
    
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return M
    
def transform(frame, M):    
    global width, height, width_ratio, height_ratio
    warped = cv2.warpPerspective(frame, M, (width, height))

    return warped

def getActualXPos(x):
    if x/width_ratio > 1:
        # Out of bounds
        return False
    return x/width_ratio

def getActualYPos(y):
    if y/height_ratio > 1:
        # Out of bounds
        return False
    return y/height_ratio

