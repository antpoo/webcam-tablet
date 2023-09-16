import cv2
import numpy as np


def transform(frame, pts):    
    width = 888
    height = 500

    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(pts, dst_pts)

    warped = cv2.warpPerspective(frame, matrix, (width, height))

    return warped
