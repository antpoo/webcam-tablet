import cv2
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
import detect_hands
import perspective

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.15, min_tracking_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

tips = [4, 8, 12, 16, 20]

pts = [(0, 0), (0, 0), (0, 0), (0, 0)]
pointIndex = 0

# Calculate destination points to match the input image size
pts2 = np.float32([[0, 0], [1920, 0], [0, 1080], [1920, 1080]])

cap = cv2.VideoCapture(0)

# mouse callback function
def draw_circle(event, x, y, flags, param):
    global img
    global pointIndex
    global pts

    if event == cv2.EVENT_LBUTTONDOWN:
        print("Mouse Callback!")
        pts[pointIndex] = (x, y)
        pointIndex = pointIndex + 1


def selectFourPoints():
    global img
    global pointIndex

    print("Please select 4 points, by double clicking on each of them in the order: \n\
    top left, top right, bottom left, bottom right.")

    while (pointIndex != 4):
        _, img = cap.read()
        cv2.imshow('Select Corners', img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            return False

    return True

cv2.namedWindow('Select Corners')
cv2.setMouseCallback('Select Corners', draw_circle)

while True:
    if (selectFourPoints()):
        cv2.destroyWindow('Select Corners')

        selected_width = abs(pts[1][0] - pts[0][0])  # Calculate the width
        selected_height = abs(pts[2][1] - pts[0][1])  # Calculate the height

        # Calculate destination points to match the selected portion's size
        M = perspective.init_transform(pts)
        while True:
            success, frame = cap.read()

            image = perspective.transform(frame, M)
            results = detect_hands.get_landmarks(frame)

            # checking whether a hand is detected
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks: # working with each hand
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)

                    mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
            
            cv2.imshow('Perspective Transformation', image)
            key = cv2.waitKey(1)

            plt.show()
            if key == 27:
                break

cap.release()
cv2.destroyAllWindows()


