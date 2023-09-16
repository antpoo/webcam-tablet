import cv2
import mouse
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
import pyautogui

mpHands = mp.solutions.hands
<<<<<<< HEAD
hands = mpHands.Hands(min_detection_confidence=0.15, min_tracking_confidence=0.5)
=======
hands = mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, model_complexity=1)
>>>>>>> 2d4e024765265a2213911f28ecb059b2f561d19c
mpDraw = mp.solutions.drawing_utils

tips = [4, 8, 12, 16, 20]

pts = [(0, 0), (0, 0), (0, 0), (0, 0)]
pointIndex = 0

# Calculate destination points to match the input image size
pts2 = np.float32([[0, 0], [1920, 0], [0, 1080], [1920, 1080]])

<<<<<<< HEAD
cap = cv2.VideoCapture(0)
=======
cap = cv2.VideoCapture(1)
>>>>>>> 2d4e024765265a2213911f28ecb059b2f561d19c
cap.set(3, 1920)
cap.set(4, 1080)
cap.set(15, -6)

# mouse callback function
def draw_circle(event, x, y, flags, param):
    global img
    global pointIndex
    global pts

    if event == cv2.EVENT_LBUTTONDOWN:
        print("Mouse Callback!")
        cv2.circle(img, (x, y), 15, (255, 0, 255), cv2.FILLED)
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
        pts2 = np.float32([[0, 0], [selected_width, 0], [0, selected_height], [selected_width, selected_height]])

        # The four points of the A4 paper in the image
        pts1 = np.float32([ \
            [pts[0][0], pts[0][1]], \
            [pts[1][0], pts[1][1]], \
            [pts[2][0], pts[2][1]], \
            [pts[3][0], pts[3][1]]])

        width_ratio = 1
        height_ratio = 1
        # width_ratio = (max(abs(pts[0][0] - pts[1][0]), abs(pts[2][0] - pts[3][0])))/1920
        # height_ratio = (max(abs(pts[0][1] - pts[2][1]), abs(pts[1][1] - pts[3][1])))/1080

        print(width_ratio, height_ratio)
        while True:
            success, frame = cap.read()

<<<<<<< HEAD
            image = cv2.warpPerspective(frame, M, (1920, 1080))
            # image = frame
            # resized = cv2.resize(image, (image.shape[1], image.shape[0]*2))
            resized = image
            imageRGB = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
=======
            image = frame
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
>>>>>>> 2d4e024765265a2213911f28ecb059b2f561d19c
            results = hands.process(imageRGB)
            saved_results = results

            # checking whether a hand is detected
            if saved_results.multi_hand_landmarks:
                for handLms in saved_results.multi_hand_landmarks: # working with each hand
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)

                        if id in tips:
                            cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                    mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
                    xPos = handLms.landmark[8].x / width_ratio
                    yPos = handLms.landmark[8].y / height_ratio
                    print(xPos, yPos)
                    if xPos <= 1 and yPos <= 1:
                        mouse.move(xPos * 1920, yPos * 1080, True)

            cv2.imshow('Perspective Transformation', image)
            key = cv2.waitKey(1)

            plt.show()
            if key == 27:
                break

cap.release()
cv2.destroyAllWindows()


