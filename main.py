import cv2
import mouse
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
import pyautogui

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, model_complexity=1)
mpDraw = mp.solutions.drawing_utils

tips = [4, 8, 12, 16, 20]

pts = [(0, 0), (0, 0), (0, 0), (0, 0)]
pointIndex = 0

# Calculate destination points to match the input image size
pts2 = np.float32([[0, 0], [1920, 0], [0, 1080], [1920, 1080]])

cap = cv2.VideoCapture(1)
cap.set(3, 1920)
cap.set(4, 1080)
cap.set(15, -4)

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

            image = frame
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
                    for a in range(4):
                        for b in range(4):
                            image = cv2.line(image, pts[a], pts[b], (255, 0, 255), 8)

                    tl_x, tl_y = pts[3]
                    bl_x, bl_y = pts[1]
                    tr_x, tr_y = pts[2]
                    br_x, br_y = pts[0]
                    
                    paper_left_pos = min(tl_x, bl_x)
                    paper_top_pos = min(tl_y, tr_y)
                    paper_right_pos = max(tr_x, br_x)
                    paper_bottom_pos = max(bl_y, br_y)

                    xPos_m = handLms.landmark[8].x * 1920
                    yPos_m = handLms.landmark[8].y * 1080

                    if paper_left_pos <= xPos_m <= paper_right_pos and paper_top_pos <= yPos_m <= paper_bottom_pos:
                        # xPos = (paper_right_pos - xPos_m) / paper_left_pos * 1920
                        # yPos = (paper_bottom_pos - yPos_m) / paper_top_pos * 1080
                        xPos = (xPos_m - paper_right_pos) / (paper_left_pos - paper_right_pos)
                        yPos = (yPos_m - paper_bottom_pos) / (paper_top_pos - paper_bottom_pos)

                        print(xPos, yPos)
                        mouse.move(xPos * 1920, yPos * 1080, absolute=True)
                    try:
                        image = cv2.putText(image, str(xPos) + " " + str(yPos), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 225), 2, cv2.LINE_AA)
                    except:
                        pass
                    image = cv2.putText(image, str(xPos_m) + " " + str(yPos_m), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 225), 2, cv2.LINE_AA)

            cv2.imshow('Perspective Transformation', image)
            key = cv2.waitKey(1)

            plt.show()
            if key == 27:
                break

            # print(pts)

cap.release()
cv2.destroyAllWindows()


