import math
import random
import time
import cv2
import mouse
import keyboard
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
import threading
import perspective

mpHands = mp.solutions.hands
mpHands2 = mp.solutions.hands

hands = mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, model_complexity=1, max_num_hands=1)
hands2 = mpHands2.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.8, model_complexity=1,
                       max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

tips = [4, 8, 12, 16, 20]

pts = [(0, 0), (0, 0), (0, 0), (0, 0)]
pointIndex = 0
smooth = 7

xPos = [0]
yPos = [0]

# Calculate destination points to match the input image size
pts2 = np.float32([[0, 0], [1920, 0], [0, 1080], [1920, 1080]])

cap = cv2.VideoCapture(1)
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
        img = cv2.putText(img, "Please select 4 points, by clicking on each of them in the order: top left, top right, bottom left, bottom right.",
                            (50, 900), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 225), 2, cv2.LINE_AA)
        cv2.imshow('Select Corners', img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            return False

    return True


def calcPerspective(p, M):
    xPos_m = (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2]) / ((M[2][0] * p[0] + M[2][1] * p[1] + M[2][2]))
    yPos_m = (M[1][0] * p[0] + M[1][1] * p[1] + M[1][2]) / ((M[2][0] * p[0] + M[2][1] * p[1] + M[2][2]))
    return (xPos_m, yPos_m)


cv2.namedWindow('Select Corners')
cv2.setMouseCallback('Select Corners', draw_circle)

click_threshold = 0
hover_threshold = 0

ltx, lty, ltz = 0, 0, 0
lkx, lky, lkz = 0, 0, 0

kb = True
db_mode = False
def dis(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x1-x2) ** 2 + (y1-y2) ** 2 + (z1-z2) ** 2)

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

        M = perspective.init_transform(pts)

        while True:
            success, frame = cap.read()
            success, frame2 = cap.read()


            image = frame
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(imageRGB)
            saved_results = results

            try:
                image = cv2.putText(image, "Press CONTROL to click" if kb else "Press SPACE to calibrate touch clicking", (50, 900), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 225), 2, cv2.LINE_AA)
            except:
                pass

            # checking whether a hand is detected
            if saved_results.multi_hand_landmarks:
                for handLms in saved_results.multi_hand_landmarks:  # working with each hand
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)

                        if id in tips:
                            cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                    mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
                    for a in range(4):
                        for b in range(4):
                            image = cv2.line(image, pts[a], pts[b], (255, 0, 255), 8)


                    tl_x, tl_y = calcPerspective(pts[3], M)
                    bl_x, bl_y = calcPerspective(pts[1], M)
                    tr_x, tr_y = calcPerspective(pts[2], M)
                    br_x, br_y = calcPerspective(pts[0], M)

                    # tl_x, tl_y = pts[3]
                    # bl_x, bl_y = pts[1]
                    # tr_x, tr_y = pts[2]
                    # br_x, br_y = pts[0]

                    paper_left_pos = min(tl_x, bl_x)
                    paper_top_pos = min(tl_y, tr_y)
                    paper_right_pos = max(tr_x, br_x)
                    paper_bottom_pos = max(bl_y, br_y)

                    xPos_m, yPos_m = calcPerspective([handLms.landmark[8].x * 1920, handLms.landmark[8].y * 1000], M)

                    # xPos_m = handLms.landmark[8].x * 1920
                    # yPos_m = handLms.landmark[8].y * 1080
                    zPos_m = handLms.landmark[8].z

                    # xPos = (xPos_m - paper_right_pos) / (paper_left_pos - paper_right_pos)
                    # yPos = (yPos_m - paper_bottom_pos) / (paper_top_pos - paper_bottom_pos)

                    # print(paper_left_pos)
                    # print(paper_top_pos)
                    # print(paper_right_pos)
                    # print(paper_bottom_pos)
                    # print(" ")
                    if 0 < xPos_m < paper_left_pos and 0 < yPos_m < paper_top_pos: #nd zPos_m <= hover_threshold and hover_threshold != 0.0:

                        if not kb and zPos_m >= click_threshold and click_threshold != 0.0 or kb and keyboard.is_pressed('ctrl'):
                            mouse.press()
                            print('click')
                        else:
                            mouse.release()
                        # xPos = (paper_right_pos - xPos_m) / paper_left_pos * 1920
                        # yPos = (paper_bottom_pos - yPos_m) / paper_top_pos * 1080
                        # xPos = (xPos_m - paper_right_pos) / (paper_left_pos - paper_right_pos)
                        # yPos = (yPos_m - paper_bottom_pos) / (paper_top_pos - paper_bottom_pos)
                        # xPos = xPos_m / paper_left_pos
                        # yPos = yPos_m / paper_top_pos

                        # print(xPos)
                        # print(yPos)

                        # get knuckle position
                        kx, ky, kz = handLms.landmark[5].x, handLms.landmark[5].y, handLms.landmark[5].z

                        if len(xPos) == 1:
                            xPos.append((xPos_m - paper_right_pos) / (paper_left_pos - paper_right_pos))
                            yPos.append((yPos_m - paper_bottom_pos) / (paper_top_pos - paper_bottom_pos))
                        else:
                            xPos[0] = xPos[1]
                            yPos[0] = yPos[1]
                            xPos[1] = (xPos_m - paper_right_pos) / (paper_left_pos - paper_right_pos)
                            yPos[1] = (yPos_m - paper_bottom_pos) / (paper_top_pos - paper_bottom_pos)

                        xMove = (xPos[0] + (xPos[1] - xPos[0]) / smooth)
                        yMove = (yPos[0] + (yPos[1] - yPos[0]) / smooth)
                        print(xMove, yMove)
                        mouse.move(xMove * 1920, yMove * 1080, absolute=True)
                    else:
                        mouse.release()

                    if db_mode:
                        try:
                            image = cv2.putText(image, "x and y relative to tablet area: " + str(xPos) + " " + str(yPos), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (255, 0, 225), 2, cv2.LINE_AA)
                        except:
                            pass
                        image = cv2.putText(image, "x: " + str(xPos_m) + " | y: " + str(yPos_m), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (255, 0, 225), 2, cv2.LINE_AA)
                        try:
                            image = cv2.putText(image, "z: " + str(zPos_m) + " | click threshold: " + str(click_threshold), (50, 150),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 225), 2, cv2.LINE_AA)
                        except:
                            pass


            #
            # correctedImage = cv2.warpPerspective(frame2, M, (1920, 1080))
            # results2 = hands2.process(cv2.cvtColor(correctedImage, cv2.COLOR_BGR2RGB))
            # saved_results2 = results2
            # if saved_results2.multi_hand_landmarks:
            #     for handLms2 in saved_results2.multi_hand_landmarks:  # working with each hand
            #         print("updating z")
            #         zPos_m = handLms2.landmark[8].z
            #         try:
            #             if zPos_m >= click_threshold and click_threshold != 0.0:
            #                 mouse.press()
            #                 print('click')
            #                 continue
            #             else:
            #                 mouse.release()
            #         except:
            #             mouse.release()
            cv2.imshow('Image', image)
            #
            key = cv2.waitKey(1)

            plt.show()
            if key == 27:
                break
            elif key == ord('1'):
                kb = not kb
            elif key == ord(' '):
                if not kb:
                    click_threshold = zPos_m
            elif key == ord('0'):
                db_mode = not db_mode

            # elif key == ord('1'):
            #     hover_threshold = zPos_m
            #     print("Hover Threshold:", hover_threshold)

            try:
                ltx, lty, ltz = xMove, yMove, zPos_m
                lkx, lky, lkz = kx, ky, kz
            except:
                pass

            # print(pts)

cap.release()
cv2.destroyAllWindows()
