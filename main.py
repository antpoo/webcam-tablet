import cv2
import mediapipe as mp

cap = cv2.VideoCapture(1)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap.set(3, 1920)
cap.set(4, 1080)

tips = [4, 8, 12, 16, 20]

while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    # checking whether a hand is detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: # working with each hand
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id in tips:
                    cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Output", image)
    cv2.waitKey(1)

