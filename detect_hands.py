import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
import cv2

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.15, min_tracking_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

def get_landmarks(image):
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    saved_results = results
    # if results.multi_hand_landmarks:
    #     for handLms in results.multi_hand_landmarks: # working with each hand
    #         for id, lm in enumerate(handLms.landmark):
    #             h, w, c = image.shape
    #             cx, cy = int(lm.x * w), int(lm.y * h)

    #         mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

    return image, saved_results
