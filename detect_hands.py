import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
import cv2

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.15, min_tracking_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

def get_landmarks(image):
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return hands.process(imageRGB)
