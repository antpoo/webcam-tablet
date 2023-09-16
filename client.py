import cv2, threading, time
import numpy as np
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import perspective
import detect_hands
import mediapipe as mp


mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.15, min_tracking_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

def update_camera_feed():
    global click_circles  

    ret, frame = cap.read() 
    if ret:
        if roi_coordinates and len(roi_coordinates) == 4:
            pts = np.array(roi_coordinates, dtype=np.float32)
            warped = perspective.transform(frame, pts)
            imageRGB = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            results = hands.process(imageRGB)
            # saved_results = results
            # if results.multi_hand_landmarks:
            #     for handLms in results.multi_hand_landmarks: # working with each hand
            #         for id, lm in enumerate(handLms.landmark):
            #             h, w, c = warped.shape
            #             cx, cy = int(lm.x * w), int(lm.y * h)

            #         mpDraw.draw_landmarks(warped, handLms, mpHands.HAND_CONNECTIONS)

        else:
            warped = frame

        for circle in click_circles:
            x, y, radius, alpha = circle
            if alpha > 0:
                color = (255, 255, 255)  
                overlay = warped.copy()
                cv2.circle(overlay, (x, y), radius, color, -1) 
                cv2.addWeighted(overlay, alpha, warped, 1 - alpha, 0, warped)

                circle[2] += 2  
                circle[3] -= 0.02 

        click_circles = [circle for circle in click_circles if circle[2] <= 20 or circle[3] > 0]

        frame_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        label.config(image=img_tk)
        label.img = img_tk

    root.after(30, update_camera_feed)

def animate_circles():
    global click_circles

    while True:
        for circle in click_circles:
            x, y, radius, alpha = circle
            if alpha > 0:
                radius += 2  
                alpha -= 0.02  
                circle[2] = radius
                circle[3] = alpha

        click_circles = [circle for circle in click_circles if circle[2] <= 20 or circle[3] > 0]

        time.sleep(0.05) 

def on_click(event):
    global roi_coordinates
    global click_circles

    x, y = event.x, event.y

    if len(roi_coordinates) < 4:
        roi_coordinates.append([x, y])

    click_circles.append([x, y, 0, 0.5]) 

    if len(roi_coordinates) == 4:
        update_camera_feed()


# Create a Tkinter window
root = tk.Tk()
root.title("Camera Feed")

# Create a label to display the camera feed
label = Label(root)
label.pack()

cap = cv2.VideoCapture(1)
cap.set(3, 1920)
cap.set(4, 1080)

roi_coordinates = []
click_circles = []

root.bind('<Button-1>', on_click)

update_camera_feed()

animation_thread = threading.Thread(target=animate_circles)
animation_thread.daemon = True
animation_thread.start()

# run the root loop 
root.mainloop()

# Release the camera and close the OpenCV window when done
cap.release()
cv2.destroyAllWindows()