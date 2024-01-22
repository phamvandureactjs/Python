import cv2

import numpy as np

import matplotlib.pyplot as plt

cap = cv2.VideoCapture("./video/oven_animation_2.mp4") # Connect to the default camera

# Create the MOG2 background subtractor object
mog = cv2.createBackgroundSubtractorMOG2()

while True:
    _, frame = cap.read() # Read a frame from the video stream
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the frame
    cv2.imshow('Video Stream', frame)

     # Apply background subtraction
    fgmask = mog.apply(gray)
    
    # Apply morphological operations to reduce noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    for contour in contours:
        # Draw bounding box around contour
        x, y, w, h = cv2.boundingRect(contour)
        print(x, y)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Display the foreground mask
    cv2.imshow('Motion Detection', frame)
    if cv2.waitKey(25) == ord('q'): # Exit if the 'q' key is pressed
        break
        
cap.release() # Release the camera
cv2.destroyAllWindows() # Close all windows