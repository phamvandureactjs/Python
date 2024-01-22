import cv2
import numpy as np

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
    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) < 1000:
            continue
        
        # Get the rotated bounding box
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Draw the bounding box
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

         # Get the coordinates of the top-left corner
        x, y = box[1]

        # Print the coordinates
        print(f"Top-left coordinates: ({x}, {y})")

    # Display the foreground mask
    cv2.imshow('Motion Detection', frame)
    if cv2.waitKey(25) == ord('q'): # Exit if the 'q' key is pressed
        break

cap.release() # Release the camera
cv2.destroyAllWindows() # Close all windows