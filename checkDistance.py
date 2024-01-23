import cv2
import numpy as np

x_center, y_center = 0, 0
x_phone, y_phone = 0, 0
def draw_line(img, point1, point2, color=(0, 111, 0), thickness=2):
    cv2.line(img, point1, point2, color, thickness)
def findCenterOfRedPoint(frame):
    # Chuyển đổi frame sang dạng HSV để dễ dàng xác định màu sắc
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Xác định một phạm vi màu đỏ trong dạng HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    # Tạo mask để chỉ giữ lại phần màu đỏ trong frame
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Tính tâm của đối tượng
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            # x_center = cx
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 10, (0, 111, 111), -1)
    # print(x_center, y_center)

def findCordinateOfPhone(frame, mog):
     # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply background subtraction
    fgmask = mog.apply(gray)
    # Apply morphological operations to reduce noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    contours, hierarchy = cv2.findContours( fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        # Get the rotated bounding box
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # Draw the bounding box
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
    
cap = cv2.VideoCapture("./video/oven_animation_2.mp4")
mog = cv2.createBackgroundSubtractorMOG2()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    findCordinateOfPhone(frame, mog)
    findCenterOfRedPoint(frame)
    # Hiển thị frame với tâm của chấm đỏ
    cv2.imshow('Video', frame)

    # # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(25) == ord('q'):
        break

cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all windows

    