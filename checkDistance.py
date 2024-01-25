import cv2
import numpy as np
import math
redPoint = [0,0]
phoneCor = [0,0]

def rotate_rect(box, angle):
    # Tìm các điểm góc của bounding box
    top_left, top_right, bottom_right, bottom_left = box
    
    # Tính tâm của bounding box
    center = ((top_left[0] + bottom_right[0]) / 2, (top_left[1] + bottom_right[1]) / 2)
    
    # Tạo ma trận affine transformation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    
    # Áp dụng affine transformation để xoay bounding bo
    rotated_box = cv2.transform(np.array([box]), rotation_matrix)[0]
    
    return np.int0(rotated_box)

def decrease_width_with_rotation(box, decrement, angle):
    # Tính tâm của bounding box
    center = np.mean(box, axis=0)
    # Tính chiều rộng hiện tại của bounding box
    width = np.linalg.norm(box[0] - box[1])
    
    # Tính ma trận xoay
    rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1)
    
    # Áp dụng ma trận xoay cho tất cả các điểm của bounding box
    rotated_box = cv2.transform(np.array([box]), rotation_matrix)[0]
    
    # Tính toán chiều mới cho bounding box
    new_width = width * (1 - decrement)
    
    # Tạo bounding box mới với chiều rộng mới
    new_box = np.array([
        center + (rotated_box[0] - center) * (new_width / width),
        center + (rotated_box[1] - center) * (new_width / width),
        center + (rotated_box[2] - center) * (new_width / width),
        center + (rotated_box[3] - center) * (new_width / width)
    ])
    
    return np.int0(new_box)
def decrease_width(box, decrement):
    # Tính tâm của bounding box
    center = np.mean(box, axis=0)
    # Tính chiều rộng hiện tại của bounding box
    width = np.linalg.norm(box[0] - box[1])
    # Tính toán chiều mới cho bounding box
    new_width = width * (1 - decrement)
    # Tạo bounding box mới với chiều rộng mới
    new_box = np.array([
        center + (box[0] - center) * (new_width / width),
        center + (box[1] - center) * (new_width / width),
        center + (box[2] - center) * (new_width / width),
        center + (box[3] - center) * (new_width / width)
    ])
    return np.int0(new_box)

def decrease_box_size(box, decrement):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # Tính tâm của bounding box
    center = np.mean(box, axis=0)
    # Dịch chuyển mỗi điểm của bounding box ra xa tâm theo các chiều
    new_box = center + (box - center) * (1 - decrement)
    return np.int0(new_box)
def draw_line(img,point1, point2, color=(110, 111, 0), thickness=2):
    cv2.line(img, point1, point2, color, thickness)
    # cv2.imshow("window_name", img) 
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
            cy = int(M["m01"] / M["m00"])
            redPoint[0] = cx
            redPoint[1] = cy
            # redPoint = list(redPoint)
            cv2.circle(frame, (cx, cy), 10, (0, 111, 111), -1)
        text = str(redPoint[0]) + " " + str(redPoint[1])
        cv2.putText(frame,  
                text,  
                (redPoint[0], redPoint[1]),  
                font, 0.5,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4)
    # print(redPoint)

def findCordinateOfPhone(frame, mog):
     # Convert the frame to grayscale
    font = cv2.FONT_HERSHEY_SIMPLEX 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply background subtraction
    fgmask = mog.apply(gray)
    # Apply morphological operations to reduce noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    contours, hierarchy = cv2.findContours( fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    for contour in contours:
        if cv2.contourArea(contour) < 1000 or \
            (cv2.contourArea(contour) > 1000 and cv2.contourArea(contour) < 2000) or \
            (cv2.contourArea(contour) > 4000 and cv2.contourArea(contour) < 6500) or \
            (cv2.contourArea(contour) > 2000 and cv2.contourArea(contour) < 3000) or \
            (cv2.contourArea(contour) >= 19125.5 and cv2.contourArea(contour) <= 19153)  :
            continue
        # else:
        #     print(cv2.contourArea(contour))
        # Get the rotated bounding box
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # newBox = box
        # newBox = np.int0(newBox)
        # Draw the bounding box
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        if box[2][0] <= 1115 and box[2][0] >= 975:
            phoneCor[0] = box[2][0]
            phoneCor[1] = box[2][1]
        text = ""
        for numStr in phoneCor:
            text += (str(numStr) + " ")
        cv2.putText(frame,  
                text,  
                (phoneCor[0], phoneCor[1]),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4)
        draw_line(frame, redPoint, phoneCor)
        # print(phoneCor)
    
cap = cv2.VideoCapture("./video/oven_animation_2.mp4")
mog = cv2.createBackgroundSubtractorMOG2()
while cap.isOpened():
    ret, frame = cap.read()
    font = cv2.FONT_HERSHEY_SIMPLEX 
    if not ret:
        break
    findCordinateOfPhone(frame, mog)
    findCenterOfRedPoint(frame)
    # cv2.putText(frame,  
    #             "Distance " + str(math.dist(redPoint, phoneCor)),  
    #             (50, 50),  
    #             font, 1,  
    #             (0, 255, 255),  
    #             2,  
    #             cv2.LINE_4)
    print(phoneCor)
    # if phoneCor[0] != 0:
    # draw_line(frame, redPoint, phoneCor)
    # Hiển thị frame với tâm của chấm đỏ
    cv2.imshow('Video', frame)

    # # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(60) == ord('q'):
        break

cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all windows

    