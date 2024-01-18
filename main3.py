import cv2
import numpy as np

# Đọc ảnh
image = cv2.imread("./data/frame0.jpg")

# Chuyển đổi ảnh sang ảnh xám
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Áp dụng GaussianBlur để làm mờ ảnh và giảm nhiễu
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Sử dụng hàm threshold để tạo mask
_, mask = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

# Áp dụng mask để tách đối tượng ra khỏi ảnh nền
result = cv2.bitwise_and(image, image, mask=mask)

# Hiển thị kết quả
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()