import cv2
from skimage import metrics
import numpy as np
# Load images
img1_path = "./data/frame0.jpg"
img2_path = "./data/frame0.jpg"
image1 = cv2.imread(img1_path)
image2 = cv2.imread(img2_path)
# image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation = cv2.INTER_AREA)

# Convert images to grayscale
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# Calculate SSIM
ssim_score = metrics.structural_similarity(image1_gray, image2_gray, full=True)
print(f"SSIM Score: ", ssim_score[0])
