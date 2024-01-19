import cv2

def calculate_similarity(img1_path, img2_path):
    # Đọc hai bức ảnh cần so sánh
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Tạo đối tượng SIFT
    sift = cv2.SIFT_create()

    # Tìm keypoint và descriptor của từng ảnh
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Sử dụng KNN để so sánh vector mô tả
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Áp dụng ratio test để lọc các điểm tốt nhất
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Tính tỷ lệ giống nhau
    similarity_ratio = len(good_matches) / min(len(kp1), len(kp2))

    return similarity_ratio

def highlight_differences(img1_path, img2_path):
    # Đọc hai bức ảnh cần so sánh
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Tạo đối tượng SIFT
    sift = cv2.SIFT_create()

    # Tìm keypoint và descriptor của từng ảnh
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Sử dụng KNN để so sánh vector mô tả
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Áp dụng ratio test để lọc các điểm tốt nhất
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Vẽ các keypoint tương ứng với điểm khác nhau
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # Hiển thị ảnh với keypoint khoanh tròn
    cv2.imshow("Differences", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Đường dẫn của hai bức ảnh cần so sánh
image1_path = "./data/frame0.jpg"
image2_path = "./data/frame2.jpg"

# Tính tỷ lệ giống nhau
similarity_ratio = calculate_similarity(image1_path, image2_path)

print(f'Tỷ lệ giống nhau giữa hai ảnh là: {similarity_ratio * 100}%')

highlight_differences(image1_path,image2_path)