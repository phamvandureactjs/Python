# Import Module
from PIL import Image

def most_common_used_color(img):
	# Get width and height of Image
	width, height = img.size

	# Initialize Variable
	r_total = 0
	g_total = 0
	b_total = 0

	count = 0

	# Iterate through each pixel
	for x in range(0, width):
		for y in range(0, height):
			# r,g,b value of pixel
			r, g, b = img.getpixel((x, y))

			r_total += r
			g_total += g
			b_total += b
			count += 1

	return (r_total/count, g_total/count, b_total/count)

def convert_to_black(image_path, common_color):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path)

    # Chuyển đổi không gian màu từ BGR sang RGB (nếu cần)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Tìm các pixel có giá trị màu bằng common_color
    mask = np.all(img_rgb == common_color, axis=-1)

    # Chuyển các pixel có giá trị màu bằng common_color thành màu đen
    img_rgb[mask] = [0, 0, 0]

    # Hiển thị bức ảnh
    cv2.imshow("Converted Image", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Read Image
img = Image.open(r'C:\Users\HP\Desktop\New folder\mix_color.png')

# Convert Image into RGB
img = img.convert('RGB')

# call function
common_color = most_common_used_color(img)

print(common_color)
# Output is (R, G, B)

convert_to_black
