import os
from glob import glob
from PIL import Image
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt

# get properly sorted images
path_img1 = "C://Users//du.pham//Pictures//testIMG//DTF_2860.jpg"
path_img2 = "C://Users//du.pham//Pictures//testIMG//DTF_2861.jpg"

idx = 56

img1_rgb = cv2.cvtColor(cv2.imread(path_img1), cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(cv2.imread(path_img2), cv2.COLOR_BGR2RGB)

# convert to grayscale
img1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
img2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

# compute grayscale image difference
grayscale_diff = cv2.subtract(img2, img1)

# Create a figure with a specific size
fig = plt.figure(figsize=(15, 5))

# Subplots
# ax1 = fig.add_subplot(131)
# ax1.imshow(img1_rgb)
# ax1.set_title('Frame 1')

# ax2 = fig.add_subplot(132)
# ax2.imshow(img2_rgb)
# ax2.set_title('Frame 2')

# ax3 = fig.add_subplot(133)
# ax3.imshow(grayscale_diff * 50)  # scale the frame difference to show the noise
# ax3.set_title('Frame Difference')

# Adjust layout
plt.tight_layout()

def get_mask(frame1, frame2, kernel=np.array((9, 9), dtype=np.uint8)):
    """ Obtains image mask
        Inputs: 
            frame1 - Grayscale frame at time t
            frame2 - Grayscale frame at time t + 1
            kernel - (NxN) array for Morphological Operations
        Outputs: 
            mask - Thresholded mask for moving pixels
    """
    frame_diff = cv2.subtract(frame2, frame1)

    # blur the frame difference
    frame_diff = cv2.medianBlur(frame_diff, 3)
    
    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

    mask = cv2.medianBlur(mask, 3)

    # morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask

kernel = np.array((9, 9), dtype=np.uint8)
mask = get_mask(img1, img2, kernel)

# plt.imshow(mask, cmap='gray')
# plt.title("Motion Mask")
# plt.show()


def get_contour_detections(mask, thresh=400):
    """ Obtains initial proposed detections from contours discovered on the mask. 
        Scores are taken as the bbox area, larger is higher.
        Inputs:
            mask - thresholded image mask
            thresh - threshold for contour size
        Outputs:
            detections - array of proposed detection bounding boxes and scores [[x1,y1,x2,y2,s]]
    """
    # get mask contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    detections = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > thresh:  # hyperparameter
            detections.append([x, y, x + w, y + h, area])

    return np.array(detections)

mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
detections = get_contour_detections(mask, thresh=400)

# separate bounding boxes and scores
bboxes = detections[:, :4]
scores = detections[:, -1]

for box in bboxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(mask_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)

# plt.imshow(mask_rgb)
# plt.title("Detected Movers")
# plt.show()



def compute_iou(box1, box2):
    """ Obtains Intersection over union (IOU) of 2 bounding boxes
        Inputs are in the form of:
            xmin, ymin, xmax, ymax = box
        """
    x11, y11, x21, y21 = box1
    x12, y12, x22, y22 = box2

    # get box points of intersection
    xi1 = max(x11, x12) # top left
    yi1 = max(y11, y12)
    xi2 = min(x21, x22) # bottom right
    yi2 = min(y21, y22)

    # compute intersectional area
    inter_area = max((xi2 - xi1 + 1), 0) * max((yi2 - yi1 + 1), 0)
    if inter_area == 0:
        return inter_area

    # compute box areas
    box1_area = (x21 - x11 + 1) * (y21 - y11 + 1)
    box2_area = (x22 - x12 + 1) * (y22 - y12 + 1)

    # return iou
    return inter_area / (box1_area + box2_area - inter_area)


def get_inter_area(box1, box2):
    """
    Obtains bounding box for the intersection area of two bounding boxes
    Inputs are in the form of:
        xmin, ymin, xmax, ymax = box
    """
    x11, y11, x21, y21 = box1
    x12, y12, x22, y22 = box2

    # get box points of intersection
    xi1 = max(x11, x12)  # top left
    yi1 = max(y11, y12)
    xi2 = min(x21, x22)  # bottom right
    yi2 = min(y21, y22)

    # compute intersectional area
    inter_area = max((xi2 - xi1 + 1), 0) * max((yi2 - yi1 + 1), 0)
    if inter_area == 0:
        return None

    return xi1, yi1, xi2, yi2


mask_rgb_iou = mask_rgb.copy()

# only compare unique cases
idx_1, idx_2 = np.triu_indices(len(bboxes), k=1)

for i in range(len(idx_1)):
    b1 = bboxes[idx_1[i]].tolist()
    b2 = bboxes[idx_2[i]].tolist()
    intersection = get_inter_area(b1, b2)
    if intersection is not None:
        x1, y1, x2, y2 = intersection
        cv2.rectangle(mask_rgb_iou, (x1, y1), (x2, y2), (0, 255, 0), 3)

# fig, ax = plt.subplots(1, 2, figsize=(15, 7))
# ax[0].imshow(mask_rgb)
# ax[0].set_title('Detected Movers')
# ax[1].imshow(mask_rgb_iou)
# ax[1].set_title('Detected Movers with Overlapping Bounding Boxes')
# plt.show()

def remove_contained_bboxes(boxes):
    """ Removes all smaller boxes that are contained within larger boxes.
        Requires bboxes to be soirted by area (score)
        Inputs:
            boxes - array bounding boxes sorted (descending) by area 
                    [[x1,y1,x2,y2]]
        Outputs:
            keep - indexes of bounding boxes that are not entirely contained 
                   in another box
        """
    check_array = np.array([True, True, False, False])
    keep = list(range(0, len(boxes)))
    for i in keep: # range(0, len(bboxes)):
        for j in range(0, len(boxes)):
            # check if box j is completely contained in box i
            if np.all((np.array(boxes[j]) >= np.array(boxes[i])) == check_array):
                try:
                    keep.remove(j)
                except ValueError:
                    continue
    return keep

def non_max_suppression(boxes, scores, threshold=1e-1):
    """
    Perform non-max suppression on a set of bounding boxes and corresponding scores.
    Inputs:
        boxes: a list of bounding boxes in the format [xmin, ymin, xmax, ymax]
        scores: a list of corresponding scores 
        threshold: the IoU (intersection-over-union) threshold for merging bounding boxes
    Outputs:
        boxes - non-max suppressed boxes
    """
    # Sort the boxes by score in descending order
    boxes = boxes[np.argsort(scores)[::-1]]

    # remove all contained bounding boxes and get ordered index
    order = remove_contained_bboxes(boxes)

    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        for j in order:
            # Calculate the IoU between the two boxes
            intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
                           max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
            union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
                    (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
            iou = intersection / union

            # Remove boxes with IoU greater than the threshold
            if iou > threshold:
                order.remove(j)
                
    return boxes[keep]

nms_bboxes = non_max_suppression(bboxes, scores, threshold=0.1)

def non_max_suppression_2(boxes, scores, threshold=1e-1):
    """
    Perform non-max suppression on a set of bounding boxes and corresponding scores.
    NOTE: Eventhough we only go through 2 loops here, this way is more complicated and slower!
    Inputs:
        boxes: a list of bounding boxes in the format [xmin, ymin, xmax, ymax]
        scores: a list of corresponding scores 
        threshold: the IoU (intersection-over-union) threshold for merging bounding boxes
    Outputs:
        boxes - non-max suppressed boxes
    """
    # Sort the boxes by score in descending order
    boxes = boxes[np.argsort(scores)[::-1]]

    keep = list(range(0, len(boxes)))
    for i in keep:
        for j in range(0, len(bboxes)):
            # check if box j is completely contained in box i
            if np.all((np.array(boxes[j]) >= np.array(boxes[i])) == np.array([True, True, False, False])):
                try:
                    keep.remove(j)
                except ValueError:
                    continue
            # if no overlap check IOU threshold
            else:
                # Calculate the IoU between the two boxes
                intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
                            max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
                union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
                        (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
                iou = intersection / union

                # Remove boxes with IoU greater than the threshold
                # ensure that we don't remove larger boxes by checking (j > i)
                if (iou > threshold) and (j > i):
                    try:
                        keep.remove(j)
                    except ValueError:
                        continue
    return boxes[keep]


nms_bboxes = non_max_suppression_2(bboxes, scores, threshold=0.1)
len(bboxes), len(nms_bboxes)


nms_bboxes = non_max_suppression(bboxes, scores, threshold=0.1)
len(bboxes), len(nms_bboxes)

mask_rgb_detections = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
for det in nms_bboxes:
    x1, y1, x2, y2 = det
    cv2.rectangle(mask_rgb_detections, (x1,y1), (x2,y2), (255,0,0), 3)

# plt.imshow(mask_rgb_detections)
# plt.title("Non-Max Suppressed Bounding Boxes")

# plt.show()

def get_detections(frame1, frame2, bbox_thresh=400, nms_thresh=1e-3, mask_kernel=np.array((9,9), dtype=np.uint8)):
    """ Main function to get detections via Frame Differencing
        Inputs:
            frame1 - Grayscale frame at time t
            frame2 - Grayscale frame at time t + 1
            bbox_thresh - Minimum threshold area for declaring a bounding box 
            nms_thresh - IOU threshold for computing Non-Maximal Supression
            mask_kernel - kernel for morphological operations on motion mask
        Outputs:
            detections - list with bounding box locations of all detections
                bounding boxes are in the form of: (xmin, ymin, xmax, ymax)
        """
    # get image mask for moving pixels
    mask = get_mask(frame1, frame2, mask_kernel)

    # get initially proposed detections from contours
    detections = get_contour_detections(mask, bbox_thresh)

    # separate bboxes and scores
    bboxes = detections[:, :4]
    scores = detections[:, -1]

    # perform Non-Maximal Supression on initial detections
    return non_max_suppression(bboxes, scores, nms_thresh)


def draw_bboxes(frame, detections):
    for det in detections:
        x1,y1,x2,y2 = det
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)

def create_gif_from_images(save_path : str, image_path : str, ext : str) -> None:
    ''' creates a GIF from a folder of images
        Inputs:
            save_path - path to save GIF
            image_path - path where images are located
            ext - extension of the images
        Outputs:
            None
    '''
