#------step0: common defination------
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

class_num = 1
input_size_index = 9
img_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
anchor_boxes = [[6, 7],  [8, 10], [10, 13], [12, 14], [14, 18], [19, 24], [27, 34], [44, 58], [99, 131]]
weight_file = "../Train/weights/YOLO_V3_final.pth"
#------step1: model------
from Train.YOLO_V3_Model import YOLO_V3
from model import set_freeze_by_idxs, unfreeze_by_idxs
YOLO_V3 = YOLO_V3(class_num=class_num)
YOLO_V3.load_state_dict(torch.load(weight_file, map_location=torch.device("cpu")))
YOLO_V3 = YOLO_V3.to(device=device)
YOLO_V3.eval()

'''
import torch.quantization
import torch.nn as nn
quantized_model = torch.quantization.quantize_dynamic(
    YOLO_V3, {nn.Conv2d}, dtype=torch.qint8                     #对YOLOv3中的Conv2d进行量化
)
'''

#------step:4 NMS------

def iou(box_one, box_two):
    LX = max(box_one[0], box_two[0])
    LY = max(box_one[1], box_two[1])
    RX = min(box_one[2], box_two[2])
    RY = min(box_one[3], box_two[3])
    if LX >= RX or LY >= RY:
        return 0
    return (RX - LX) * (RY - LY) / ((box_one[2]-box_one[0]) * (box_one[3] - box_one[1]) + (box_two[2]-box_two[0]) * (box_two[3] - box_two[1]))

import math
import numpy as np
def NMS_MultiSacle(small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes, input_size, anchor_boxes, small_downsample = 8, middle_downsample = 16, big_downsample = 32, confidence_threshold=0.8, iou_threshold=0.5):
    predict_boxes = []
    nms_boxes = []

    # batch_size * height * witdh * 3 * (5 + class_num)
    small_bounding_boxes = small_bounding_boxes.cpu().detach().numpy()
    middle_bounding_boxes = middle_bounding_boxes.cpu().detach().numpy()
    big_bounding_boxes = big_bounding_boxes.cpu().detach().numpy()

    small_grids_num = input_size // small_downsample
    middle_grids_num = input_size // middle_downsample
    big_grids_num = input_size // big_downsample

    for batch_data in small_bounding_boxes:
        for row_index in range(small_grids_num):
            for col_index in range(small_grids_num):
                small_gird_predict = batch_data[row_index][col_index]
                max_confidence_index = np.argmax(small_gird_predict[::, 4])
                small_predict = small_gird_predict[max_confidence_index]
                confidence = small_predict[4]
                if confidence < confidence_threshold:
                    continue
                center_x = (col_index + small_predict[0]) * small_downsample
                center_y = (row_index + small_predict[1]) * small_downsample
                width = anchor_boxes[max_confidence_index][0] * math.pow(math.e, small_predict[2])
                height = anchor_boxes[max_confidence_index][1] * math.pow(math.e, small_predict[3])
                xmin = max(0, round(center_x - width / 2))
                ymin = max(0, round(center_y - height / 2))
                xmax = min(input_size - 1, round(center_x + width / 2))
                ymax = min(input_size - 1, round(center_y + height / 2))
                class_index = np.argmax(small_predict[5:])
                predict_box = [xmin, ymin, xmax, ymax, confidence, class_index]
                predict_boxes.append(predict_box)

    for batch_data in middle_bounding_boxes:
        for row_index in range(middle_grids_num):
            for col_index in range(middle_grids_num):
                middle_gird_predict = batch_data[row_index][col_index]
                max_confidence_index = np.argmax(middle_gird_predict[::, 4])
                middle_predict = middle_gird_predict[max_confidence_index]
                confidence = middle_predict[4]
                if confidence < confidence_threshold:
                    continue
                center_x = (col_index + middle_predict[0]) * middle_downsample
                center_y = (row_index + middle_predict[1]) * middle_downsample
                width = anchor_boxes[3 + max_confidence_index][0] * math.pow(math.e, middle_predict[2])
                height = anchor_boxes[3 + max_confidence_index][1] * math.pow(math.e, middle_predict[3])
                xmin = max(0, round(center_x - width / 2))
                ymin = max(0, round(center_y - height / 2))
                xmax = min(input_size - 1, round(center_x + width / 2))
                ymax = min(input_size - 1, round(center_y + height / 2))
                class_index = np.argmax(middle_predict[5:])
                predict_box = [xmin, ymin, xmax, ymax, confidence, class_index]
                predict_boxes.append(predict_box)

    for batch_data in big_bounding_boxes:
        for row_index in range(big_grids_num):
            for col_index in range(big_grids_num):
                big_gird_predict = batch_data[row_index][col_index]
                max_confidence_index = np.argmax(big_gird_predict[::, 4])
                big_predict = big_gird_predict[max_confidence_index]
                confidence = round(big_predict[4], 2)
                if confidence < confidence_threshold:
                    continue
                center_x = (col_index + big_predict[0]) * middle_downsample
                center_y = (row_index + big_predict[1]) * middle_downsample
                width = anchor_boxes[3 + max_confidence_index][0] * math.pow(math.e, big_predict[2])
                height = anchor_boxes[3 + max_confidence_index][1] * math.pow(math.e, big_predict[3])
                xmin = max(0, round(center_x - width / 2))
                ymin = max(0, round(center_y - height / 2))
                xmax = min(input_size - 1, round(center_x + width / 2))
                ymax = min(input_size - 1, round(center_y + height / 2))
                class_index = np.argmax(big_predict[5:])
                predict_box = [xmin, ymin, xmax, ymax, confidence, class_index]
                predict_boxes.append(predict_box)

    while len(predict_boxes) != 0:
        predict_boxes.sort(key=lambda box: box[4])
        assured_box = predict_boxes[0]
        temp = []
        nms_boxes.append(assured_box)
        i = 1
        while i < len(predict_boxes):
            if iou(assured_box, predict_boxes[i]) <= iou_threshold:
                temp.append(predict_boxes[i])
            i = i + 1
        predict_boxes = temp

    return nms_boxes

#------step:5 detection ------
import cv2
from utils import image
import torchvision.transforms as transforms
img_file_name = "../DataSet/WiderFace/WIDER_train/images/15--Stock_Market/15_Stock_Market_Stock_Market_15_3.jpg"
input_size = img_sizes[input_size_index]

transform = transforms.Compose([
    transforms.ToTensor(), # height * width * channel -> channel * height * width
    transforms.Normalize(mean=(0.408, 0.448, 0.471), std=(0.242, 0.239, 0.234))
])
img = cv2.imread(img_file_name)
img = image.resize_image_without_annotation(img, input_size, input_size)
train_data = transform(img)
train_data = train_data.unsqueeze(0)
train_data = train_data.to(device=device)

with torch.no_grad():
    small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes = YOLO_V3(train_data)
NMS_boxes = NMS_MultiSacle(small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes, input_size, anchor_boxes)

for box in NMS_boxes:
    print(box)
    img = cv2.rectangle(img, (box[0],box[1]),(box[2],box[3]),(0,255,0),1)
    confidence = round(box[4],2)
    img = cv2.putText(img, "face-{}".format(confidence),(box[0],box[1]),cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,255),1)
    print("class_name:{} confidence:{}".format("face",confidence))

cv2.imshow("img_detection",img)
cv2.waitKey()
cv2.destroyAllWindows()


