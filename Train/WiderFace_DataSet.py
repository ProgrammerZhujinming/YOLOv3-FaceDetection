from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os
import time
import random
import torch
import math
from utils import image
import numpy as np
class widerface_detection(Dataset):
    def __init__(self,imgs_path = "../DataSet/WiderFace/WIDER_train/images", gtbox_path = "../DataSet/WiderFace/WIDER_train/wider_face_train_bbx_gt.txt", input_size=608, is_train=True, class_num=80):  # input_size:输入图像的尺度

        img_path = ""
        gtbox_num = 0
        coords = []
        self.train_data = []  # [img_path,[[coord, class_id]]]

        with open(gtbox_path, 'r') as gtbox_file:
            for gtbox_data in gtbox_file:
                if gtbox_data.find(".jpg") != -1:
                    if img_path != "" and len(coords) != 0:
                        img_path = os.path.join(imgs_path, img_path.replace("\n", ""))
                        self.train_data.append([img_path, coords])
                        coords = []
                    img_path = gtbox_data
                elif len(gtbox_data.split(" ")) == 1:
                    gtbox_num = int(gtbox_data)
                else:
                    gtbox_data = gtbox_data.split(" ")
                    xmin = int(gtbox_data[0])
                    ymin = int(gtbox_data[1])
                    gt_w = int(gtbox_data[2])
                    gt_h = int(gtbox_data[3])
                    coords.append([xmin, ymin, xmin + gt_w, ymin + gt_h, 0])

        if img_path != "" and len(coords) != 0:
            img_path = os.path.join(imgs_path, img_path)
            if os.path.exists(img_path):
                self.train_data.append([img_path, coords])
                coords = []

        self.transform_common = transforms.Compose([
            transforms.ToTensor(),  # height * width * channel -> channel * height * width
            transforms.Normalize(mean=(0.408, 0.448, 0.471), std=(0.242, 0.239, 0.234))  # 归一化后.不容易产生梯度爆炸的问题
        ])

        self.input_size = input_size
        self.is_train = is_train
        self.class_num = class_num
        self.small_downsmaple = 8
        self.middle_downsmaple = 16
        self.big_downsmaple = 32
        self.iou_threshold = 0.3

    def __getitem__(self, item):
        transform_seed = random.randint(0, 4)
        img_path, coords = self.train_data[item]
        img = cv2.imread(img_path)

        if transform_seed == 0:  # 原图
            img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
            img = self.transform_common(img)

        elif transform_seed == 1:  # 缩放+中心裁剪
            img, coords = image.center_crop_with_coords(img, coords)
            img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
            img = self.transform_common(img)

        elif transform_seed == 2:  # 平移
            img, coords = image.transplant_with_coords(img, coords)
            img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
            img = self.transform_common(img)

        elif transform_seed == 3:  # 明度调整 YOLO在论文中称曝光度为明度
            img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            H, S, V = cv2.split(img)
            cv2.merge([np.uint8(H), np.uint8(S), np.uint8(V * 1.5)], dst=img)
            cv2.cvtColor(src=img, dst=img, code=cv2.COLOR_HSV2BGR)
            img = self.transform_common(img)

        else:  # 饱和度调整
            img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
            H, S, V = cv2.split(img)
            cv2.merge([np.uint8(H), np.uint8(S * 1.5), np.uint8(V)], dst=img)
            cv2.cvtColor(src=img, dst=img, code=cv2.COLOR_HSV2BGR)
            img = self.transform_common(img)

        small_ground_truth, small_positive_modulus, small_anchor_mark_positive, small_anchor_mark_negative, small_positive_modulus_mark, middle_ground_truth, middle_positive_modulus, middle_anchor_mark_positive, middle_anchor_mark_negative, middle_positive_modulus_mark, big_ground_truth, big_positive_modulus, big_anchor_mark_positive, big_anchor_mark_negative, big_positive_modulus_mark = self.getGroundTruth(coords)

        # 通道变化方法: img = img[:, :, ::-1]

        return img, small_ground_truth, small_positive_modulus, small_anchor_mark_positive, small_anchor_mark_negative, small_positive_modulus_mark, middle_ground_truth, middle_positive_modulus, middle_anchor_mark_positive, middle_anchor_mark_negative, middle_positive_modulus_mark, big_ground_truth, big_positive_modulus, big_anchor_mark_positive, big_anchor_mark_negative, big_positive_modulus_mark

    def __len__(self):
        return len(self.train_data)

    def setInputSize(self, input_size, anchors_size):
        self.input_size = input_size
        self.anchors_size = anchors_size

    def anchor_ground_IoU(self, anchor, ground):  # anchor:width height  ground:centerX centerY width height
        # 1.第一种实现方案 将左上角对齐直接算IoU
        # 2.第二种方案 将anchor与ground中心对齐后计算IoU
        # 注意：这两种方案实现起来得到最终的IoU值是一样的
        interArea = min(anchor[0], ground[0]) * min(anchor[1], ground[1])
        Area_anchor = anchor[0] * anchor[1]
        Area_ground = ground[0] * ground[1]
        return interArea / (Area_anchor + Area_ground - interArea)

    def getGroundTruth(self, coords):

        small_feature_size = round(self.input_size / self.small_downsmaple)
        middle_feature_size = round(self.input_size / self.middle_downsmaple)
        big_feature_size = round(self.input_size / self.big_downsmaple)

        small_ground_truth = np.zeros([small_feature_size, small_feature_size, 3,
                                       5 + self.class_num])  # YOLO V3中正样本的置信度为1, 此处将置信度替换为大小样本宽高损失的权衡系数
        middle_ground_truth = np.zeros([middle_feature_size, middle_feature_size, 3, 5 + self.class_num])
        big_ground_truth = np.zeros([big_feature_size, big_feature_size, 3, 5 + self.class_num])

        # positive mask
        small_anchor_mark_positive = np.zeros([small_feature_size, small_feature_size, 3, 5 + self.class_num])
        middle_anchor_mark_positive = np.zeros([middle_feature_size, middle_feature_size, 3, 5 + self.class_num])
        big_anchor_mark_positive = np.zeros([big_feature_size, big_feature_size, 3, 5 + self.class_num])

        # negative mask
        small_anchor_mark_negative = np.zeros([small_feature_size, small_feature_size, 3, 5 + self.class_num])
        middle_anchor_mark_negative = np.zeros([middle_feature_size, middle_feature_size, 3, 5 + self.class_num])
        big_anchor_mark_negative = np.zeros([big_feature_size, big_feature_size, 3, 5 + self.class_num])

        # positive modulus
        small_positive_modulus = np.zeros([small_feature_size, small_feature_size, 3, 6])
        middle_positive_modulus = np.zeros([middle_feature_size, middle_feature_size, 3, 6])
        big_positive_modulus = np.zeros([big_feature_size, big_feature_size, 3, 6])

        # modulus mask
        small_positive_modulus_mark = np.zeros([small_feature_size, small_feature_size, 3, 6])
        middle_positive_modulus_mark = np.zeros([middle_feature_size, middle_feature_size, 3, 6])
        big_positive_modulus_mark = np.zeros([big_feature_size, big_feature_size, 3, 6])

        # 扣出置信度 首先默认所有的都是负样本
        small_anchor_mark_negative[:, :, :, 4] = 1
        middle_anchor_mark_negative[:, :, :, 4] = 1
        big_anchor_mark_negative[:, :, :, 4] = 1

        for coord in coords:
            # bounding box归一化
            [xmin, ymin, xmax, ymax, class_index] = coord

            ground_width = (xmax - xmin)
            ground_height = (ymax - ymin)

            box_width = ground_width * self.input_size
            box_height = ground_height * self.input_size

            centerX = (xmin + xmax) / 2
            centerY = (ymin + ymax) / 2

            # 计算当前中心点分别落于3个特征尺度下的哪个grid内
            small_indexRow = (int)(centerY * small_feature_size)
            small_indexCol = (int)(centerX * small_feature_size)

            middle_indexRow = (int)(centerY * middle_feature_size)
            middle_indexCol = (int)(centerX * middle_feature_size)

            big_indexRow = (int)(centerY * big_feature_size)
            big_indexCol = (int)(centerX * big_feature_size)

            max_iou = 0
            max_iou_index = -1
            for anchor_index in range(len(self.anchors_size)):
                anchor_size = self.anchors_size[anchor_index]
                iou = self.anchor_ground_IoU(anchor_size, [box_width, box_height])
                # print("iou-index:{} iou:{}".format(anchor_index, iou))
                if iou > self.iou_threshold:

                    if anchor_index < 3 and \
                            small_anchor_mark_positive[small_indexRow][small_indexCol][anchor_index][0] == 0:
                        small_anchor_mark_negative[small_indexRow][small_indexCol][anchor_index][4] = 0
                        # small_anchor_mark_negative[small_indexRow][small_indexCol][anchor_index] = np.zeros(5 + self.class_num) # ingore sample

                    elif anchor_index >= 3 and anchor_index < 6 and \
                            middle_anchor_mark_positive[middle_indexRow][middle_indexCol][anchor_index - 3][0] == 0:
                        middle_anchor_mark_negative[middle_indexRow][middle_indexCol][anchor_index - 3][4] = 0
                        # middle_anchor_mark_negative[middle_indexRow][middle_indexCol][anchor_index - 3] = np.zeros(5 + self.class_num) # confidence_mark

                    elif anchor_index >= 6 and anchor_index < 9 and \
                            big_anchor_mark_positive[big_indexRow][big_indexCol][anchor_index - 6][0] == 0:
                        big_anchor_mark_negative[big_indexRow][big_indexCol][anchor_index - 6][4] = 0
                        # big_anchor_mark_negative[big_indexRow][big_indexCol][anchor_index - 6] = np.zeros(5 + self.class_num) # confidence_mark

                if iou > max_iou:
                    max_iou = iou
                    max_iou_index = anchor_index

            scale_width = math.log(box_width / self.anchors_size[max_iou_index][0])
            scale_height = math.log(box_height / self.anchors_size[max_iou_index][1])
            # 大小物体损失值平衡系数
            scale_adjust_modulu = 2 - ground_width * ground_height

            # 分类标签 label_smooth
            class_list = [1]

            # 定位数据预设
            ground_box = [0, 0, scale_width, scale_height, 1]
            ground_box.extend(class_list)
            modulus = [scale_adjust_modulu, self.input_size * xmin, self.input_size * ymin, self.input_size * xmax,
                       self.input_size * ymax, max_iou_index]

            # print(ground_box)
            # print(max_iou_index)

            if max_iou_index < 3:
                # cnt = cnt + 1
                # 已经使用过的需要标记
                small_anchor_mark_positive[small_indexRow][small_indexCol][max_iou_index] = np.ones(
                    [5 + self.class_num])
                # 定位数据
                center_y = centerY * small_feature_size
                ground_box[1] = center_y - small_indexRow
                # print("small center_y:{} ground_box[1]:{}".format(center_y, ground_box[1]))
                center_x = centerX * small_feature_size
                ground_box[0] = center_x - small_indexCol
                # print("small center_x:{} ground_box[0]:{}".format(center_x, ground_box[0]))

                # modulus[5] = max_iou_index
                small_ground_truth[small_indexRow][small_indexCol][max_iou_index] = np.array(ground_box)
                small_positive_modulus[small_indexRow][small_indexCol][max_iou_index] = np.array(modulus)
                small_positive_modulus_mark[small_indexRow][small_indexCol][max_iou_index] = np.ones([6])
                small_anchor_mark_negative[small_indexRow][small_indexCol][max_iou_index] = np.zeros(
                    [5 + self.class_num])
                # small_anchor_mark_negative[small_indexRow][small_indexCol][max_iou_index][4] = 0
                # print("small assign row:{} col:{}".format(small_indexRow, small_indexCol))

            elif max_iou_index < 6:
                # cnt = cnt + 1
                middle_anchor_mark_positive[middle_indexRow][middle_indexCol][max_iou_index - 3] = np.ones(
                    [5 + self.class_num])
                center_y = centerY * middle_feature_size
                ground_box[1] = center_y - middle_indexRow  # offsetY
                # print("middle center_y:{} ground_box[1]:{}".format(center_y, ground_box[1]))
                center_x = centerX * middle_feature_size
                ground_box[0] = center_x - middle_indexCol  # offsetX
                # print("middle center_x:{} ground_box[0]:{}".format(center_x, ground_box[0]))

                middle_ground_truth[middle_indexRow][middle_indexCol][max_iou_index - 3] = np.array(ground_box)
                middle_positive_modulus[middle_indexRow][middle_indexCol][max_iou_index - 3] = np.array(modulus)
                middle_positive_modulus_mark[middle_indexRow][middle_indexCol][max_iou_index - 3] = np.ones([6])
                middle_anchor_mark_negative[middle_indexRow][middle_indexCol][max_iou_index - 3] = np.zeros(
                    [5 + self.class_num])
                # middle_anchor_mark_negative[middle_indexRow][middle_indexCol][max_iou_index - 3][4] = 0
                # print("middle assign row:{} col:{}".format(middle_indexRow, middle_indexCol))

            else:
                # cnt = cnt + 1
                big_anchor_mark_positive[big_indexRow][big_indexCol][max_iou_index - 6] = np.ones(
                    [5 + self.class_num])
                center_y = centerY * big_feature_size
                ground_box[1] = center_y - big_indexRow
                # print("big center_y:{} ground_box[1]:{}".format(center_y, ground_box[1]))
                center_x = centerX * big_feature_size
                ground_box[0] = center_x - big_indexCol
                # print("big center_x:{} ground_box[0]:{}".format(center_x, ground_box[0]))

                # test------------
                '''
                cy = round((ground_box[1] + big_indexRow) * self.big_downsmaple)
                cx = round((ground_box[0] + big_indexCol) * self.big_downsmaple)
                box_width = math.pow(math.e, ground_box[2]) * self.anchors_size[max_iou_index][0]
                box_height = math.pow(math.e, ground_box[3]) * self.anchors_size[max_iou_index][1]
                sx = max(0, round(cx - box_width / 2))
                sy = max(0, round(cy - box_height / 2))
                ox = min(self.input_size - 1, round(cx + box_width / 2))
                oy = min(self.input_size - 1, round(cy + box_height / 2))
                img = cv2.rectangle(img, (sx, sy), (ox, oy), (0, 255, 0), 1)

                positive_num = positive_num + 1
                '''

                big_ground_truth[big_indexRow][big_indexCol][max_iou_index - 6] = np.array(ground_box,
                                                                                           dtype=np.float)
                big_positive_modulus[big_indexRow][big_indexCol][max_iou_index - 6] = np.array(modulus)
                big_positive_modulus_mark[big_indexRow][big_indexCol][max_iou_index - 6] = np.ones([6])
                big_anchor_mark_negative[big_indexRow][big_indexCol][max_iou_index - 6] = np.zeros(
                    [5 + self.class_num])
                # big_anchor_mark_negative[big_indexRow][big_indexCol][max_iou_index - 6][4] = 0
                # print("big assign row:{} col:{}".format(big_indexRow, big_indexCol))

        return torch.Tensor(small_ground_truth).float(), small_positive_modulus, torch.Tensor(
            small_anchor_mark_positive).bool(), torch.Tensor(small_anchor_mark_negative).bool(), torch.Tensor(
            small_positive_modulus_mark).bool(), \
               torch.Tensor(middle_ground_truth).float(), middle_positive_modulus, torch.Tensor(
            middle_anchor_mark_positive).bool(), torch.Tensor(middle_anchor_mark_negative).bool(), torch.Tensor(
            middle_positive_modulus_mark).bool(), \
               torch.Tensor(big_ground_truth).float(), big_positive_modulus, torch.Tensor(
            big_anchor_mark_positive).bool(), torch.Tensor(big_anchor_mark_negative).bool(), torch.Tensor(
            big_positive_modulus_mark).bool(),


