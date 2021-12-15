import numpy as np
from PIL import Image
import cv2


def readImage(path):
    """
    使用 opencv 读取图片
    :param path: 路径
    :return:
    """
    # 读取图片
    image = cv2.imread(path)
    # 颜色通道转换
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image



def preprocess_input(image):
    """
    归一化
    :param image: 图片
    :return:
    """

    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image


def dataset_collate(batch):
    """
    数据处理
    :param batch: 每个 batch 的数据
    :return:
    """
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)

    # 整合在一起，变成 (N, 3, h, w)
    images = np.array(images)
    return images, bboxes


