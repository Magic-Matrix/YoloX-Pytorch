
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


def weights_init(model, init_type='normal', init_gain=0.02):
    """
    权重初始化
    :param model: 神经网络
    :param init_type: 初始化规则
    :param init_gain: 初始化
    :return:
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    model.apply(init_func)

def resize(image, size):
    """
    图片缩放（会添加灰边变成正方形图片）
    :param image: 图片
    :param size: 缩放后的图片
    :return:
    """

    ih, iw, _ = image.shape
    w, h = size, size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    new_image = np.ones((h, w, 3), dtype="uint8") * 128

    image = cv2.resize(image, (nw, nh))

    new_image[:nh, :nw] = image


    return new_image

def preprocess(image):
    """
    归一化
    :param image: 图片
    :return:
    """
    image = image.astype(np.float32)

    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image

def dimension(image):
    """
    维度整合
    :param image:
    :return:
    """
    # 添加上batch_size维度并维度转换
    image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)

    return image






