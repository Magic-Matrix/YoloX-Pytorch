from random import sample, shuffle

import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from .read_image import DataRead
from .utils import preprocess_input, dataset_collate


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes,
                 epoch_length, mosaic, train, mosaic_ratio=0.9):
        """
        YoloX
        :param annotation_lines: 每张图片的信息（图片路径和框的数据）
        :param input_shape: 图片尺寸
        :param num_classes: 分类个数
        :param epoch_length: 需要迭代次数
        :param mosaic: 是否使用马赛克数据增强
        :param train: 是否是训练模式
        :param mosaic_ratio: 马赛克数据增强因子
        """
        super(YoloDataset, self).__init__()

        # 图片的信息
        self.annotation_lines = annotation_lines

        # 图片个数
        self.length = len(self.annotation_lines)

        # 图片输出尺寸
        self.inputshape = input_shape

        # 分类个数
        self.num_classes = num_classes

        # 迭代次数
        self.epoch_length = epoch_length

        # 是否使用马赛克数据增强
        self.mosaic = mosaic

        # 测试还是训练
        self.train = train

        self.step_now = -1

        # 马赛克数据增强因子
        self.mosaic_ratio = mosaic_ratio

        self.data_read = DataRead(input_shape)

    def __len__(self):
        return self.length

    def __call__(self, index):
        # 加载图片和数据
        return self.data_read(self.annotation_lines[index])


    def __getitem__(self, index):

        # 防止超过
        index = index % self.length

        # 增加一次
        self.step_now += 1

        # 训练时进行数据的随机增强
        # 验证时不进行数据的随机增强
        if self.mosaic:
            # 马赛克数据增强有两个条件：
            # 一个随机条件
            # 迭代次数 x 马赛克数据增强的因数 x 数据集batch数量
            if self.rand() < 0.5 and self.step_now < self.epoch_length * self.mosaic_ratio * self.length:

                # 随机拿出三个图片（三张辅助图片）
                lines = sample(self.annotation_lines, 3)

                # 加入主要图片（主图片）
                lines.append(self.annotation_lines[index])

                # 随机打乱图片顺序
                shuffle(lines)

                # 加载一张马赛克数据增强图片
                image, box = self.data_read.getMosaicData(lines)
            else:
                # 加载一张图片
                # 后期不使用马赛克数据增强的时候才会使用
                image, box = self.data_read.getData(self.annotation_lines[index], random=self.train)
        else:
            # 加载一张图片
            image, box = self.data_read.getData(self.annotation_lines[index], random=self.train)

        # 转换成 float32 类型
        image = np.array(image, dtype=np.float32)

        # 归一化
        image = preprocess_input(image)

        # 维度转换
        image = np.transpose(image, (2, 0, 1))

        # 转换成 float32 类型
        box = np.array(box, dtype=np.float32)

        # 两点坐标转换成中心框格式
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

        return image, box

    def rand(self, a=0., b=1.):
        """
        随机数
        :param a:起始范围
        :param b:终止范围
        :return:
        """
        return np.random.rand()*(b-a) + a




def dataLoader(train_dataset, val_dataset, batch_size, num_workers):
    # 数据加载
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                     num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=dataset_collate)

    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size,
                         num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=dataset_collate)

    return train_loader, val_loader


if __name__ == "__main__":

    # 读取数据集对应的txt
    with open("../model_data/datas/2012_val.txt") as f:
        train_lines = f.readlines()


    dataset = YoloDataset(
        train_lines, (640, 640), 20, 50, mosaic=True, train=True)

