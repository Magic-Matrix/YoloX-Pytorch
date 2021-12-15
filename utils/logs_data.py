from torch.utils.tensorboard import SummaryWriter
import os


class MatrixLog(object):
    def __init__(self, path):

        files = os.listdir(path)

        for file in files:
            os.remove(os.path.join(path, file))

        self.writer = SummaryWriter(path)

        self.datas = {}

    def add_scalar(self, group, tag, value, step):
        """
        添加标量
        :param group: 分组
        :param tag: 标签
        :param value: 数字
        :param step: 迭代次数
        :return:
        """
        if group not in self.datas.keys():
            # 没有这个分组，直接创建分组，再创建列表
            self.datas[group] = {tag: []}
        else:
            if tag not in self.datas[group].keys():
                # 有这个分组，直接创建列表
                self.datas[group][tag] = []

            else:
                # 添加数字
                self.datas[group][tag].append(value)

        self.writer.add_scalar(group + '/' + tag, value, step)

    def get_data(self):
        """
        获取存档
        :return:
        """
        return self.datas