import torch
from torchvision.ops import boxes


class Decode(object):
    def __init__(self, input_shape, output_shape, num_classes,
                 conf_thres=0.5, nms_thres=0.4):
        """

        :param input_shape:
        :param output_shape:
        :param num_classes: 分类数量
        """
        grids = []
        strides = []

        # 计算出每种特征图的尺寸
        hw = [torch.Size(shape) for shape in output_shape]

        # 利用循环针对每种特征图进行计算
        for h, w in hw:
            # 根据特征层的高宽生成网格点
            # 两个组成坐标
            grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])

            # 坐标拼接
            # (H, W) 和 (H, W)拼接 (H, W, 2) -> (1, H * W, 2)
            grid = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
            shape = grid.shape[:2]
            grids.append(grid)

            # 滑动步长填充
            strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h))

        # 将网格点堆叠到一起
        # (1, number_boxes, 2)
        self.grids = torch.cat(grids, dim=1).type(torch.float32)
        # 滑动堆叠在一起
        # (1, number_boxes, 1)
        self.strides = torch.cat(strides, dim=1).type(torch.float32)

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.conf_thres = conf_thres
        self.nms_thres = nms_thres


    def decode_boxes(self, outputs):
        """
        神经网络输出数据转换
        :param outputs: 神经网络输出
        :return:
        """

        # 后两个维度拼在一起
        # (N, 4 + 1 + num_classes, H, W) -> (N, 4 + 1 + num_classes, H * W)
        outputs = [x.flatten(start_dim=2) for x in outputs]

        # 三种特征图全部拼在一起
        # (N, 4 + 1 + num_classes, number_boxes)
        outputs = torch.cat(outputs, dim=2)

        # 维度转换
        # (N, 4 + 1 + num_classes, number_boxes) -> (N, number_boxes, 4 + 1 + num_classes)
        outputs = outputs.permute(0, 2, 1)

        # 使用 sigmoid 计算每种分类的概率，前面4个数值不是概率
        outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])

        # 根据网格点进行解码
        # 中心坐标和网格进行相加，乘上步长
        outputs[..., :2] = (outputs[..., :2] + self.grids) * self.strides
        # 框的宽高使用e指数可以防止出现负数，乘上步长
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * self.strides

        # 归一化，针对任何尺寸都可以合适
        outputs[..., [0, 2]] = outputs[..., [0, 2]] / self.input_shape[1]
        outputs[..., [1, 3]] = outputs[..., [1, 3]] / self.input_shape[0]

        return outputs

    def non_max_suppression(self, outputs):

        # 点框格式转换成角点格式
        box_corner = outputs.new(outputs.shape)
        box_corner[:, :, 0] = outputs[:, :, 0] - outputs[:, :, 2] / 2
        box_corner[:, :, 1] = outputs[:, :, 1] - outputs[:, :, 3] / 2
        box_corner[:, :, 2] = outputs[:, :, 0] + outputs[:, :, 2] / 2
        box_corner[:, :, 3] = outputs[:, :, 1] + outputs[:, :, 3] / 2
        outputs[:, :, :4] = box_corner[:, :, :4]

        # 针对图片数量进行直接转换
        new_output = [None for _ in range(len(outputs))]

        for i, image_pred in enumerate(outputs):
            # 每张图片进行分析

            # 对种类预测部分取max。
            # class_conf  (num_anchors, 1)    种类置信度
            # class_pred  (num_anchors, 1)    种类
            class_conf, class_pred = torch.max(
                image_pred[:, 5:5 + self.num_classes], 1, keepdim=True)

            # 利用置信度进行第一轮筛选
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= self.conf_thres).squeeze()

            # 拼接出新的结果
            # detections  [num_anchors, 7]
            # 的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
            detections = torch.cat(
                (image_pred[:, :5], class_conf, class_pred.float()), 1)

            # 筛选
            detections = detections[conf_mask]

            # 非极大值抑制
            nms_out_index = boxes.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                self.nms_thres,
            )

            new_output[i] = detections[nms_out_index]

            if new_output[i] is not None:

                # 转换成 numpy 的变量
                new_output[i] = new_output[i].cpu().numpy()


        return new_output

    def __call__(self, outputs):

        outputs = self.decode_boxes(outputs)
        outputs = self.non_max_suppression(outputs)

        return outputs


