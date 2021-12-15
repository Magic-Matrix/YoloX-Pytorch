
import torch
import torch.nn as nn
import numpy as np
from .layers import YOLOPAFPN, YOLOXHead
from .utils import resize, preprocess, dimension
from .decode import Decode


class YoloX(nn.Module):
    def __init__(self, num_classes, phi, input_shape=640):
        super().__init__()

        self.input_shape = (input_shape, input_shape)

        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        depth, width = depth_dict[phi], width_dict[phi]
        depthwise = True if phi == 'nano' else False

        self.backbone = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.head = YOLOXHead(num_classes, width, depthwise=depthwise)
        self.num_classes = num_classes

        self.conf_thres = 0.5
        self.nms_thres = 0.3

        self.decode = Decode(self.input_shape, [[80, 80], [40, 40], [20, 20]], num_classes)

    def forward(self, x):
        fpn_outs = self.backbone.forward(x)
        outputs = self.head.forward(fpn_outs)
        return outputs

    def load(self, path):
        self.load_state_dict(torch.load(path))


    def predict(self, image, cuda=False):
        self.eval()

        # 缩放图片，添加灰边，整合成正方形
        input_image = resize(image, self.input_shape[0])

        # 图片归一化
        input_image = preprocess(input_image)

        # 维度转换
        input_image = dimension(input_image)

        # 图片转换成 tensor 类型
        input_image = torch.from_numpy(input_image)

        if cuda:
            # 放入GPU
            input_image = input_image.cuda()
            self.cuda()
        else:
            self.cpu()

        with torch.no_grad():
            # 输出
            output = self(input_image)

        results = self.decode(output)

        h, w, _ = np.shape(image)

        size = max(np.shape(image)[:2])

        results = results[0] * np.array([size, size, size, size, 1, 1, 1])

        outputs = []

        for result in results:
            outputs.append({
                "box": result[:4].astype(int).tolist(),
                "score": result[4] * result[5],
                "class": int(result[6])
            })



        return outputs



























