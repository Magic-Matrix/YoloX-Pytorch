## YoloX-Pytorch

### 网络清单

| 模型名称   |            输入图片大小 | mAP 0.5:0.95 | mAP 0.5 |
| ---------- |   ------------ | ------------ | ------- |
| YoloX_nano |         640x640      | 27.4         | 44.5    |
| YoloX_tiny |          640x640      | 34.7         | 53.6    |
| YoloX_s    |         640x640      | 38.2         | 57.7    |
| YoloX_m    |          640x640      | 44.8         | 63.9    |
| YoloX_l    |         640x640      | 47.9         | 66.6    |
| YoloX_x    |           640x640      | 49.0         | 67.7    |

### 论文

- 原论文地址：[https://arxiv.org/abs/2107.08430](https://arxiv.org/abs/2107.08430)
- 翻译（自己的翻译，仅供学习）：[YoloX-Pytorch/YOLOX_Exceeding_YOLO_Series_in_2021.md](https://github.com/Magic-Matrix/YoloX-Pytorch/blob/master/YOLOX_Exceeding_YOLO_Series_in_2021.md)

### 训练

- 修改此目录下的`train.py`中的配置，根据自己的需求去选择。
- 此次项目提供了VOC数据集的训练过程，如果想训练自己的数据集请看下面的**数据集制作**。
- 提前下载好预训练权重，最好把所有权重放在`./model_data/weights`路径下

### 测试

- 修改此目录下的`detect.py`中的配置，根据自己的需求去选择。
- 从`./weights`路径下选择训练好的权重。

### 数据集制作

数据集制作需要遵循以下规则：

- 无论用什么办法，导出一个文件，放在`./model_data./datas`路径下，编写格式在`./model_data/README.md`中查看。
- 制作分类文件，每行一个分类的名字，放在`./model_data/classes`路径下。

### 使用tensorboard显示

- 此次项目使用了tensorboard显示损失函数，需要提前安装tensorboard。

- 在终端中cd到此项目路径下，直接打入`tensorboard --logdir logs`命令即可打开tensorboard

