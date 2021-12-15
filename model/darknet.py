
import torch
from torch import nn


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    """
    选择使用哪种激活函数
    :param name: 激活函数名
    :param inplace:
    :return:
    """
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1,
                 act="silu", resize=2):
        """
        Focus层（用于缩放图片）
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param ksize: 卷积核尺寸
        :param act: 激活函数
        """
        super().__init__()
        self.resize = resize
        resize = resize * resize

        self.conv = BaseConv(in_channels * resize, out_channels, ksize, 1, act=act)

    def forward(self, x):

        # 区域像素分块
        patch_top_left = x[..., ::self.resize, ::self.resize]
        patch_bot_left = x[..., 1::self.resize, ::self.resize]
        patch_top_right = x[..., ::self.resize, 1::self.resize]
        patch_bot_right = x[..., 1::self.resize, 1::self.resize]

        # 通道方向的拼接
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1)
        return self.conv(x)


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize,
                 stride, groups=1, bias=False, act="silu"):
        """
        基本卷积块
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param ksize: 卷积核尺寸
        :param stride: 滑动步长
        :param groups: 是否分组
        :param bias: 是否使用偏置
        :param act: 激活函数
        """
        super().__init__()
        pad = (ksize - 1) // 2

        # 卷积
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=ksize, stride=stride, padding=pad,
                              groups=groups,bias=bias)

        # 标准化
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)

        # 激活函数
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        """
        前向传播
        :param x: 输入
        :return:
        """
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        """
        前向传播（不使用标准化）
        :param x: 输入
        :return:
        """
        return self.act(self.conv(x))


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize,
                 stride=1, act="silu"):
        """
        深度可分离卷积
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param ksize: 卷积核尺寸
        :param stride: 滑动步长
        :param act: 激活函数
        """
        super().__init__()

        # 深度可分离卷积
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize,
                              stride=stride, groups=in_channels, act=act, )

        # 1x1卷积调整通道
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        """
        前向传播
        :param x: 输入
        :return:
        """
        x = self.dconv(x)
        return self.pconv(x)


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_sizes=(5, 9, 13), activation="silu"):
        """
        SPP结构
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param kernel_sizes: 多个卷积核尺寸
        :param activation: 激活函数
        """
        super().__init__()

        # 隐藏层为输入一半
        hidden_channels = in_channels // 2

        # 第一个卷积
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)

        # 多尺度池化增强感受野
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])

        # 1x1卷积调整通道并融合
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        """
        前向传播
        :param x: 输入
        :return:
        """
        x = self.conv1(x)
        # 合并
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True,
                 expansion=0.5, depthwise=False, act="silu", ):
        """
        小残差结构
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param shortcut: 是否使用残差
        :param expansion: 缩放因子
        :param depthwise: 是否使用深度可分离卷积
        :param act: 激活函数
        """
        super().__init__()
        # 隐藏层的通道
        hidden_channels = int(out_channels * expansion)

        # 选择卷积类型
        Conv = DWConv if depthwise else BaseConv
        # 1x1卷积通道缩减，缩减率一般是50%
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)

        # 利用3x3卷积进行通道数的拓张。并且完成特征提取
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)

        # 只有选择残差且通道都相等的时候才会添加残差
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        """
        前向传播
        :param x: 输入
        :return:
        """
        # 两个卷积
        y = self.conv2(self.conv1(x))
        if self.use_add:
            # 添加残差
            y = y + x
        return y


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True,
                 expansion=0.5, depthwise=False, act="silu", ):
        """
        大残差结构
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param n: 小残差结构个数
        :param shortcut: 是否使用残差（小残差结构）
        :param expansion: 通道缩减因子
        :param depthwise: 是否使用深度可分离卷积
        :param act: 激活函数
        """

        super().__init__()
        # 隐藏层通道
        hidden_channels = int(out_channels * expansion)

        # 主干部分的初次卷积
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)

        # 大的残差边部分的初次卷积
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)

        # 对堆叠的结果进行卷积的处理
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)

        # 根据循环的次数构建上述Bottleneck残差结构
        module_list = [Bottleneck(hidden_channels, hidden_channels,
                                  shortcut, 1.0, depthwise,
                                  act=act) for _ in range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        """
        前向传播
        :param x: 输入
        :return:
        """

        # x_1是主干部分
        x_1 = self.conv1(x)

        # x_2是大的残差边部分
        x_2 = self.conv2(x)

        # 主干部分利用残差结构堆叠继续进行特征提取
        x_1 = self.m(x_1)

        # 主干部分和大的残差边部分进行堆叠
        x = torch.cat((x_1, x_2), dim=1)

        # 对堆叠的结果进行卷积的处理
        return self.conv3(x)


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul,
                 out_features=("dark3", "dark4", "dark5"),
                 depthwise=False, act="silu", ):
        """
        CSPDarknet主干特征提取网络
        :param dep_mul: 深度缩放尺寸
        :param wid_mul: 通道缩放尺寸
        :param out_features: 输出特征图的位置
        :param depthwise: 是否使用深度可分离卷积
        :param act: 激活函数
        """
        super().__init__()

        # 检查特征图输出位置
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features

        # 选择卷积类型
        Conv = DWConv if depthwise else BaseConv

        # 假设输入图片是(640, 640, 3)
        # 计算基本通道（初始的基本通道是64，再乘上缩放系数）
        base_channels = int(wid_mul * 64)
        # 计算基本深度（初始的基本通道是3，再乘上缩放系数）
        base_depth = max(round(dep_mul * 3), 1)

        # 利用focus网络结构进行特征提取
        # 640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # 第二阶段特征图卷积
        self.dark2 = nn.Sequential(
            # 320, 320, 64 -> 160, 160, 128
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            # 160, 160, 128 -> 160, 160, 128
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        )

        # 第三阶段特征图卷积
        self.dark3 = nn.Sequential(
            # 160, 160, 128 -> 80, 80, 256
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            # 80, 80, 256 -> 80, 80, 256
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        # 第四阶段特征图卷积
        self.dark4 = nn.Sequential(
            # 80, 80, 256 -> 40, 40, 512
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            # 40, 40, 512 -> 40, 40, 512
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        # 第五阶段特征图卷积
        self.dark5 = nn.Sequential(
            # 40, 40, 512 -> 20, 20, 1024
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            # 20, 20, 1024 -> 20, 20, 1024
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            # 20, 20, 1024 -> 20, 20, 1024
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth,
                     shortcut=False, depthwise=depthwise,act=act),
        )

    def forward(self, x):
        outputs = {}

        # 初始部分的输出为320, 320, 64
        x = self.stem(x)
        outputs["stem"] = x

        # dark2的输出为160, 160, 128
        x = self.dark2(x)
        outputs["dark2"] = x

        # dark3的输出为80, 80, 256
        x = self.dark3(x)
        outputs["dark3"] = x

        # dark4的输出为40, 40, 512
        x = self.dark4(x)
        outputs["dark4"] = x

        # dark5的输出为20, 20, 1024
        x = self.dark5(x)
        outputs["dark5"] = x

        return {k: v for k, v in outputs.items() if k in self.out_features}
