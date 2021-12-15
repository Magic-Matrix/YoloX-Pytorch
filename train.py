
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from utils import get_classes, boolean_string, train_data_text, MatrixLog, one_epoch
from model import YoloX, weights_init, YOLOLoss
from datas import YoloDataset, dataLoader
import argparse


def train(batch_size, lr, epochs, model, freeze):

    print()
    print('>', "本次训练即将开始")

    # 计算数据集数量
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    # 优化器选择
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-4)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr, weight_decay=5e-4)
    else:
        raise ValueError("优化器选择错误")

    print('>', "使用的优化器为:", optimizer_name)

    # 是否使用余弦退火
    if Cosine_scheduler:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        print('>', "使用的余弦退火调整学习率")
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    # 加载数据集
    train_dataset = YoloDataset(
        train_lines, input_shape, num_classes, epochs, mosaic=mosaic, train=True)
    val_dataset = YoloDataset(
        val_lines, input_shape, num_classes, epochs, mosaic=False, train=False)

    print('>', "训练集和测试集导入成功")

    train_loader, val_loader = dataLoader(
        train_dataset, val_dataset, batch_size, num_workers)

    if freeze:
        # 冻结
        for param in model.backbone.parameters():
            # 关闭梯度计算
            param.requires_grad = False
        print('>', "冻结神经网络")

    for epoch in range(epochs):
        model = one_epoch(model, yolo_loss, logs, optimizer, epoch,
                      epoch_step, epoch_step_val, train_loader, val_loader, epochs, Cuda)
        # 学习率更新
        lr_scheduler.step()

    if freeze:
        # 解冻
        for param in model.backbone.parameters():
            # 开启梯度计算
            param.requires_grad = True
        print('>', "解冻神经网络")

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train the YoloX --Matrix')
    parser.add_argument('--GPU', type=boolean_string, default=True,
                        help='是否启用GPU，默认启用')
    parser.add_argument('--classes_path', type=str, default='model_data/classes/voc_classes.txt',
                        help='分类文件，存储了全部分类的名字')
    parser.add_argument('--model_type', type=str, default='l',
                        help='模型类型，可以选择模型种类：[\'nano\', \'tiny\', \'s\', \'m\', \'l\', \'x\']')
    parser.add_argument('--input_shape', type=int, default=640, help='输入图片的形状')
    parser.add_argument('--train_annotation_path', type=str, default='model_data/datas/2012_train.txt',
                        help='训练集的集合文件')
    parser.add_argument('--val_annotation_path', type=str, default='model_data/datas/2012_val.txt',
                        help='测试集的集合文件')
    parser.add_argument('--model_path', type=str, default="", help='预训练权重是否加载')
    parser.add_argument('--optimizer', type=str, default="Adam", help='优化器选择，Adam或SGD')
    parser.add_argument('--mosaic', type=boolean_string, default=True, help='是否使用马赛克数据增强，默认开启')
    parser.add_argument('--cosine', type=boolean_string, default=True, help='是否使用余弦退火调整学习率')
    parser.add_argument('--freeze', type=boolean_string, default=True, help='是否冻结网络')
    parser.add_argument('--freeze_epoch', type=int, default=50, help='冻结网络训练次数')
    parser.add_argument('--freeze_batch_size', type=int, default=16, help='冻结网络的 batch 尺寸')
    parser.add_argument('--freeze_lr', type=float, default=1e-3, help='冻结网络的学习率')
    parser.add_argument('--epoch', type=int, default=100, help='训练次数')
    parser.add_argument('--batch_size', type=int, default=8, help='batch 尺寸')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num_workers', type=int, default=4, help='多线程读取图片')

    opt = parser.parse_args()

    # 是否使用Cuda
    Cuda = opt.GPU

    # 训练前一定要修改classes_path，使其对应自己的数据集
    classes_path = opt.classes_path

    # 预训练权重，空字符串时不添加预训练权重
    if opt.model_path == "":
        model_path = "model_data/weights/yolox_" + opt.model_type + ".pth"
    else:
        model_path = opt.model_path

    # 输入的shape大小，一定要是32的倍数
    input_shape = [opt.input_shape, opt.input_shape]

    # 所使用的YoloX的版本。nano、tiny、s、m、l、x
    phi = opt.model_type

    # 是否使用马赛克数据增强
    # YOLOX作者强调要在训练结束前的N个epoch关掉Mosaic
    # 因为Mosaic生成的训练图片，远远脱离自然图片的真实分布。
    mosaic = opt.mosaic

    # 是否使用余弦退火学习率
    Cosine_scheduler = opt.cosine

    # 使用冻结网络训练，分为有两个阶段，分别是冻结阶段和解冻阶段
    Freeze_Train = opt.freeze

    # 冻结阶段训练参数
    Init_Epoch = 0
    Freeze_Epoch = opt.freeze_epoch
    Freeze_batch_size = opt.freeze_batch_size
    Freeze_lr = opt.freeze_lr

    # 解冻阶段训练参数
    UnFreeze_Epoch = opt.epoch
    Unfreeze_batch_size = opt.batch_size
    Unfreeze_lr = opt.lr

    # 用于设置是否使用多线程读取数据
    num_workers = opt.num_workers

    optimizer_name = opt.optimizer

    # 获得图片路径和标签
    train_annotation_path = opt.train_annotation_path
    val_annotation_path = opt.val_annotation_path

    # 获取分类名字和分类个数
    class_names, num_classes = get_classes(classes_path)

    opt.class_names = num_classes

    train_data_text(opt)

    # cmd = input("\nTrain?(Y/N):")
    #
    # if not (cmd == 'Y' or cmd == 'y'):
    #     exit()

    print()
    # 创建yolo模型
    model = YoloX(num_classes, phi)

    print('>', "网络创建成功:", "YoloX_" + phi)

    # 权重初始化
    weights_init(model)
    print('>', "模型参数初始化")

    weight = torch.load(model_path)

    if model_path != '':
        device = torch.device('cuda' if Cuda else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    print('>', "模型加载成功")

    # 训练模式
    model.train()

    if Cuda:
        cudnn.benchmark = True
        model = model.cuda()
        print('>', "神经网络放入", "cuda:0" if Cuda else "cpu")

    # 损失函数
    print('>', "损失函数创建成功")
    yolo_loss = YOLOLoss(num_classes)

    # Matrix日志
    logs = MatrixLog("logs/")

    # 读取数据集对应的txt
    with open(train_annotation_path) as f:
        train_lines = f.readlines()

    with open(val_annotation_path) as f:
        val_lines = f.readlines()

    # 数据集的数量
    num_train = len(train_lines)
    num_val = len(val_lines)
    print('>', "训练集数量为:", num_train, "测试集数量为:", num_val)

    # 是否冻结网络
    if Freeze_Train:
        # 冻结训练
        model = train(Freeze_batch_size, Freeze_lr, Freeze_Epoch, model, True)
        # 解冻训练
        model = train(Unfreeze_batch_size, Unfreeze_lr, UnFreeze_Epoch - Freeze_Epoch, model, False)
    else:
        # 直接训练
        model = train(Unfreeze_batch_size, Unfreeze_lr, UnFreeze_Epoch, model, False)

    print('>', "训练结束")
