
import torch
from tqdm import tqdm
from utils import get_lr

def eval_one_epoch(model, yolo_loss, optimizer, epoch,
                    val_batch_number,val_loader, epochs, cuda):
    """

    :param model: 神经网络
    :param yolo_loss: 损失函数
    :param optimizer: 优化器
    :param epoch: 当前训练的次数
    :param val_batch_number:测试集 batch 个数
    :param val_loader: 测试集加载器
    :param epochs: 所有训练次数
    :param cuda: 是否使用GPU
    :return:
    """

    val_loss = []

    # 调节成测试模式
    model.eval()

    with tqdm(total=val_batch_number, desc=f'Epoch-train {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
        pbar.set_postfix(**{'val_loss': 0})
        pbar.update(1)
        for iteration, batch in enumerate(val_loader):

            images, targets = batch[0], batch[1]
            with torch.no_grad():

                images = torch.from_numpy(images).float()
                targets = [torch.from_numpy(ann).float() for ann in targets]


                if cuda:
                    images = images.cuda()
                    targets = [ann.cuda() for ann in targets]

                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                outputs = model(images)

                # 计算损失
                loss_value = yolo_loss(outputs, targets)


            val_loss.append(loss_value.item())
            pbar.set_postfix(**{'val_loss': sum(val_loss) / len(val_loss)})
            pbar.update(1)

    val_loss = sum(val_loss) / len(val_loss)

    return val_loss


def train_one_epoch(model, yolo_loss, optimizer, epoch,
              train_batch_number, train_loader, epochs, cuda):

    """
    训练一次
    :param model: 神经网络
    :param yolo_loss: 损失函数
    :param optimizer: 优化器
    :param epoch: 当前训练的次数
    :param train_batch_number: 训练集 batch 个数
    :param train_loader: 训练集加载器
    :param epochs: 所有训练次数
    :param cuda: 是否使用GPU
    :return:
    """
    loss = []

    # 调节成训练模式
    model.train()
    with tqdm(total=train_batch_number, desc=f'Epoch-eval {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
        pbar.set_postfix(**{'loss': 0,
                            'lr': 0})
        pbar.update(1)
        for iteration, batch in enumerate(train_loader):

            # 获取数据
            images, targets = batch[0], batch[1]
            with torch.no_grad():

                images = torch.from_numpy(images).float()
                targets = [torch.from_numpy(ann).float() for ann in targets]

                if cuda:
                    images = images.cuda()
                    targets = [ann.cuda() for ann in targets]

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss_value = yolo_loss(outputs, targets)

            # 反向传播
            loss_value.backward()

            # 梯度下降
            optimizer.step()

            loss.append(loss_value.item())



            pbar.set_postfix(**{'loss': sum(loss) / len(loss),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    loss = sum(loss) / len(loss)

    return loss



def one_epoch(model, yolo_loss, logs, optimizer, epoch,
                    train_batch_number, val_batch_number,
                    train_loader, val_loader, epochs, cuda):
    """
    训练一次
    :param model: 神经网络
    :param yolo_loss: 损失函数
    :param logs: 记录器
    :param optimizer: 优化器
    :param epoch: 当前训练的次数
    :param train_batch_number: 训练集 batch 个数
    :param val_batch_number: 测试集 batch 个数
    :param train_loader: 训练集加载器
    :param val_loader: 测试集加载器
    :param epochs: 所有训练次数
    :param cuda: 是否使用GPU
    :return:
    """

    loss = train_one_epoch(model, yolo_loss, optimizer, epoch,
                           train_batch_number, train_loader, epochs, cuda)

    val_loss = eval_one_epoch(model, yolo_loss, optimizer, epoch,
                              val_batch_number, val_loader, epochs, cuda)

    print('>', "epoch:{:^4d}   train_loss:{:^10s}   eval_loss:{:^10s}".format(
        epoch, str(loss)[:8], str(val_loss)[:8]))

    logs.add_scalar("loss", "train_loss", loss, epoch)
    logs.add_scalar("loss", "eval_loss", val_loss, epoch)

    save_name = 'weights/epoch%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss, val_loss)
    torch.save(model.state_dict(), save_name)

    return model
