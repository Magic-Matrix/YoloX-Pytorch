

def get_classes(classes_path):
    """
    获取所有的分类
    :param classes_path: 分类名文件
    :return:
    """
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'



def get_lr(optimizer):
    """
    获取学习率
    :param optimizer:
    :return:
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']