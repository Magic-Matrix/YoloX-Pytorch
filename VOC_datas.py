from utils import get_classes
import os
import random
import xml.etree.ElementTree as ET


VOCdevkit_path = 'E:/VOCdevkit'

trainval_percent = 0.9
train_percent = 0.9

VOCdevkit_sets = [('2012', 'train'), ('2012', 'val')]
classes_path = 'model_data/voc_classes.txt'
classes, _ = get_classes(classes_path)

def convert_annotation(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml' % (year, image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == "__main__":
    random.seed(0)

    # xml文件夹
    xmlfilepath = os.path.join(VOCdevkit_path, 'VOC2012/Annotations')
    # VOC指引文件夹
    saveBasePath = os.path.join(VOCdevkit_path, 'VOC2012/ImageSets/Main')

    # 获取数据集所有的 xml 文件
    total_xml = os.listdir(xmlfilepath)

    # 文件数量
    num = len(total_xml)

    # 包装迭代
    range_number = range(num)

    # 训练数据集个数
    tv = int(num * trainval_percent)

    # 训练数据集的训练数据个数
    tr = int(tv * train_percent)

    # 训练数据集随机打乱
    trainval = random.sample(range_number, tv)

    # 训练数据集的训练数据随机打乱
    train = random.sample(trainval, tr)

    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')


    for i in range_number:
        name = total_xml[i][:-4]+'\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


    for year, image_set in VOCdevkit_sets:

        image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()

        list_file = open('model_data/datas/%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')
        for image_id in image_ids:
            list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(os.path.abspath(VOCdevkit_path), year, image_id))

            convert_annotation(year, image_id, list_file)
            list_file.write('\n')
        list_file.close()
    print("Generate 2007_train.txt and 2007_val.txt for train done.")


