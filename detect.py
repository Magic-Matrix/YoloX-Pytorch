from model import YoloX
import cv2
import argparse
from utils import get_classes, DrawBox

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train the YoloX --Matrix')
    parser.add_argument('--classes_path', type=str, default='model_data/classes/coco_classes.txt',
                        help='分类文件，存储了全部分类的名字')
    parser.add_argument('--model_type', type=str, default='l',
                        help='模型类型，可以选择模型种类：[\'nano\', \'tiny\', \'s\', \'m\', \'l\', \'x\']')
    parser.add_argument('--input_shape', type=int, default=640, help='输入图片的形状')
    parser.add_argument('--model_path', type=str, default="./weights/yolox_l.pth",
                        help='已经训练好的权重')

    parser.add_argument('--image_path', type=str, default="./images/image1.jpg",
                        help='需要处理的图片')
    parser.add_argument('--save_image_path', type=str, default="./images/output.jpg",
                        help='最后的结果保存的图片')

    opt = parser.parse_args()

    # 获取分类名字和分类个数
    class_names, num_classes = get_classes(opt.classes_path)

    # 网络版本
    phi = opt.model_type

    # 需要加载模型的路径
    model_path = opt.model_path

    print()

    # 创建网络
    model = YoloX(num_classes, phi)
    print('>', "网络创建成功:", "YoloX_" + phi)

    # 加载模型
    model.load(model_path)
    print('>', "模型加载成功:", model_path)

    # 图片路径
    image_path = opt.image_path

    # 读取图片，并转换通道
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    print('>', "图片加载成功:", image_path)

    # 检测图片
    outputs = model.predict(image)

    draw = DrawBox(class_names)

    image = draw(image, outputs)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print('>', "新图片保存到了:", opt.save_image_path)

    cv2.imshow("image", image)

    cv2.waitKey(0)

