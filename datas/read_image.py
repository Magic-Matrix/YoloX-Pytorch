
import cv2
import numpy as np
from PIL import Image
from .utils import readImage

class DataRead(object):

    def __init__(self, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):


        self.jitter = jitter
        self.hue = hue
        self.sat = sat
        self.val = val

        self.input_shape = input_shape

        h, w = input_shape
        self.gery_image = Image.new('RGB', (w, h), (128, 128, 128))


    def __call__(self, annotation_line):
        """
        获取图片和信息
        :param annotation_line:
        :return:
        """

        # 以空格为分隔符，包含 \n
        line = annotation_line.split()

        # 使用opencv读取图片
        image = readImage(line[0])

        # 获得预测框和分类
        box = self.getBoxes(line)

        return image, box


    def getData(self, annotation_line, random=True):
        """
        获取图片和方框
        :param annotation_line: 图片信息
        :param random: 是否随机
        :return: 图片和方框
        """
        # 以空格为分隔符，包含 \n
        line = annotation_line.split()

        # 读取图片
        image = self.imread(line[0])

        # 当前图片的尺寸
        iw, ih = image.size

        # 更改后的图片
        h, w = self.input_shape

        # 获得预测框和分类
        box = self.getBoxes(line)

        if random:

            # 对图像进行缩放并且进行长和宽的扭曲
            new_ar = w / h * self.rand(1 - self.jitter, 1 + self.jitter) / self.rand(1 - self.jitter, 1 + self.jitter)
            scale = self.rand(.25, 2)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # 将图像多余的部分加上灰条
            dx = int(self.rand(0, w - nw))
            dy = int(self.rand(0, h - nh))
            new_image = self.gery_image.copy()
            new_image.paste(image, (dx, dy))
            image = new_image

            # 翻转图像
            flip = self.rand() < .5
            if flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # 色域扭曲
            hue = self.rand(-self.hue, self.hue)
            sat = self.rand(1, self.sat) if self.rand() < .5 else 1 / self.rand(1, self.sat)
            val = self.rand(1, self.val) if self.rand() < .5 else 1 / self.rand(1, self.val)
            x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
            x[..., 0] += hue * 360
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x[:, :, 0] > 360, 0] = 360
            x[:, :, 1:][x[:, :, 1:] > 1] = 1
            x[x < 0] = 0
            image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

            # 对真实框进行调整
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]

        else:
            # 不随机图片

            # 缩放系数
            scale = min(w / iw, h / ih)

            # 新的宽度
            nw = int(iw * scale)
            nh = int(ih * scale)

            # 图片移动到中间位置
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # 缩放图片
            image = image.resize((nw, nh), Image.BICUBIC)

            # 复制一张灰图
            new_image = self.gery_image.copy()

            # 图片放灰图上
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)


            # 对真实框进行调整
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box


        return image_data, box

    def getMosaicData(self, line_list):

        # 图片尺寸
        h, w = self.input_shape

        # 随机最小坐标
        min_offset_x = self.rand(0.25, 0.75)
        min_offset_y = self.rand(0.25, 0.75)

        # 四张图片的随机宽度
        nws = [int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1)),
               int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1))]

        # 四张图片的随机高度
        nhs = [int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1)),
               int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1))]

        # 随机坐标
        place_x = [int(w * min_offset_x) - nws[0], int(w * min_offset_x) - nws[1],
                   int(w * min_offset_x), int(w * min_offset_x)]
        place_y = [int(h * min_offset_y) - nhs[0], int(h * min_offset_y),
                   int(h * min_offset_y), int(h * min_offset_y) - nhs[3]]

        image_datas = []
        box_datas = []


        for index, line in enumerate(line_list):
            # 每一行进行分割
            line_content = line.split()
            # 打开图片
            image = self.imread(line_content[0])

            # 图片的大小
            iw, ih = image.size

            # 整理框的信息
            box = self.getBoxes(line_content)

            # 是否翻转图片
            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # 缩放尺寸
            nw = nws[index]
            nh = nhs[index]
            # 缩放
            image = image.resize((nw, nh), Image.BICUBIC)

            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[index]
            dy = place_y[index]

            # 获取一张灰图
            new_image = self.gery_image.copy()
            # 图片放在灰图上
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            box_data = []

            # 对box进行重新处理
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            # 图片放进列表
            image_datas.append(image_data)

            # 方框放入列表
            box_datas.append(box_data)

        # 将图片分割，放在一起
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)
        new_image = np.zeros([h, w, 3])



        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        # 进行色域变换
        hue = self.rand(-self.hue, self.hue)
        sat = self.rand(1, self.sat) if self.rand() < .5 else 1 / self.rand(1, self.sat)
        val = self.rand(1, self.val) if self.rand() < .5 else 1 / self.rand(1, self.val)
        x = cv2.cvtColor(np.array(new_image / 255, np.float32), cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        new_image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        # 对框进行进一步的处理
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

    def merge_bboxes(self, bboxes, cutx, cuty):
        """
        调整图片相对偏移
        :param bboxes:
        :param cutx:
        :param cuty:
        :return:
        """
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def cvtColor(self, image):
        """
        通道转换
        :param image:
        :return:
        """
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image
        else:
            image = image.convert('RGB')
            return image

    def imread(self, path):
        """
        读取图片
        :param path: 读取地址
        :return:
        """
        # 读取图像并转换成RGB图像
        image = Image.open(path)
        image = self.cvtColor(image)

        return image

    def getBoxes(self, line):
        """
        获得框
        :param line: 每一行
        :return:
        """
        box = np.array([
            np.array(list(map(int, box.split(',')))) for box in line[1:]])
        return box


    def rand(self, a=0., b=1.):
        """
        随机数
        :param a:起始范围
        :param b:终止范围
        :return:
        """
        return np.random.rand()*(b-a) + a

if __name__ == "__main__":

    line1 = "E:\Project\Python\datas\VOCdevkit/VOC2012/JPEGImages/2009_004091.jpg 76,101,376,333,17 339,71,422,207,8"
    line2 = "E:\Project\Python\datas\VOCdevkit/VOC2012/JPEGImages/2008_004372.jpg 232,105,276,172,8 230,96,377,267,14 35,102,169,267,14 147,67,260,183,14 143,196,168,253,4"
    line3 = "E:\Project\Python\datas\VOCdevkit/VOC2012/JPEGImages/2011_004008.jpg 179,62,346,500,14"

    read = DataRead((640, 640))

    image, boxes = read.getMosaicData([line1, line2, line1, line3])

    image = image.astype("uint8")

    for box in boxes:

        x1, y1, x2, y2, _ = box

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))

    cv2.imshow("image", image)
    cv2.waitKey(0)

    # image, boxes = read.getData(line1, True)
    #
    # image = image.astype("uint8")
    #
    # for box in boxes:
    #
    #     x1, y1, x2, y2, _ = box
    #     cv2.rectangle(image, (x1, y1),(x2, y2),(0,0,255))
    #
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
