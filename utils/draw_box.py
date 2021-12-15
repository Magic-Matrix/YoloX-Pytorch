import random
import cv2
import numpy as np

class DrawBox(object):
    def __init__(self, class_name, seed=0):

        # 随机数种子
        random.seed(seed)

        # 分类个数
        self.class_number = len(class_name)

        # 分类的名字
        self.class_name = class_name

        # 分类的颜色
        self.class_color = [self.randColor() for _ in range(self.class_number)]

    def __call__(self, image, outputs):

        image = image.copy()

        for output in outputs:
            x1, y1, x2, y2 = output["box"]
            score = output["score"]
            label = output["class"]
            name = self.class_name[label]
            color = self.class_color[label]

            txt = str(score)[:4] + " " + name

            w, h = x2 - x1, y2 - y1

            x3, y3 = x1 + w, y1 + 20

            # 区域颜色
            part = np.ones((20, w, 3)) * np.array(list(color), dtype="uint8")

            image[y1:y3, x1:x3] = part

            cv2.putText(image, txt, (x1, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.rectangle(image, (x1, y1), (x2, y2), color)

        return image





            # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))





    def randColor(self):
        """
        随机一个颜色
        :return:
        """

        return tuple([random.randint(0, 255) for _ in range(3)])


