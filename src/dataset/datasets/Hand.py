from skimage import io
import os
import numpy as np
import cv2
import hashlib
import torch
from itertools import repeat
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import mediapipe as mp
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageOps
import random
import math
from ..utils import cvtColor, preprocess_input, verify_image_label
from multiprocessing.pool import ThreadPool
from tqdm import tqdm as tqdm_original

DATASET_CACHE_VERSION = '1.0.3'

def get_hash(paths):
    """对一组路径（文件或目录）计算统一哈希值。"""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # 汇总文件大小
    h = hashlib.sha256(str(size).encode())  # 先对大小做哈希
    h.update(''.join(paths).encode())  # 再加入路径字符串
    return h.hexdigest()  # 返回哈希结果

class TQDM(tqdm_original):
    """
    Ultralytics 的 tqdm 封装，提供不同的默认参数。

    Args:
        *args (list): 传给原始 tqdm 的位置参数。
        **kwargs (dict): 关键字参数，会应用自定义默认值。
    """

    def __init__(self, *args, **kwargs):
        VERBOSE = str(os.getenv('YOLO_VERBOSE', True)).lower() == 'true'  # 全局详细输出开关
        TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}' if VERBOSE else None  # 进度条格式
        """初始化自定义 tqdm 默认参数。"""
        # 设置默认值（调用时仍可覆盖）
        kwargs['disable'] = not VERBOSE or kwargs.get('disable', False)  # 若外部显式传入则优先使用外部值
        kwargs.setdefault('bar_format', TQDM_BAR_FORMAT)  # 未传入时使用默认格式
        super().__init__(*args, **kwargs)

def is_dir_writeable(dir_path):
    """
    检查目录是否可写。

    Args:
        dir_path (str | Path): 目录路径。

    Returns:
        (bool): 可写返回 True，否则返回 False。
    """
    return os.access(str(dir_path), os.W_OK)

def save_dataset_cache_file(prefix, path, x):
    """将数据集缓存字典 x 保存到 *.cache 文件。"""
    x['version'] = DATASET_CACHE_VERSION  # 写入缓存版本
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # 若已存在则先删除
        np.save(str(path), x)  # 保存缓存供下次使用
        path.with_suffix('.cache.npy').rename(path)  # 去掉 .npy 后缀

class Hand(Dataset):
    """
    手部数据集读取类。

    该类可加载本项目定义的图像、分割标签和关键点标签。
    """

    def __init__(self, phase, transform=None, opt=None):
        self.dir_dataset = '/data/Hand_dataset/dataset'
        # 设定固定随机种子，保证可复现
        np.random.seed(42)
        self.phase = phase
        # self.imgdir = os.path.join(self.dir_dataset, 'img256_' + phase + '_new')
        self.imgdir = os.path.join(self.dir_dataset, 'images/', phase)
        self.labeldir = os.path.join(self.dir_dataset, 'labels/' + phase)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.imglist = sorted(os.listdir(self.imgdir))
        self.segm_label_list = []
        self.keypoint_label_list = []
        self.prefix = ''
        for i in range(len(self.imglist)):
            self.segm_label_list.append(os.path.join(self.labeldir, 'segm', self.imglist[i]))
            self.keypoint_label_list.append(os.path.join(self.labeldir, 'keypoint', self.imglist[i][:-4] + '.txt'))
            self.imglist[i] = os.path.join(self.imgdir, self.imglist[i])
        # self.transform = transform
        self.opt = opt
        self.input_shape = self.opt.image_size
        self.num_classes = self.opt.num_classes
        self.class_names = [['daling','yuji','shaoshang','xiaochang','dachang',
                             'laogong','sanjiao','xinxue','zhongchong','ganxue',
                             'feixue','shaofu','mingmen','shenxue','taiyuan','shenmen']]

        # 手部关键点检测模型
        mp_hand = mp.solutions.hands
        # 导入模型
        self.hands = mp_hand.Hands(static_image_mode=False,
                                  max_num_hands=1,
                                  min_detection_confidence=0.3,
                                  min_tracking_confidence=0.3
                                  )

    def __getitem__(self, index):
        """返回单条样本及其标签。

        Parameters:
            index (int): 样本索引。

        Returns:
            img: 输入图像张量。
            acupoint: 先验关键点。
            segm: 分割标签。
            seg_labels: one-hot 分割标签。
            keypoint_labels: 关键点真值。
        """

        img_path = self.imglist[index]
        segm_label_path = self.segm_label_list[index]
        keypoint_label_path = self.keypoint_label_list[index]

        f = open(keypoint_label_path)
        line = f.readline().strip()
        keypoint_labels = np.array(list(map(float, line.split(' ')[5:]))).reshape(16, 3)

        # img = Image.open(img_path)
        segm = Image.open(segm_label_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # segm = cv2.imread(segm_label_path)
        # img = np.uint8(img)
        # result = Image.fromarray(img).convert('RGB')
        # plt.imshow(img)
        # plt.show()

        img, segm = self.get_random_data(img, segm, [self.input_shape, self.input_shape], random=(self.phase == 'train'))

        img = np.transpose(preprocess_input(np.array(img, np.float64)), [2, 0, 1])

        segm = np.array(segm).astype(np.int64)
        segm[segm >= self.num_classes] = self.num_classes
        segm[segm == 0] = -1

        # -------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        # -------------------------------------------------------#
        seg_labels = np.eye(self.num_classes + 1)[segm.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape), int(self.input_shape), self.num_classes + 1))

        # """传统模型先验知识处理"""
        acupoint = self.get_Key_point_prior_knowledge(img_path, keypoint_labels)

        return img, acupoint, segm, seg_labels, keypoint_labels

    def __len__(self):
        """返回数据集样本总数。"""
        return len(self.imglist)

    def get_Key_point_prior_knowledge(self, img_path, keypoint_labels):
        lmList = []
        acupoint = np.zeros_like(keypoint_labels)
        img_RGB = cv2.imread(img_path)
        img_RGB = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2RGB)
        h, w, c = img_RGB.shape
        results = self.hands.process(img_RGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img_RGB.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

        if len(lmList) != 21:
            acupoint = keypoint_labels
            return acupoint
        if lmList[12][2] < lmList[0][2]:
            """大陵"""
            cx0, cy0 = lmList[0][1], lmList[0][2]
            acupoint[0, 0] = cx0
            acupoint[0, 1] = cy0
            """鱼际"""
            cx1, cy1 = lmList[1][1], lmList[1][2]
            acupoint[1, 0] = cx1
            acupoint[1, 1] = cy1
            """太渊"""
            cx2, cy2 = lmList[2][1], lmList[2][2]
            cx2 = int(cx1 + (cx1 - cx2) / 2.0)
            cy2 = int(cy1 + (cy1 - cy2) / 2.0)
            acupoint[14, 0] = cx2
            acupoint[14, 1] = cy2
            """神门"""
            cx_sm = int(cx0 + (cx0 - cx2))
            cy_sm = int(cy0 + (cy0 - cy2))
            acupoint[15, 0] = cx_sm
            acupoint[15, 1] = cy_sm
            """少商"""
            cx3, cy3 = lmList[3][1], lmList[3][2]
            acupoint[2, 0] = cx3 - 50
            acupoint[2, 1] = cy3 + 50
            """小肠"""
            cx6, cy6 = lmList[6][1], lmList[6][2]
            acupoint[3, 0] = cx6
            acupoint[3, 1] = cy6
            """大肠"""
            cx7, cy7 = lmList[7][1], lmList[7][2]
            acupoint[4, 0] = cx7
            acupoint[4, 1] = cy7
            """三焦"""
            cx10, cy10 = lmList[10][1], lmList[10][2]
            acupoint[6, 0] = cx10
            acupoint[6, 1] = cy10
            """劳宫"""
            cx9, cy9 = lmList[9][1], lmList[9][2]
            cx9 = int(cx9 + (cx9 - cx10))
            cy9 = int(cy9 + (cy9 - cy10))
            acupoint[5, 0] = cx9
            acupoint[5, 1] = cy9
            """心穴"""
            cx11, cy11 = lmList[11][1], lmList[11][2]
            acupoint[7, 0] = cx11
            acupoint[7, 1] = cy11
            """中冲"""
            cx12, cy12 = lmList[12][1], lmList[12][2]
            cx12 = (2 * cx12 + cx11) // 3
            cy12 = (2 * cy12 + cy11 + 50) // 3
            acupoint[8, 0] = cx12
            acupoint[8, 1] = cy12
            """肝穴"""
            cx14, cy14 = lmList[14][1], lmList[14][2]
            acupoint[9, 0] = cx14
            acupoint[9, 1] = cy14
            """肺穴"""
            cx15, cy15 = lmList[15][1], lmList[15][2]
            acupoint[10, 0] = cx15
            acupoint[10, 1] = cy15
            """命门"""
            cx18, cy18 = lmList[18][1], lmList[18][2]
            acupoint[12, 0] = cx18
            acupoint[12, 1] = cy18
            """少府"""
            cx17, cy17 = lmList[17][1], lmList[17][2]
            cx17 = int(cx17 + (cx17 - cx18))
            cy17 = int(cy17 + (cy17 - cy18))
            acupoint[11, 0] = cx17
            acupoint[11, 1] = cy17
            """肾穴"""
            cx19, cy19 = lmList[19][1], lmList[19][2]
            acupoint[13, 0] = cx19
            acupoint[13, 1] = cy19

            acupoint[:, 0] = acupoint[:, 0] / w
            acupoint[:, 1] = acupoint[:, 1] / h
            acupoint[:, 2] = keypoint_labels[:, 2]
            acupoint[keypoint_labels[:, 2] == 0] = 0

        return acupoint


    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        # iw, ih = image.size
        h, w = input_shape

        if not random:
            # iw, ih = image.size
            # scale = min(w / iw, h / ih)
            # nw = int(iw * scale)
            # nh = int(ih * scale)

            # image = image.resize((w, h), Image.BICUBIC)
            # # new_image = Image.new('RGB', [w, h], (128, 128, 128))
            # # new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
            #
            # label = label.resize((w, h), Image.NEAREST)
            # # new_label = Image.new('L', [w, h], (0))
            # # new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            image = cv2.resize(image, [w, h], interpolation=cv2.INTER_CUBIC)
            label = label.resize((w, h), Image.NEAREST)
            return image, label

        # # ------------------------------------------#
        # #   对图像进行缩放并且进行长和宽的扭曲
        # # ------------------------------------------#
        # new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        # scale = self.rand(0.25, 2)
        # if new_ar < 1:
        #     nh = int(scale * h)
        #     nw = int(nh * new_ar)
        # else:
        #     nw = int(scale * w)
        #     nh = int(nw / new_ar)
        # image = image.resize((nw, nh), Image.BICUBIC)
        # label = label.resize((nw, nh), Image.NEAREST)


        image = cv2.resize(image, [w, h], interpolation=cv2.INTER_CUBIC)
        # label = cv2.resize(label, [w, h], interpolation=cv2.INTER_NEAREST)
        label = label.resize((w, h), Image.NEAREST)
        #
        # # ------------------------------------------#
        # #   翻转图像
        # # ------------------------------------------#
        # flip = self.rand() < .5
        # if flip:
        #     image = image.transpose(Image.FLIP_LEFT_RIGHT)
        #     label = label.transpose(Image.FLIP_LEFT_RIGHT)
        #
        # # ------------------------------------------#
        # #   将图像多余的部分加上灰条
        # # ------------------------------------------#
        # dx = int(self.rand(0, w - nw))
        # dy = int(self.rand(0, h - nh))
        # new_image = Image.new('RGB', (w, h), (128, 128, 128))
        # new_label = Image.new('L', (w, h), (0))
        # new_image.paste(image, (dx, dy))
        # new_label.paste(label, (dx, dy))
        # image = new_image
        # label = new_label

        image_data = np.array(image, np.uint8)

        # ------------------------------------------#
        #   高斯模糊
        # ------------------------------------------#
        blur = self.rand() < 0.25
        if blur:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        # # ------------------------------------------#
        # #   旋转
        # # ------------------------------------------#
        # rotate = self.rand() < 0.25
        # if rotate:
        #     center = (w // 2, h // 2)
        #     rotation = np.random.randint(-10, 11)
        #     M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
        #     image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
        #     label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label

    def maxmin_norm(self, data):
        data = (data - data.min()) / (data.max() - data.min()) * 255
        return data.astype('uint8')
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    # transforms = CustomDataAugmentation(256, 0.08)
    train_dataset = Luojiassr('val')
    # img_path = '/data/HyperSpectralNet/luojiassr/img256_train_new/13_obj_0_1__0_1_.tif'
    # label_path = '/data/HyperSpectralNet/luojiassr/label256_train_new/1_obj_0_0__0_0__0_1_.tif'
    # im_width, im_height, im_bands, im_proj, im_geotrans, im_data = train_dataset.read_img(img_path)
    # l_width, l_height, l_bands, l_proj, l_geotrans, l_data = dataset.read_img(label_path)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    new = torch.zeros(1, dtype=torch.int64)
    for im_data, l_data in train_loader:
        new = torch.cat([new, torch.unique(l_data)])
        new = torch.unique(new)

    print('OK')
    print(new)


