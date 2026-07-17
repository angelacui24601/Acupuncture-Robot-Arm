import json
import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset

from ..utils import preprocess_input


class CervicoCranial(Dataset):
    """AcuSim/CervicoCranial dataset adapter with Hand-compatible outputs."""

    def __init__(self, phase, transform=None, opt=None):
        self.phase = phase
        self.opt = opt
        self.transform = transform
        self.input_shape = int(self.opt.image_size)
        self.num_classes = int(self.opt.num_classes)

        self.dataset_root = os.path.abspath(self.opt.cervico_dataset_root)
        self.test_image_dir = os.path.abspath(self.opt.cervico_test_image_dir)
        self.test_label_dir = os.path.abspath(self.opt.cervico_test_label_dir)
        self.image_subdir = self.opt.cervico_image_subdir

        map_file = self.opt.cervico_map_file.strip()
        if not map_file:
            map_file = os.path.join(self.dataset_root, 'map.txt')
        self.map_file = os.path.abspath(map_file)

        self.available_point_names = self._read_point_names(self.map_file)
        self.target_point_names = self._resolve_target_points(self.available_point_names)

        self.imgdir, self.labeldir = self._resolve_split_dirs(phase)
        self.samples, strict_pairs = self._build_samples(self.imgdir, self.labeldir)

        if phase != 'train' and not strict_pairs:
            train_imgdir = os.path.join(self.dataset_root, 'train', 'image', self.image_subdir)
            train_labeldir = os.path.join(self.dataset_root, 'train', 'label', 'label')
            train_samples, train_strict = self._build_samples(train_imgdir, train_labeldir)
            if not train_strict or len(train_samples) < 2:
                raise RuntimeError(
                    'Validation split has no stem-matched pairs and fallback train holdout could not be built.'
                )
            holdout_start = max(1, int(0.9 * len(train_samples)))
            self.samples = train_samples[holdout_start:]

        if len(self.samples) == 0:
            raise RuntimeError(
                f'No matched image/label pairs found for phase={phase}. '
                f'image_dir={self.imgdir}, label_dir={self.labeldir}'
            )

    def _read_point_names(self, map_file):
        if not os.path.isfile(map_file):
            raise FileNotFoundError(f'map.txt not found: {map_file}')

        names = []
        with open(map_file, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name:
                    names.append(name)
        if not names:
            raise RuntimeError(f'map.txt is empty: {map_file}')
        return names

    def _resolve_target_points(self, all_point_names):
        configured = self.opt.cervico_keypoints.strip()
        if configured:
            points = [x.strip() for x in configured.split(',') if x.strip()]
            if len(points) != 16:
                raise ValueError(
                    'cervico_keypoints must contain exactly 16 names to match the current model output shape.'
                )
            return points
        return all_point_names[:16]

    def _resolve_split_dirs(self, phase):
        if phase == 'train':
            imgdir = os.path.join(self.dataset_root, 'train', 'image', self.image_subdir)
            labeldir = os.path.join(self.dataset_root, 'train', 'label', 'label')
        else:
            candidate_imgdir = os.path.join(self.dataset_root, phase, 'image', self.image_subdir)
            candidate_labeldir = os.path.join(self.dataset_root, phase, 'label', 'label')
            if os.path.isdir(candidate_imgdir) and os.path.isdir(candidate_labeldir):
                imgdir, labeldir = candidate_imgdir, candidate_labeldir
            elif os.path.isdir(self.test_image_dir) and os.path.isdir(self.test_label_dir):
                imgdir, labeldir = self.test_image_dir, self.test_label_dir
            else:
                imgdir = os.path.join(self.dataset_root, 'train', 'image', self.image_subdir)
                labeldir = os.path.join(self.dataset_root, 'train', 'label', 'label')

        if not os.path.isdir(imgdir):
            raise FileNotFoundError(f'Image directory not found: {imgdir}')
        if not os.path.isdir(labeldir):
            raise FileNotFoundError(f'Label directory not found: {labeldir}')
        return imgdir, labeldir

    def _build_samples(self, imgdir, labeldir):
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        image_map = {}
        for filename in sorted(os.listdir(imgdir)):
            stem, ext = os.path.splitext(filename)
            if ext.lower() in image_exts:
                path = os.path.join(imgdir, filename)
                image_map[stem] = path

        label_map = {}
        for filename in sorted(os.listdir(labeldir)):
            stem, ext = os.path.splitext(filename)
            if ext.lower() == '.json':
                path = os.path.join(labeldir, filename)
                label_map[stem] = path

        shared_stems = sorted(set(image_map).intersection(label_map))
        samples = [(image_map[s], label_map[s], s) for s in shared_stems]
        if samples:
            return samples, True

        # No reliable filename match between images and labels in this split.
        return [], False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label_path, _ = self.samples[index]

        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f'Failed to read image: {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        keypoint_labels = self._parse_keypoint_json(label_path)

        img = self._resize_and_augment(img, random_aug=(self.phase == 'train'))
        img = np.transpose(preprocess_input(np.array(img, np.float64)), [2, 0, 1])

        segm = np.zeros((self.input_shape, self.input_shape), dtype=np.int64)
        seg_labels = np.eye(self.num_classes + 1, dtype=np.float32)[segm.reshape([-1])]
        seg_labels = seg_labels.reshape((self.input_shape, self.input_shape, self.num_classes + 1))

        acupoint = keypoint_labels.copy()

        return img, acupoint, segm, seg_labels, keypoint_labels

    def _parse_keypoint_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        labels = data.get('label', [])
        name_to_coord = {}
        for item in labels:
            name = item.get('name', '')
            coord = item.get('coordinate', {})
            if not name:
                continue
            x = float(coord.get('x', 0.0))
            y = float(coord.get('y', 0.0))
            z = float(coord.get('h', 0.0))
            name_to_coord[name] = (
                np.clip(x, 0.0, 1.0),
                np.clip(y, 0.0, 1.0),
                np.clip(z, 0.0, 1.0),
            )

        keypoints = np.zeros((16, 3), dtype=np.float32)
        for i, name in enumerate(self.target_point_names):
            if name in name_to_coord:
                keypoints[i] = name_to_coord[name]
        return keypoints

    def _resize_and_augment(self, image, random_aug):
        image_data = cv2.resize(image, [self.input_shape, self.input_shape], interpolation=cv2.INTER_CUBIC)

        if not random_aug:
            return image_data

        if random.random() < 0.25:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        hue = 0.1
        sat = 0.7
        val = 0.3
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        h, s, v = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(h, lut_hue), cv2.LUT(s, lut_sat), cv2.LUT(v, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data
