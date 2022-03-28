import os
import multiprocessing
import random
import numpy as np
import cv2
from PIL import Image
from pycocotools.coco import COCO


def has_valid_annotation(anno):
    """Check annotation file."""
    # if it's empty, there is no annotation
    if not anno:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different criteria for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCOYoloDataset:
    """YOLOV5 Dataset for COCO."""
    def __init__(self, root, ann_file, remove_images_without_annotations=True,
                 filter_crowd_anno=True, is_training=True):
        self.coco = COCO(ann_file)
        self.root = root
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        self.filter_crowd_anno = filter_crowd_anno
        self.is_training = is_training
        self.mosaic = True
        # filter images without any annotations
        if remove_images_without_annotations:
            img_ids = []
            for img_id in self.img_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    img_ids.append(img_id)
            self.img_ids = img_ids

        self.categories = {cat["id"]: cat["name"] for cat in self.coco.cats.values()}

        self.cat_ids_to_continuous_ids = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.continuous_ids_cat_ids = {
            v: k for k, v in self.cat_ids_to_continuous_ids.items()
        }
        self.count = 0

    def _mosaic_preprocess(self, index, input_size):
        labels4 = []
        s = 384
        self.mosaic_border = [-s // 2, -s // 2]
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]
        indices = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        for i, img_ids_index in enumerate(indices):
            coco = self.coco
            img_id = self.img_ids[img_ids_index]
            img_path = coco.loadImgs(img_id)[0]["file_name"]
            img = Image.open(os.path.join(self.root, img_path)).convert("RGB")
            img = np.array(img)
            h, w = img.shape[:2]

            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 128, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

            padw = x1a - x1b
            padh = y1a - y1b

            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)
            # filter crowd annotations
            if self.filter_crowd_anno:
                annos = [anno for anno in target if anno["iscrowd"] == 0]
            else:
                annos = [anno for anno in target]

            target = {}
            boxes = [anno["bbox"] for anno in annos]
            target["bboxes"] = boxes

            classes = [anno["category_id"] for anno in annos]
            classes = [self.cat_ids_to_continuous_ids[cl] for cl in classes]
            target["labels"] = classes

            bboxes = target['bboxes']
            labels = target['labels']
            out_target = []

            for bbox, label in zip(bboxes, labels):
                tmp = []
                # convert to [x_min y_min x_max y_max]
                bbox = self._convetTopDown(bbox)
                tmp.extend(bbox)
                tmp.append(int(label))
                # tmp [x_min y_min x_max y_max, label]
                out_target.append(tmp)  # 杩欓噷out_target鏄痩abel鐨勫疄闄呭楂橈紝瀵瑰簲浜庡浘鐗囦腑鐨勫疄闄呭害閲?
            labels = out_target.copy()
            labels = np.array(labels)
            out_target = np.array(out_target)

            labels[:, 0] = out_target[:, 0] + padw
            labels[:, 1] = out_target[:, 1] + padh
            labels[:, 2] = out_target[:, 2] + padw
            labels[:, 3] = out_target[:, 3] + padh
            labels4.append(labels)

        if labels4:
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, :4], 0, 2 * s, out=labels4[:, :4])  # use with random_perspective
        flag = np.array([1])
        return img4, labels4, input_size, flag

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            (img, target) (tuple): target is a dictionary contains "bbox", "segmentation" or "keypoints",
                generated by the image's annotation. img is a PIL image.
        """
        coco = self.coco
        img_id = self.img_ids[index]
        img_path = coco.loadImgs(img_id)[0]["file_name"]
        if not self.is_training:
            img = Image.open(os.path.join(self.root, img_path)).convert("RGB")
            return img, img_id

        input_size = [640, 640]
        if self.mosaic and random.random() < 0.5:
            return self._mosaic_preprocess(index, input_size)
        img = np.fromfile(os.path.join(self.root, img_path), dtype='int8')
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        # filter crowd annotations
        if self.filter_crowd_anno:
            annos = [anno for anno in target if anno["iscrowd"] == 0]
        else:
            annos = [anno for anno in target]

        target = {}
        boxes = [anno["bbox"] for anno in annos]
        target["bboxes"] = boxes

        classes = [anno["category_id"] for anno in annos]
        classes = [self.cat_ids_to_continuous_ids[cl] for cl in classes]
        target["labels"] = classes

        bboxes = target['bboxes']
        labels = target['labels']
        out_target = []
        for bbox, label in zip(bboxes, labels):
            tmp = []
            # convert to [x_min y_min x_max y_max]
            bbox = self._convetTopDown(bbox)
            tmp.extend(bbox)
            tmp.append(int(label))
            # tmp [x_min y_min x_max y_max, label]
            out_target.append(tmp)
        flag = np.array([0])
        return img, out_target, None, input_size

    def __len__(self):
        return len(self.img_ids)

    def _convetTopDown(self, bbox):
        x_min = bbox[0]
        y_min = bbox[1]
        w = bbox[2]
        h = bbox[3]
        return [x_min, y_min, x_min+w, y_min+h]
