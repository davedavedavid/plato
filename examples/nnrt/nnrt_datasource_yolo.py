"""
The COCO dataset or other datasets for the YOLOv5 model with using NNRT.
"""

import logging
import os
import math
from plato.config import Config
from plato.datasources import base
from examples.nnrt.nnrt_datasource_yolo_utils import LoadImagesAndLabels

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print(
            'WARNING: --img-size %g must be multiple of max stride %g, updating to %g'
            % (img_size, s, new_size))
    return new_size


class DataSource(base.DataSource):
    """The YOLO dataset."""
    def __init__(self):
        super().__init__()
        _path = Config().data.data_path

        if not os.path.exists(_path):
            os.makedirs(_path)

            logging.info(
                "Downloading the YOLO dataset. This may take a while.")

            urls = Config().data.download_urls
            for url in urls:
                if not os.path.exists(_path + url.split('/')[-1]):
                    DataSource.download(url, _path)

        assert 'grid_size' in Config().params

        self.grid_size = Config().params['grid_size']
        self.image_size = check_img_size(Config().data.image_size,
                                         self.grid_size)

        self.train_set = None
        self.test_set = None

    def num_train_examples(self):
        return Config().data.num_train_examples

    def num_test_examples(self):
        return Config().data.num_test_examples

    def classes(self):
        """Obtains a list of class names in the dataset."""
        return Config().data.classes

    def get_train_set(self):
        single_class = (Config().data.num_classes == 1)

        hpy = {'optimizer': 'SGD',  # ['adam', 'SGD', None] if none, default is SGD
            'momentum': 0.937,  # SGD momentum/Adam beta1
            'weight_decay': 5e-4,  # optimizer weight decay
            'giou': 0.05,  # giou loss gain
            'cls': 0.5,  # cls loss gain
            'cls_pw': 1.0,  # cls BCELoss positive_weight
            'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
            'obj_pw': 1.0,  # obj BCELoss positive_weight
            'iou_t': 0.20,  # iou training threshold
            'anchor_t': 4.0,  # anchor-multiple threshold
            'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
            'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
            'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
            'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
            'degrees': 0.0,  # image rotation (+/- deg)
            'translate': 0.0,  # image translation (+/- fraction)
            'scale': 0.5,  # image scale (+/- gain)
            'shear': 0.0}

        if self.train_set is None:
            self.train_set = LoadImagesAndLabels(
                Config().data.train_path,
                self.image_size,
                Config().trainer.batch_size,
                augment=True,  # augment images
                hyp=hpy,  # augmentation hyperparameters
                rect=False,  # rectangular training
                cache_images=False,
                single_cls=single_class,
                stride=int(self.grid_size),
                pad=0.0,
                image_weights=False)

        return self.train_set

    def get_test_set(self):
        single_class = (Config().data.num_classes == 1)

        if self.test_set is None:
            self.test_set = LoadImagesAndLabels(
                Config().data.test_path,
                self.image_size,
                Config().trainer.batch_size,
                augment=False,  # augment images
                hyp=None,  # augmentation hyperparameters
                rect=False,  # rectangular training
                cache_images=False,
                single_cls=single_class,
                stride=int(self.grid_size),
                pad=0.0,
                image_weights=False)

        return self.test_set
