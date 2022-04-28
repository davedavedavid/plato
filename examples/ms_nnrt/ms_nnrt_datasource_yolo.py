import logging
import os
import math
from plato.config import Config
from plato.datasources import base
from examples.ms_nnrt.ms_nnrt_datasource_yolo_utils import COCOYoloDataset


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
        #print('Config().data.train_annFile, train_path:',Config().data.train_annFile, Config().data.train_path)
        if self.train_set is None:
            self.train_set = COCOYoloDataset(
                root=Config().data.train_path,
                ann_file=Config().data.train_annFile,
                filter_crowd_anno=True,
                remove_images_without_annotations=True,
                is_training=True)
            for image1, annotation, input_size, mosaic_flag in self.trainset:
                print('image1: ', image1, image1.shape, flush=True)
        return self.train_set

    def get_test_set(self):

        if self.test_set is None:
            self.test_set = COCOYoloDataset(
                root=Config().data.test_path,
                ann_file=Config().data.test_annFile,
                filter_crowd_anno=False,
                remove_images_without_annotations=False,
                is_training=False)

        return self.test_set