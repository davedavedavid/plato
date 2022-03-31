import logging
import time
import cv2
import random

import numpy as np
from examples.ms_nnrt.ms_nnrt_algorithms import ms_fedavg
from plato.config import Config
from plato.utils import unary_encoding
from examples.ms_nnrt.ms_nnrt_datasource_yolo_utils import DistributedSampler, MultiScaleTrans, PreprocessTrueBox
from examples.ms_nnrt.config import ConfigYOLOV5

class Algorithm(ms_fedavg.Algorithm):
    """The NNRT-based MistNet algorithm, used by both the client and the
    server.
    """
    def extract_features(self, dataset, sampler, cut_layer: str, epsilon=None):
        """Extracting features using layers before the cut_layer.
        dataset: The training or testing dataset. This datasets does not based on
                 torch.utils.data.Datasets
        cut_layer: Layers before this one will be used for extracting features.
                TODO: This cannot be changed dynamically due to the static properties of OM file.
        epsilon: If epsilon is not None, local differential privacy should be
                applied to the features extracted.
        """

        tic = time.perf_counter()

        feature_dataset = []

        _randomize = getattr(self.trainer, "randomize", None)

        features_shape = self.features_shape()

        config = ConfigYOLOV5()
        device_num = 1
        distributed_sampler = DistributedSampler(len(dataset), device_num, rank=None, shuffle=True)
        dataset.size = len(distributed_sampler)
        config.dataset_size = len(dataset)
        #cores = multiprocessing.cpu_count()
        #num_parallel_workers = int(cores / device_num)
        multi_scale_trans = MultiScaleTrans(config, device_num)
        dataset.transforms = multi_scale_trans

        def concatenate(images):
            images = np.concatenate((images[..., ::2, ::2], images[..., 1::2, ::2],
                                     images[..., ::2, 1::2], images[..., 1::2, 1::2]), axis=0)
            return images

        #for i in range(5):
        #for inputs, targets, *__ in dataset:
        for img, anno, input_size, mosaic_flag in dataset:
            image, annotation, size = multi_scale_trans(img=img, anno=anno, input_size=input_size, mosaic_flag=mosaic_flag)
            annotation, bbox1, bbox2, bbox3, gt_box1, gt_box2, gt_box3 = PreprocessTrueBox(annotation, size)
            mean = [m * 255 for m in [0.485, 0.456, 0.406]]
            std = [s * 255 for s in [0.229, 0.224, 0.225]]
            image = (image - mean) /std
            image = image.swapaxes(1,2).swapaxes(0,1)       #   HWC->HCW->CHW    CV.HWC2CHW  or images.transpose((2,0,1))
            ds = concatenate(image)

            inputs = ds.astype(np.float32)
            #inputs = inputs / 255.0  # normalize image and convert image type at the same time
                # inputs = np.expand_dims(inputs, axis=0)
                # np.save("/home/data/model/test_image.npy", inputs)

            logits = self.model.foward(inputs)
                # logits = inputs

            logits = np.reshape(logits, features_shape)
                # np.save("/home/data/model/test_feat.npy", logits)
            targets = np.expand_dims(
                    targets, axis=0
                )  # add batch axis to make sure self.train.randomize correct
                #count += 1
                #logging.info("[Client #%d] Extracting %d features from %s examples.",
                #         self.client_id, count, len(dataset))
            if epsilon is not None:
                logging.info("epsilon is %d.",epsilon)
                logits = unary_encoding.encode(logits)
                if callable(_randomize):
                    logits = self.trainer.randomize(logits, targets, epsilon)
                else:
                    logits = unary_encoding.randomize(logits, epsilon)
                    # Pytorch is currently not supported on A500 and we cannot convert
                    # numpy array to tensor
                if self.trainer.device != 'cpu':
                    logits = logits.astype('float16')
                else:
                    logits = logits.astype('float32')

            for i in np.arange(logits.shape[0]):  # each sample in the batch
                feature_dataset.append((logits[i], targets[i]))
                # feature_dataset.append((inputs, targets))

        toc = time.perf_counter()
        logging.info("[Client #%d] Features extracted from %s examples.",
                     self.client_id, len(feature_dataset))
        logging.info("[Client #{}] Time used: {:.2f} seconds.".format(
            self.client_id, toc - tic))

        return feature_dataset

    def features_shape(self):
        """ Return the features shape of the cutlayer output. """
        # TODO: Do not hard code the features shape
        #return [-1, 320, 184, 184]
        return [-1, 320, 120, 120]