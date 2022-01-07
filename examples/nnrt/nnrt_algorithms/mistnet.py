"""
The NNRT-based MistNet algorithm, used by both the client and the server.

Reference:

P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local
Differential Privacy," found in docs/papers.
"""

import logging
import time
import cv2
import random

import numpy as np
from examples.nnrt.nnrt_algorithms import fedavg
from plato.config import Config
from plato.utils import unary_encoding


class Algorithm(fedavg.Algorithm):
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

        for i in range(10):
            for inputs, targets, *__ in dataset:
                # assert inputs.shape[1] == Config().data.input_height and inputs.shape[2] == Config().data.input_width, \
                #     "The input shape is not consistent with the requirement predefined model."
                # print("Load image is ", inputs, flush=True)
                # print("Load targets is ", targets, flush=True)
                # print("Random numpy number is ", np.random.randn(), flush=True)
                # print("Random randn number is ", random.random(), flush=True)
                # inputs = cv2.imread("/home/data/model/image1.jpg")
                # inputs = np.moveaxis(inputs, -1, 0)
                # inputs = np.reshape(inputs, (1, -1))
                inputs = inputs.astype(np.float32)
                inputs = inputs / 255.0  # normalize image and convert image type at the same time
                # np.save("/home/data/model/test_image.npy", inputs)
                logits = self.model.forward(inputs)
                logits = np.reshape(logits, features_shape)
                # np.save("/home/data/model/test_feat.npy", logits)
                targets = np.expand_dims(
                    targets, axis=0
                )  # add batch axis to make sure self.train.randomize correct
                # count += 1
                # logging.info("[Client #%d] Extracting %d features from %s examples.",
                #          self.client_id, count, len(dataset))
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

        toc = time.perf_counter()
        logging.info("[Client #%d] Features extracted from %s examples.",
                     self.client_id, len(feature_dataset))
        logging.info("[Client #{}] Time used: {:.2f} seconds.".format(
            self.client_id, toc - tic))

        return feature_dataset

    def features_shape(self):
        """ Return the features shape of the cutlayer output. """
        # TODO: Do not hard code the features shape
        return [-1, 320, 80, 80]
