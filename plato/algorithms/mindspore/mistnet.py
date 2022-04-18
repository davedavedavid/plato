"""
The federated learning trainer for MistNet, used by both the client and the
server.

Reference:

P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local
Differential Privacy," found in docs/papers.
"""

import time
import logging
import mindspore
import mindspore.dataset as ds

from plato.utils import unary_encoding
from plato.algorithms.mindspore import fedavg


class Algorithm(fedavg.Algorithm):
    """The PyTorch-based MistNet algorithm, used by both the client and the
    server.
    """
    def extract_features(self, dataset, cut_layer, epsilon=None):
        """Extracting features using layers before the cut_layer.

        dataset: The training or testing dataset.
        cut_layer: Layers before this one will be used for extracting features.
        epsilon: If epsilon is not None, local differential privacy should be
                applied to the features extracted.
        """
        self.model.set_train(False)

        tic = time.perf_counter()

        feature_dataset = []

        for inputs, targets in dataset:
            inputs = mindspore.Tensor(inputs)
            targets = mindspore.Tensor(targets)

            logits = self.model.forward_to(inputs, cut_layer)

            if epsilon is not None:
                logits = logits.asnumpy()
                logits = unary_encoding.encode(logits)
                logits = unary_encoding.randomize(logits, epsilon)
                logits = mindspore.Tensor(logits.astype('float32'))

            feature_dataset.append((logits, targets))

        toc = time.perf_counter()
        logging.info("[Client #%d] Features extracted from %s examples.",
                     self.client_id, len(feature_dataset))
        logging.info("[Client #{}] Time used: {:.2f} seconds.".format(
            self.client_id, toc - tic))

        return feature_dataset

    # @staticmethod
    # class SddDataset(trainset):
    #     def __init__(self, trainset) -> None:
    #         self.labels = trainset[1]
    #         self.images = trainset[0]
    #         self.index = 0
    #
    #     def __len__(self):
    #         return self
    #
    #     def __iter__(self):
    #         return self
    #
    #     def __next__(self):
    #         x, y = self.images, self.labels
    #         return x, y
    @staticmethod
    def dataset_generator(trainset):
        """The generator used to produce a suitable Dataset for the MineSpore trainer."""
        image = trainset[0]
        label = trainset[1]
        # annotation = trainset[1][0]
        # batch_y_true_0 = trainset[1][1]
        # batch_y_true_1=trainset[1][2]
        # batch_y_true_2 = trainset[1][3]
        # batch_gt_box0 = trainset[1][4]
        # batch_gt_box1 = trainset[1][5]
        # batch_gt_box2 = trainset[1][6]
        # img_hight = trainset[1][7]
        # img_width = trainset[1][8]
        # input_shape = trainset[1][9]
        #print('logit, target: ', image, label, flush=True)
        yield image, label
    def train(self, trainset, *args):
        """The main training loop used in the MistNet server.
        Arguments:
        trainset: The training dataset.
        """
        # column_out_names = ["image", "annotation", "batch_y_true_0", "batch_y_true_1", "batch_y_true_2",
        #                     "batch_gt_box0","batch_gt_box1", "batch_gt_box2", "img_hight", "img_width", "input_shape"]
        column_out_names = ["image", "label"]
        print('----Algorithm.dataset_generator(trainset)-----: ', list(Algorithm.dataset_generator(trainset)), flush=True)
        dataset= ds.GeneratorDataset(list(Algorithm.dataset_generator(trainset)), column_names=column_out_names)
        #feature_dataset = feature_dataset.batch(batch_size, num_parallel_workers=min(4, num_parallel_workers), drop_remainder=True)
        print('----dataset-----: ', dataset, flush=True)
        # for image, label in dataset:
        #     print('----image, label-----: ', image, label, flush=True)
        self.trainer.train(dataset)






























