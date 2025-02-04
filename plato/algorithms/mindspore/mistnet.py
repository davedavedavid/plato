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
import multiprocessing
from plato.config import Config

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

    @staticmethod
    def dataset_generator(trainset):
        """The generator used to produce a suitable Dataset for the MineSpore trainer."""
        for i in range(len(trainset)): #[[image,[]],[]]
            image = trainset[i][0]
            annotation = trainset[i][1][0]
            batch_y_true_0 = trainset[i][1][1]
            batch_y_true_1=trainset[i][1][2]
            batch_y_true_2 = trainset[i][1][3]
            batch_gt_box0 = trainset[i][1][4]
            batch_gt_box1 = trainset[i][1][5]
            batch_gt_box2 = trainset[i][1][6]
            img_hight = trainset[i][1][7]
            img_width = trainset[i][1][8]
            input_shape = trainset[i][1][9]
            yield image,annotation, batch_y_true_0,batch_y_true_1,batch_y_true_2,batch_gt_box0,\
                  batch_gt_box1,batch_gt_box2,img_hight,img_width,input_shape

    def train(self, trainset, *args):
        column_out_names = ["image", "annotation", "batch_y_true_0", "batch_y_true_1", "batch_y_true_2",
                             "batch_gt_box0","batch_gt_box1", "batch_gt_box2", "img_hight", "img_width", "input_shape"]
        data_size = len(trainset)
        print('------data_size----: ', data_size, flush=True)
        dataset= ds.GeneratorDataset(source=list(Algorithm.dataset_generator(trainset)), column_names=column_out_names)

        num_parallel_workers = 1
        per_batch_size = Config().trainer.per_batch_size
        repeat_epoch = Config().trainer.repeat_epoch
        group_size = Config().trainer.group_size
        dataset = dataset.batch(per_batch_size, num_parallel_workers= num_parallel_workers,
                                        drop_remainder=True)
        dataset = dataset.repeat(repeat_epoch)

        self.trainer.train(dataset, data_size, per_batch_size, repeat_epoch, group_size)






























