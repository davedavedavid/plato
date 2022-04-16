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

    @staticmethod
    def dataset_generator(trainset):
        """The generator used to produce a suitable Dataset for the MineSpore trainer."""
        logit = trainset[0]
        label = trainset[1]
        print('logit, target', len(trainset[0]), len(trainset[1]), flush=True)
        yield logit, label
    def train(self, trainset, *args):
        """The main training loop used in the MistNet server.
        Arguments:
        trainset: The training dataset.
        """
        #dataset = Algorithm.dataset_generator(trainset)
        dataset= ds.GeneratorDataset(Algorithm.dataset_generator(trainset), column_names=["image", "label"])
        #feature_dataset = feature_dataset.batch(batch_size, num_parallel_workers=min(4, num_parallel_workers), drop_remainder=True)
        print('----dataset-----: ', dataset, flush=True)
        for img, lab in dataset:
            print('----img, lab-----: ', img, lab, flush=True)
        self.trainer.train(dataset)






























