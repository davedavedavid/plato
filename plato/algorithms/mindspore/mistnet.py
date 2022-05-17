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
import numpy as np

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
        #print('trainset: ', len(trainset), flush=True)

        for i in range(len(trainset)): #[[image,[]],[]]
            image = trainset[i][0]
            #print('image: ', image, image.shape, flush=True)
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
        #print('trainset: ', trainset, len(trainset), flush=True)
        #trainset = np.load('/home/data/pretrained/mzoo_data.npy', allow_pickle=True)
        #print('trainset: ', trainset, len(trainset), flush=True)
        column_out_names = ["image", "annotation", "batch_y_true_0", "batch_y_true_1", "batch_y_true_2",
                             "batch_gt_box0","batch_gt_box1", "batch_gt_box2", "img_hight", "img_width", "input_shape"]
        data_size = len(trainset)
        dataset= ds.GeneratorDataset(source=list(Algorithm.dataset_generator(trainset)), column_names=column_out_names)
        device_num = 1
        cores = multiprocessing.cpu_count()
        num_parallel_workers = int(cores / device_num)
        per_batch_size = 1#Config().trainer.per_batch_size
        max_epoch =200# Config().trainer.max_epoch
        dataset = dataset.batch(per_batch_size, num_parallel_workers=min(4, num_parallel_workers),
                                        drop_remainder=True)

        #dataset = dataset.repeat(max_epoch)
        # for image,annotation, batch_y_true_0,batch_y_true_1,batch_y_true_2,batch_gt_box0,\
        #           batch_gt_box1,batch_gt_box2,img_hight,img_width,input_shape in dataset:
        #     #print('----image-----: ',image, image.shape, annotation, annotation.shape, flush=True)
        #     print('----batch_y_true_0-----: ', batch_y_true_0,batch_y_true_0.shape, flush=True)
        #     print('----batch_gt_box0-----: ', batch_gt_box0, batch_gt_box0.shape, flush=True)
        #     print('----batch_gt_box1-----: ', batch_gt_box1, batch_gt_box1.shape, flush=True)
        #     print('----batch_gt_box2-----: ', batch_gt_box2,batch_gt_box2.shape, flush=True)
        self.trainer.train(dataset, data_size, per_batch_size, max_epoch)






























