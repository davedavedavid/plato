"""
A federated learning server for MistNet.
Reference:
P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local
Differential Privacy," found in docs/papers.
"""

import logging
import os
from itertools import chain
from plato.algorithms.mistnet import FeatureDataset

from plato.config import Config
from plato.samplers import all_inclusive
from plato.servers import fedavg
import torch
import mindspore

class Server(fedavg.Server):
    """The MistNet server for federated learning."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        # MistNet requires one round of client-server communication
        assert Config().trainer.rounds == 1

    def load_trainer(self):
        """Setting up a pre-trained model to be loaded on the server."""
        super().load_trainer()
        
        # check local model exist or not
        logging.info("[Server #%d] Loading a pre-trained model.", os.getpid())
        # self.trainer.load_model()

    async def process_reports(self):
        """Process the features extracted by the client and perform server-side training."""
        features = [features for (__, features) in self.updates]
        # Faster way to deep flatten a list of lists compared to list comprehension
        feature_dataset = list(chain.from_iterable(features))
        print("feature_dataset", feature_dataset, type(feature_dataset), len(feature_dataset), flush=True)
		# convert feature dataset from numpy to torch tensor
        feature_dataset_tensor = []
        for feature in feature_dataset:
            print("len(feature) ", len(feature), flush=True)
            if hasattr(Config().trainer, 'use_mindspore'):
                feature_dataset_tensor.append([mindspore.Tensor(elem) for elem in feature])
            else:
                feature_dataset_tensor.append([torch.from_numpy(elem) for elem in feature])

            # Training the model using all the features received from the client
        print("len feature_dataset_tensor ", len(feature_dataset_tensor), flush=True)
        sampler = all_inclusive.Sampler(feature_dataset_tensor)
        self.algorithm.train(feature_dataset_tensor, sampler,
                             Config().algorithm.cut_layer)

        # Test the updated model
        if not Config().clients.do_test:
            self.accuracy = self.trainer.test(FeatureDataset(feature_dataset_tensor))
            logging.info('[Server #{:d}] Finish testing model.'.format(os.getpid()))
            # logging.info('[Server #{:d}] Global model accuracy: {:.2f}%\n'.format(
            #     os.getpid(), 100 * self.accuracy))

        await self.wrap_up_processing_reports()
