import copy
from typing import Tuple
import numpy as np
from plato.trainers import base
from plato.utils import unary_encoding


class Trainer(base.Trainer):
    """A trainer for NNRT."""
    def __init__(self, model=None):
        super().__init__()

        # TODO: if the model is none, we get model from registry
        self.model = model

    def save_model(self, filename=None):
        pass

    def load_model(self, filename=None):
        pass

    def train(self, trainset, sampler, cut_layer=None) -> Tuple[bool, float]:
        pass

    def test(self, testset) -> float:
        pass

    async def server_test(self, testset):
        pass

    def randomize(self, bit_array: np.ndarray, targets: np.ndarray, epsilon):
        """
        The object detection unary encoding method.
        """
        assert isinstance(bit_array, np.ndarray)
        img = unary_encoding.symmetric_unary_encoding(bit_array, 100)
        label = unary_encoding.symmetric_unary_encoding(bit_array, epsilon)
        targets_new = copy.deepcopy(targets)  #[1,10,5]
        # print("targets_new: ", targets_new, flush=True)
        # print("targets_new.shape:", targets_new.shape, flush=True)
        #targets_new = targets_new.transpose(0, 2, 1)
        for i in range(targets_new.shape[1]):
            box = self.convert(bit_array.shape[2:], targets_new[0][i])
            img[:, :, box[0]:box[2],
                box[1]:box[3]] = label[:, :, box[0]:box[2], box[1]:box[3]]
        # print("img: ", img, flush=True)
        # print("img.shape:", img.shape, flush=True)
        return img


    def convert(self, size, box):
        """The convert for YOLOv5.
              Arguments:
                  size: Input feature size(w,h)
                  box:(xmin,xmax,ymin,ymax).
              """
        # print("box:", box, flush=True)
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        # x1 = max(x - 0.5 * w - 3, 0)
        # x2 = min(x + 0.5 * w + 3, size[0])
        # y1 = max(y - 0.5 * h - 3, 0)
        # y2 = min(y + 0.5 * h + 3, size[1])
        # x1 = round(x1/640 * size[0])
        # x2 = round(x2/640 * size[0])
        # y1 = round(y1/640 * size[1])
        # y2 = round(y2/640 * size[1])
        x1 = max(round(x1/640 * size[0])-3, 0)
        x2 = min(round(x2/640 * size[0])+3, size[0])
        y1 = max(round(y1/640 * size[1])-3, 0)
        y2 = min(round(y2/640 * size[1])+3, size[1])

        return (int(x1), int(y1), int(x2), int(y2))