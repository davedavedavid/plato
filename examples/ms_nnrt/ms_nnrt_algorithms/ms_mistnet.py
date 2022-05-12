import logging
import random
import time
import numpy as np
from examples.ms_nnrt.ms_nnrt_algorithms import ms_fedavg
from plato.config import Config
from plato.utils import unary_encoding_1b
from examples.ms_nnrt.ms_nnrt_datasource_yolo_utils import DistributedSampler, MultiScaleTrans, PreprocessTrueBox
from examples.ms_nnrt.config import ConfigYOLOV5
import random

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
        np.random.seed(5)
        random.seed(1)
        tic = time.perf_counter()

        feature_dataset = []

        _randomize = getattr(self.trainer, "randomize", None)

        features_shape = self.features_shape()
        np.random.seed(5)
        config = ConfigYOLOV5()
        device_num = 1
        distributed_sampler = DistributedSampler(len(dataset), device_num, rank=None, shuffle=True)
        dataset.size = len(distributed_sampler)
        config.dataset_size = len(dataset)
        multi_scale_trans = MultiScaleTrans(config, device_num)
        dataset.transforms = multi_scale_trans
        PreprocessTrueBox_ = PreprocessTrueBox(config)

        def concatenate(images):
            images = np.concatenate((images[..., ::2, ::2], images[..., 1::2, ::2],
                                     images[..., ::2, 1::2], images[..., 1::2, 1::2]), axis=0)
            return images

        # for i in range(5):
        for img, anno, input_size, mosaic_flag in dataset:
            np.array(anno)
            img_hight = input_size[0]
            img_wight = input_size[1]
            #print("img1:", img, img.shape, flush=True)
            #print("max:", np.max(img))
            #print("anno:", anno, flush=True)
            #input_size = [img_hight, img_wight]
            image, annotation, size = multi_scale_trans(img=img, anno=np.array(anno), input_size=input_size,
                                                        mosaic_flag=mosaic_flag)
            #print("img2:", image, image.shape, flush=True)
            print("annotation1: ", annotation, flush=True)
            annotation, bbox1, bbox2, bbox3, gt_box1, gt_box2, gt_box3 = PreprocessTrueBox_(annotation, size)
            #print("annotation2, bbox1, bbox2, bbox3, gt_box1, gt_box2, gt_box3: ", annotation, bbox1, bbox2, bbox3, gt_box1, gt_box2, gt_box3, flush=True)

            annotation_x = [annotation, bbox1, bbox2, bbox3, gt_box1, gt_box2, gt_box3, img_hight, img_wight, size]

            image = np.array(image, dtype='float32')
            mean = [m * 255 for m in [0.485, 0.456, 0.406]]
            std = [s * 255 for s in [0.229, 0.224, 0.225]]
            mean = np.array(mean, dtype=image.dtype)
            std = np.array(std, dtype=image.dtype)
            image = (image - mean) / std
            image = image.swapaxes(1, 2).swapaxes(0, 1)  # HWC->HCW->CHW    CV.HWC2CHW  or images.transpose((2,0,1))
            #print("img3:", image, image.shape, flush=True)
            ds = concatenate(image)
            inputs = ds.astype(np.float32)
            #print("inputs:", inputs, inputs.shape, flush=True)
            #  1*12*320*320 input   logits: 1 * 128 *80 *80
            #inputs = np.zeros((12, 320, 320))
            #inputs = inputs.astype(np.float32)
            #print("inputs: ", inputs, inputs.shape, flush=True)
            logits = self.model.forward(inputs)
            logits = np.reshape(logits, features_shape)
            #print("logits: ", logits, logits.shape, flush=True)
            #np.save("/home/data/test_logits.npy", logits)
            #print("input_data: ", bbox1, bbox2, bbox3, gt_box1, gt_box2, gt_box3, flush=True)
            annotation_x[0] = np.expand_dims(annotation_x[0],
                                             axis=0)  # add batch axis to make sure self.train.randomize correct

            if epsilon is not None:
                logging.info("epsilon is %d.", epsilon)
                # logits = unary_encoding_1b.encode(logits)
                # if callable(_randomize):
                #     logits = self.trainer.randomize(logits, annotation_x[0], epsilon=1)
                #     # print(' annotation_x[0], logits.shape, epsilon', annotation_x[0].shape, logits.shape, epsilon, flush=True)
                # else:
                #     logits = unary_encoding_1b.randomize(logits, epsilon)
                    # Pytorch is currently not supported on A500 and we cannot convert
                    # numpy array to tensor
                # if self.trainer.device != 'cpu':
                #     logits = logits.astype('float16')
                # else:
                #     logits = logits.astype('float32')

            #for i in np.arange(logits.shape[0]):  # each sample in the batch
            feature_dataset.append((logits[0], annotation_x[:]))
            #print("feature_dataset: ", feature_dataset, len(feature_dataset), flush=True)
        toc = time.perf_counter()
        logging.info("[Client #%d] Features extracted from %s examples.",
                     self.client_id, len(feature_dataset))
        logging.info("[Client #{}] Time used: {:.2f} seconds.".format(
            self.client_id, toc - tic))
        #print('feature_dataset: ', len(feature_dataset), feature_dataset,  flush=True)
        return feature_dataset

    def features_shape(self):
        """ Return the features shape of the cutlayer output. """
        # TODO: Do not hard code the features shape
        # return [-1, 320, 120, 120]
        return [-1, 128, 80, 80]  # input: 12*320*320
