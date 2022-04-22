import argparse
from packages.ms_yolov5.src.yolo import YOLOV5s, YoloWithLossCell, TrainingWrapper
from packages.ms_yolov5.src.yolov5_backbone import YOLOv5Backbone_to
from packages.ms_yolov5.src.initializer import default_recurisive_init, load_yolov5_params
from packages.ms_yolov5.src.util import AverageMeter, get_param_groups
from mindspore.nn.optim.momentum import Momentum
from mindspore import Tensor
import mindspore.nn as nn

class Model(nn.Cell):
    def __init__(self, args=None):
        super(Model, self).__init__(auto_prefix = False)

        self.network_to = YOLOv5Backbone_to()

        self.yolo_network = YOLOV5s(is_training=True)

    def load_model_train(self, args, lr):
        default_recurisive_init(self.yolo_network)
        load_yolov5_params(args, self.yolo_network)
        self.network_from = YoloWithLossCell(self.yolo_network)

        #lr = get_lr(args)

        opt = Momentum(params=get_param_groups(self.network_from),
                       learning_rate=Tensor(lr),
                       momentum=args.momentum,
                       weight_decay=args.weight_decay,
                       loss_scale=args.loss_scale)

        self.network_from = TrainingWrapper(self.network_from, opt, args.loss_scale // 2)
        #self.network_from.set_train()
        #print('load network_from.', self.network_from, flush=True)

    def forward_to(self, x):

        logits = self.network_to(x)
        return logits

    def forward_from(self, logits, *args): # *args = batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1,batch_gt_box2, img_hight, img_width, input_shape
        loss = self.network_from(logits, *args)
        return loss