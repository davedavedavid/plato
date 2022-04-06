import argparse
from packages.ms_yolov5.src.yolo import YOLOV5s, YoloWithLossCell, TrainingWrapper
from packages.ms_yolov5.src.yolov5_backbone import YOLOv5Backbone_to
from packages.ms_yolov5.src.initializer import default_recurisive_init, load_yolov5_params
from packages.ms_yolov5.src.lr_scheduler import get_lr
from packages.ms_yolov5.src.util import AverageMeter, get_param_groups
from mindspore.nn.optim.momentum import Momentum
from mindspore import Tensor

class Model(nn.Cell):
    def __init__(self, model_config=None, args=None):
        super(Model, self).__init__()

        self.network_to = YOLOv5Backbone_to()

        self.network_from = YOLOV5s(is_training=True)

    def load_model_train(self, args):
        default_recurisive_init(self.network_from)
        load_yolov5_params(args, self.network_from)
        self.network_from = YoloWithLossCell(self.network_from)

        lr = get_lr(args)

        opt = Momentum(params=get_param_groups(self.network_from),
                       learning_rate=Tensor(lr),
                       momentum=args.momentum,
                       weight_decay=args.weight_decay,
                       loss_scale=args.loss_scale)

        self.network_from = TrainingWrapper(self.network_from, opt, args.loss_scale // 2)
        self.network_from.set_train()
        print('load network_from.', self.network_from, lr, flush=True)

    def forward_to(self, x):

        logits = self.network_to(x)
        return logits

    def forward_from(self, logits, *args): # *args = batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1,batch_gt_box2, img_hight, img_width, input_shape
        loss = self.network_from(logits, *args)
        return loss