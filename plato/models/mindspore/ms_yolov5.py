import argparse
from src.lr_scheduler import get_lr
from ms_yolov5.src.yolo import YOLOV5s, YoloWithLossCell, TrainingWrapper
from ms_yolov5.src.yolov5_backbone import YOLOv5Backbone_to
from ms_yolov5.src.initializer import default_recurisive_init, load_yolov5_params
from ms_yolov5.src.lr_scheduler import get_lr
from mindspore.nn.optim.momentum import Momentum
from ms_yolov5.src.util import AverageMeter, get_param_groups
from mindspore import Tensor

class Model(nn.Cell):
    def __init__(self, model_config, opt, args):
        super(Model, self).__init__()

        network_to = YOLOv5Backbone_to()

        network_from = YOLOV5s(is_training=True)

        # for param in network_to.trainable_params():
        #     param.requires_grad = False
        #     print(param)

        # default is kaiming-normal
        default_recurisive_init(network_t)
        load_yolov5_params(args, network_t)
        network_from = YoloWithLossCell(network_from)

        lr = get_lr(args)

        opt = Momentum(params=get_param_groups(network_t),
                       learning_rate=Tensor(lr),
                       momentum=args.momentum,
                       weight_decay=args.weight_decay,
                       loss_scale=args.loss_scale)

        network_from = TrainingWrapper(network_from, opt, args.loss_scale // 2)
        network_from.set_train()

    def forward_to(self, x):

        logits = network_to(x)
        return logits

    def forward_from(self, logits, *args): # *args = batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1,batch_gt_box2, img_hight, img_width, input_shape
        loss = network_from(logits, *args)
        return loss