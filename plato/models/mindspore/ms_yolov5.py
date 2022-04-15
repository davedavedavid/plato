import argparse
from packages.ms_yolov5.src.yolo import YOLOV5s, YoloWithLossCell, TrainingWrapper
from packages.ms_yolov5.src.yolov5_backbone import YOLOv5Backbone_to
from packages.ms_yolov5.src.initializer import default_recurisive_init, load_yolov5_params
from packages.ms_yolov5.src.lr_scheduler import get_lr
from packages.ms_yolov5.src.util import AverageMeter, get_param_groups
from mindspore.nn.optim.momentum import Momentum
from mindspore import Tensor
import mindspore.nn as nn

class Model(nn.Cell):
    def parse_args(cloud_args=None):
        """Parse train arguments."""
        parser = argparse.ArgumentParser('mindspore coco training')
        # network related
        parser.add_argument('--resume_yolov5', default='/home/data/pretrained/YoloV5_for_MindSpore_0-300_274800.ckpt',
                            type=str,
                            help='The ckpt file of YOLOv5, which used to fine tune. Default: ""')
        # optimizer and lr related
        parser.add_argument('--lr_scheduler', default='cosine_annealing', type=str,
                            help='Learning rate scheduler, options: exponential, cosine_annealing. Default: exponential')
        parser.add_argument('--lr', default=0.013, type=float, help='Learning rate. Default: 0.01')
        parser.add_argument('--lr_epochs', type=str, default='220,250',
                            help='Epoch of changing of lr changing, split with ",". Default: 220,250')
        parser.add_argument('--lr_gamma', type=float, default=0.1,
                            help='Decrease lr by a factor of exponential lr_scheduler. Default: 0.1')

        parser.add_argument('--weight_decay', type=float, default=0.0005,
                            help='Weight decay factor. Default: 0.0005')
        parser.add_argument('--momentum', type=float, default=0.9, help='Momentum. Default: 0.9')
        # loss related
        parser.add_argument('--loss_scale', type=int, default=1024, help='Static loss scale. Default: 1024')

        args, _ = parser.parse_known_args()

        def merge_args(args, cloud_args):
            args_dict = vars(args)
            if isinstance(cloud_args, dict):
                for key in cloud_args.keys():
                    val = cloud_args[key]
                    if key in args_dict and val:
                        arg_type = type(args_dict[key])
                        if arg_type is not type(None):
                            val = arg_type(val)
                        args_dict[key] = val
            return args

        args = merge_args(args, cloud_args)
        return args
    def __init__(self, args=None):
        super(Model, self).__init__()

        self.network_to = YOLOv5Backbone_to()

        self.yolo_network = YOLOV5s(is_training=True)

    #def load_model_train(self, args):
        default_recurisive_init(self.yolo_network)
        load_yolov5_params(args, self.yolo_network)
        self.network_from = YoloWithLossCell(self.yolo_network)

        lr = get_lr(args)

        opt = Momentum(params=get_param_groups(self.network_from),
                       learning_rate=Tensor(lr),
                       momentum=args.momentum,
                       weight_decay=args.weight_decay,
                       loss_scale=args.loss_scale)

        self.network_from = TrainingWrapper(self.network_from, opt, args.loss_scale // 2)
        self.network_from.set_train()
        #print('load network_from.', self.network_from, flush=True)

    def forward_to(self, x):

        logits = self.network_to(x)
        return logits

    def forward_from(self, logits, *args): # *args = batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1,batch_gt_box2, img_hight, img_width, input_shape
        loss = self.network_from(logits, *args)
        return loss