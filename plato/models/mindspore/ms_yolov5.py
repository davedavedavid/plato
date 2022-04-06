import argparse
from src.lr_scheduler import get_lr
from ms_yolov5.src.yolo import YOLOV5s, YoloWithLossCell, TrainingWrapper
from ms_yolov5.src.yolov5_backbone import YOLOv5Backbone_to


class Model(nn.Cell):
    def __init__(self, model_config, num_classes):
        super(Model, self).__init__()

        network_to = YOLOv5Backbone_to()

        network_from = YOLOV5s(is_training=True)

        # for param in network_to.trainable_params():
        #     param.requires_grad = False
        #     print(param)

        network_from = YoloWithLossCell(network_from)

        network_from = TrainingWrapper(network_from, opt, args.loss_scale // 2)

    def forward_to(self, x):

        logits = network_to(x)
        return logits

    def forward_from(self, logits, *args): # *args = batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1,batch_gt_box2, img_hight, img_width, input_shape
        loss = network_from(logits, *args)
        return loss