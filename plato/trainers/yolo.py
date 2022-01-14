"""The YOLOV5 model for PyTorch."""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml

from plato.config import Config
from plato.datasources import yolo
from plato.trainers import basic
from plato.utils import unary_encoding
#from torch.cuda import amp
from tqdm import tqdm
from yolov5.utils.general import (box_iou, check_dataset, one_cycle, clip_coords,
                                  non_max_suppression, scale_coords, xywh2xyxy)
from yolov5.utils.loss import compute_loss
from yolov5.utils.metrics import ap_per_class
from yolov5.utils.torch_utils import time_synchronized

mixed_precision = True
try:
    import apex
    from apex import amp 
except:
    print("Unable to load mixed precision training library.")
    mixed_precision = False


class Trainer(basic.Trainer):
    """The YOLOV5 trainer."""
    def __init__(self):
        super().__init__()
        Config().params['grid_size'] = int(self.model.stride.max())

    def train_loader(self,
                     batch_size,
                     trainset,
                     sampler,
                     extract_features=False,
                     cut_layer=None):
        """The train loader for training YOLOv5 using the COCO dataset or other datasets for the
           YOLOv5 model.
        """
        return yolo.DataSource.get_train_loader(batch_size, trainset, sampler,
                                                extract_features, cut_layer)

    def train_model(self, config, trainset, sampler, cut_layer=None):  # pylint: disable=unused-argument
         
        """The training loop for YOLOv5.

        Arguments:
        config: A dictionary of configuration parameters.
        trainset: The training dataset.
        cut_layer (optional): The layer which training should start from.
        """

        logging.info("[Client #%d] Setting up training parameters.",
                     self.client_id)

        batch_size = config['batch_size']
        total_batch_size = batch_size
        epochs = config['epochs']

        if epochs == 0:
            print("jump to test")
            return

        # cuda = (self.device != 'cpu')
        nc = Config().data.num_classes  # number of classes
        names = Config().data.classes  # class names

        with open(Config().trainer.train_params) as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

        freeze = []  # parameter names to freeze (full or partial)
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False

        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / total_batch_size),
                         1)  # accumulate loss before optimizing

        hyp.update({'optimizer': 'SGD',  # ['adam', 'SGD', None] if none, default is SGD
            #：'momentum': 0.937,  # SGD momentum/Adam beta1
            'weight_decay': 5e-4,  # optimizer weight decay
            'giou': 0.05,  # giou loss gain
            'cls': 0.5,  # cls loss gain
            'cls_pw': 1.0,  # cls BCELoss positive_weight
            'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
            'obj_pw': 1.0,  # obj BCELoss positive_weight
            'iou_t': 0.20,  # iou training threshold
            'anchor_t': 4.0,  # anchor-multiple threshold
            'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
            'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
            'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
            'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
            'degrees': 0.0,  # image rotation (+/- deg)
            'translate': 0.0,  # image translation (+/- fraction)
            'scale': 0.5,  # image scale (+/- gain)
            'shear': 0.0})  # image shear (+/- deg)

        print("Learning rate is ", hyp['lr0'], flush=True)
        
        hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

        # freeze model first 4 layers
        freeze_list = ['model.0.', 'model.1.', 'model.2.', 'model.3.']
        for name, param in self.model.named_parameters():
            for freeze_layer in freeze_list:
                if name.startswith(freeze_layer):
                    param.requires_grad = False
                    
        # Sending the model to the device used for training
        self.model.to(self.device)

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        # Initializing the optimizer
        if Config().trainer.optimizer == 'Adam':
            optimizer = optim.Adam(pg0,
                                   lr=hyp['lr0'],
                                   betas=(hyp['momentum'],
                                          0.999))  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(pg0,
                                  lr=hyp['lr0'],
                                  momentum=hyp['momentum'],
                                  nesterov=True)

        optimizer.add_param_group({
            'params': pg1,
            'weight_decay': hyp['weight_decay']
        })  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        logging.info(
            '[Client %s] Optimizer groups: %g .bias, %g conv.weight, %g other',
            self.client_id, len(pg2), len(pg1), len(pg0))
        del pg0, pg1, pg2


        # Mixed precision training
        if mixed_precision:
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level='O1', verbosity=0, loss_scale=1024)
        if Config().trainer.linear_lr:
            lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp[
                'lrf']  # linear
        else:
            lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
        lr_schedule = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        # Image sizes
        nl = self.model.model[
            -1].nl  # number of detection layers (used for scaling hyp['obj'])

        # Trainloader
        logging.info("[Client #%d] Loading the dataset.", self.client_id)
        train_loader = self.train_loader(batch_size,
                                         trainset,
                                         sampler,
                                         cut_layer=cut_layer)
        
        nb = len(train_loader)

        # Model parameters
        hyp['box'] *= 3. / nl  # scale to layers
        hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        hyp['obj'] *= (Config().data.image_size /
                       640)**2 * 3. / nl  # scale to image size and layers
        self.model.nc = nc  # attach number of classes to model
        self.model.hyp = hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        self.model.names = names

        # Start training
        nw = max(
            round(hyp['warmup_epochs'] * nb),
            1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
        #scaler = amp.GradScaler(enabled=cuda)
        #compute_loss = ComputeLoss(self.model)

        cut_layer = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            logging.info(
                ('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls',
                                       'total', 'targets', 'img_size'))
            pbar = enumerate(train_loader)
            pbar = tqdm(pbar, total=nb)
            mloss = torch.zeros(4,
                                device=self.device)  # Initializing mean losses
            optimizer.zero_grad()

            # for i, (imgs, targets, *__) in pbar:
            for i, (imgs, targets) in pbar: # clients send original images as feature dataset

                ni = i + nb * epoch  # number integrated batches (since train start)
                targets = targets.to(torch.float32)
                # targets = np.moveaxis(targets, -1, -2)
                # targets = torch.from_numpy(targets).to(torch.float32)

                #print('imgs dtype before', imgs.dtype, flush=True)--torch.float64
                imgs = imgs.to(torch.float16)
                
                # save targets
                imgs, targets = imgs.to(self.device), targets.to(self.device)

                print("Images shape is ", imgs.shape, flush=True)

                # print('imgs dtype', imgs.dtype, flush=True)
                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    accumulate = max(
                        1,
                        np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        x['lr'] = np.interp(ni, xi, [
                            hyp['warmup_bias_lr'] if j == 2 else 0.0,
                            x['initial_lr'] * lf(epoch)
                        ])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(
                                ni, xi,
                                [hyp['warmup_momentum'], hyp['momentum']])

                # Forward
                if cut_layer is None:
                    pred = self.model(imgs)
                else:
                    pred = self.model.forward_from(imgs, cut_layer)
                
                loss, loss_items = compute_loss(
                        pred, targets, self.model)  # loss scaled by batch_size

                # Backward
                if mixed_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # Optimize
                if ni % accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # Print
                mloss = (mloss * i + loss_items) / (i + 1
                                                    )  # update mean losses
                mem = '%.3gG' % (torch.npu.memory_reserved() / 1E9
                                 if torch.npu.is_available() else 0)  # (GB)
                s = ('%10s' * 2 +
                     '%10.4g' * 6) % ('%g/%g' % (epoch, epochs), mem, *mloss,
                                      targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

            lr_schedule.step()
            torch.save(self.model.state_dict(),  '/home/data/model/yolov5.pth')

    def test_model(self, config, testset):  # pylint: disable=unused-argument
        """The testing loop for YOLOv5.

        Arguments:
            config: Configuration parameters as a dictionary.
            testset: The test dataset.
        """
        assert Config().data.datasource == 'YOLO'
        logging.info("[Server] Loading the dataset.")
        test_loader = yolo.DataSource.get_test_loader(config['batch_size'],
                                                      testset)

        device = next(self.model.parameters()).device  # get model device

        logging.info("[Server] Setting hyparameters.")
        # NPU yolov5 paramters
        conf_thres=0.001
        iou_thres=0.6
        merge=False
        # Configure
        self.model.eval()
        with open(Config().data.data_params) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        nc = Config().data.num_classes  # number of classes
        iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        seen = 0
        s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
        loss = torch.zeros(3, device=device)
        jdict, stats, ap, ap_class = [], [], [], []

        logging.info("[Server] Start testing model.")

        for __, (img, targets, *__) in enumerate(tqdm(test_loader, desc=s)):
            # load images from disk
            img = img.to(torch.float32).to(device, non_blocking=True)
            targets = targets.to(torch.float32).to(device)

            nb, _, height, width = img.shape  # batch size, channels, height, width
            # TODO: here we do not have orignial image shape, I temporarily mannuly set this
            height, width = 640, 640
            whwh = torch.Tensor([width, height, width, height])

            with torch.no_grad():
                # Run model
                t = time_synchronized()
                inf_out, train_out = self.model.forward_from(img)  # inference and training outputs
                t0 += time_synchronized() - t

                # Run NMS
                t = time_synchronized()
                output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge)
                t1 += time_synchronized() - t

            targets = targets.cpu().t()
            for si, pred in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if pred is None:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                clip_coords(pred, (height, width))
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                            # Append detections
                            for j in (ious > iouv[0]).nonzero():
                                d = ti[i[j]]  # detected target
                                if d not in detected:
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats)
            p, r, ap50, ap = p[:], r[:], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
            print(ap, flush=True)
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = '%20s' + '%12.3g' * 6  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map), flush=True)

        # TODO: use config to provide imgsz arguments
        imgsz = 640
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, config['batch_size'])  # tuple
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t, flush=True)

        maps = np.zeros(nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]
        return (mp, mr, map50, map, *(loss.cpu() / len(test_loader)).tolist()), maps, t

    def randomize(self, bit_array: np.ndarray, targets: np.ndarray, epsilon):
        """
        The object detection unary encoding method.
        """
        assert isinstance(bit_array, np.ndarray)
        img = unary_encoding.symmetric_unary_encoding(bit_array, 1)
        label = unary_encoding.symmetric_unary_encoding(bit_array, epsilon)
        targets_new = targets.clone().detach()
        targets_new = targets_new.detach().numpy()
        for i in range(targets_new.shape[1]):
            box = self.convert(bit_array.shape[2:], targets_new[0][i][2:])
            img[:, :, box[0]:box[2],
                box[1]:box[3]] = label[:, :, box[0]:box[2], box[1]:box[3]]
        return img

    def convert(self, size, box):
        """The convert for YOLOv5.
              Arguments:
                  size: Input feature size(w,h)
                  box:(xmin,xmax,ymin,ymax).
              """
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        x1 = max(x - 0.5 * w - 3, 0)
        x2 = min(x + 0.5 * w + 3, size[0])
        y1 = max(y - 0.5 * h - 3, 0)
        y2 = min(y + 0.5 * h + 3, size[1])

        x1 = round(x1 * size[0])
        x2 = round(x2 * size[0])
        y1 = round(y1 * size[1])
        y2 = round(y2 * size[1])

        return (int(x1), int(y1), int(x2), int(y2))
