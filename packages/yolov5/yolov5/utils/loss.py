# Loss functions

import torch
import torch.nn as nn

# from yolov5.utils.general import bbox_iou
from yolov5.utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


# NPU version of computing loss
class DeterministicIndex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices_list):
        ctx.x = x
        ctx.indices_list = indices_list
        return x[indices_list[0], indices_list[1], :, indices_list[2], indices_list[3]]

    @staticmethod
    def backward(ctx, grad_output):
        tmp = torch.zeros_like(ctx.x)
        ind0, ind1, ind2, ind3 = ctx.indices_list
        tmp[ind0, ind1, :, ind2, ind3] = grad_output
        return tmp, None

def compute_loss(p, targets, model):  # predictions, targets, model
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors, targets_mask, targets_sum_mask = build_targets(p, targets, model)  # targets
    h = model.hyp  # hyperparameters

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']], device=device), reduction='sum')
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']], device=device), reduction='mean')

    # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # per output
    nt = 0  # number of targets
    np = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        allmask = targets_mask[i]
        sum_mask = targets_sum_mask[i]
        tobj = torch.zeros_like(pi[:, :, 0, :, :]).to(device)  # target obj

        nb = b.shape[0]  # number of targets
        if sum_mask.item() > 0:
            nt += nb  # cumulative targets
            #ps = pi[b, a, :, gj, gi].permute(1, 0).contiguous()  # prediction subset corresponding to targets
            ps = DeterministicIndex.apply(pi, (b, a, gj, gi)).permute(1, 0).contiguous()
            # GIoU
            pxy = ps.index_select(0, torch.tensor([0, 1], device=targets.device))
            pwh = ps.index_select(0, torch.tensor([2, 3], device=targets.device))

            pxy = pxy.sigmoid() * 2. - 0.5
            pwh = (pwh.sigmoid() * 2) ** 2 * (anchors[i].T)
            pbox = torch.cat((pxy, pwh), 0)  # predicted box
            # FIXME: 08/13 AttributeError: module 'torch' has no attribute 'npu_giou'
            # FIXME: it needs FrameworkPTAdapter v2.0.3
            # giou = torch.npu_giou(pbox, tbox[i], trans=True, is_cross=False).squeeze()
            giou = bbox_iou(pbox, tbox[i], x1y1x2y2=False, GIoU=True)
            giou = giou * (allmask) + (1. - allmask)
            lbox += (1.0 - giou).sum() / (sum_mask) # giou loss
            # Obj
            giou = giou * (allmask)
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # Class
            if model.nc > 1:  # cls loss (only if multiple classes)
                tmp = ps[5:, :]
                tmp = tmp * (allmask) - (1.- allmask) * 50.
                t = torch.full_like(tmp, cn).to(device)  # targets
                range_nb = torch.arange(nb, device=device).long()
                t[tcls[i], range_nb] = cp

                t = t * (allmask)
                lcls += (BCEcls(tmp, t) / (sum_mask * t.shape[0]).float()) # BCE
            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[:, :, 4, :, :], tobj) * balance[i]  # obj loss

    s = 3 / np  # output count scaling
    lbox *= h['giou'] * s
    lobj *= h['obj'] * s * (1.4 if np == 4 else 1.)
    lcls *= h['cls'] * s
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    det = model.module.model[-1] if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel) \
        else model.model[-1]  # Detect() module
    na, nt = det.na, targets.shape[1]  # number of anchors, targets
    batch_size = p[0].shape[0]
    nt_max = 32 * batch_size
    while nt > nt_max:
        nt_max *= 2
        print('**************** nt max=', nt_max)
    max_target = torch.zeros(6, nt_max, device=targets.device)   #  (6, 1024)
    print("max_target type ", max_target.dtype)
    print("target type ", target.dtype)
    
    max_target[0, :nt] = targets[0, :].dtype(torch.float32)
    max_target[1, :nt] = targets[1, :].dtype(torch.float32)
    max_target[2, :nt] = targets[2, :].dtype(torch.float32)
    max_target[3, :nt] = targets[3, :].dtype(torch.float32)
    max_target[4, :nt] = targets[4, :].dtype(torch.float32)
    max_target[5, :nt] = targets[5, :].dtype(torch.float32)
    
    tcls, tbox, indices, anch, targets_mask, targets_sum_mask = [], [], [], [], [], []
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
    off_list = [
        torch.tensor([[1.], [0.]], device=targets.device),
        torch.tensor([[0.], [1.]], device=targets.device),
        torch.tensor([[-1.], [0.]], device=targets.device),
        torch.tensor([[0.], [-1.]], device=targets.device)
    ]
    at = torch.arange(na).view(na, 1).repeat(1, nt_max)  # anchor tensor, same as .repeat_interleave(nt)  (1024, 3)
    a = at.view(-1)
    a = torch.cat((a, a, a, a, a), 0)

    g = 0.5  # offset
    style = 'rect4'
    for i in range(det.nl):
        anchors = det.anchors[i].float()
        gain[2:] = torch.tensor(p[i].shape)[[4, 3, 4, 3]].float()  # xyxy gain

        # Match targets to anchors
        t, offsets = max_target * gain[:, None], 0
        allmask = torch.zeros((15 * nt_max)).to(targets.device)
        sum_mask = torch.zeros((1)).to(targets.device)
        if nt:
            r = t[None, 4:6, :] / anchors[..., None]  # wh ratio
            fmask = torch.max(r, 1. / r).max(1)[0] < model.hyp['anchor_t']  # compare
            fmask = fmask.view(1, -1)
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            t = t.repeat(1, 1, na).view(6, -1)  # filter

            # overlaps
            gxy = t.index_select(0, torch.tensor([2, 3], device=targets.device)) # (3072, 2)
            z = torch.zeros_like(gxy)

            jk = (gxy % 1. < g) & (gxy > 1.)
            lm = (gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]][:, None] - 1.))
            jk, lm = jk&fmask, lm&fmask
            allmask = torch.cat((fmask, jk, lm), 0).view(1, -1).float()
            t = torch.cat((t, t, t, t, t), 1)
            offsets = torch.cat((z, z + off_list[0], z + off_list[1], z + off_list[2], z + off_list[3]), 1) * g

            sum_mask = allmask.sum()
            t = t * allmask

        # Define
        b = t.index_select(0, torch.tensor([0], device=targets.device)).long().view(-1)   #(3072 * 5)
        c = t.index_select(0, torch.tensor([1], device=targets.device)).long().view(-1)   #(3072 * 5)
        gxy = t.index_select(0, torch.tensor([2, 3], device=targets.device)) #(2, 3072 * 5)
        gwh = t.index_select(0, torch.tensor([4, 5], device=targets.device)) #(2, 3072 * 5)
        gij = gxy - offsets
        gij2 = gij.long()
        gi = gij2.index_select(0, torch.tensor([0], device=targets.device)).view(-1) #(2, 3072 * 5)
        gj = gij2.index_select(0, torch.tensor([1], device=targets.device)).view(-1) #(2, 3072 * 5)

        # Append
        indices.append((b, a, gj, gi))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij2.float(), gwh), 0))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class
        targets_mask.append(allmask)
        targets_sum_mask.append(sum_mask)

    return tcls, tbox, indices, anch, targets_mask, targets_sum_mask

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou