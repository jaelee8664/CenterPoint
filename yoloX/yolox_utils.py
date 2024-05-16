import pdb
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from yolox.models.network_blocks import BaseConv
from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
from yolox.utils import vis
from yolox.data.datasets import COCO_CLASSES

def visual(output, img, test_size, cls_conf=0.5):
    """
    Visualize bounding boxes
    Args:
        output (array): [N,5] box coordinates
        img (array): images
        cls_conf (float, optional): confidence score
    """
    if output is None:
        return img, None, None, None, None
    output = output.cpu()

    bboxes = output[:, 0:4]

    # preprocessing: resize
    scale = min(
                test_size[0] / float(img.shape[0]), test_size[1] / float(img.shape[1])
            )
    bboxes /= scale
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]
    vis_res = vis(img, bboxes, scores, cls, cls_conf, COCO_CLASSES)

    return vis_res, bboxes, scores, cls, COCO_CLASSES

def preproc(img, input_size, mean, std, swap=(2, 0, 1)):
    """
    preprocess of yolox model
    Args:
        img (array): images
        input_size (_type_): size of img to put yolox
        mean (_type_): mean of img
        std (_type_): std of img
        swap (tuple, optional): axis swap Defaults to (2, 0, 1).

    Returns:
        padded img for yolox and padding information
    """
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0

    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def postprocess(prediction, num_classes, conf_thre=0.1, nms_thre=0.45):
    """
    postprocess of yolox
    Args:
        prediction (array): predcied bbox coordinates
        num_classes (_type_): number of classes
        conf_thre (float, optional): confidence thres. Defaults to 0.1.
        nms_thre (float, optional): nms thres. Defaults to 0.45.

    Returns:
        _type_: corrected bboxes
    """
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5: 5 + num_classes], 1, keepdim=True
        )

        conf_mask = (image_pred[:, 4] *
                     class_conf.squeeze() >= conf_thre).squeeze()
        detections = torch.cat(
            (image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue
        if detections.shape[1] == 1:
            detections = detections.squeeze(0)
        try:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre
            )
        except ValueError:
            pdb.set_trace()
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def fuse_conv_and_bn(conv, bn):
    """
    Conv2d + batchnorm
    Args:
        conv : convolution2d
        bn : batch normalization

    Returns:
        fused layer
    """
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(
        torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_model(model):
    """
    fuse batch normalization to Convolution layer
    """
    for m in model.modules():
        if isinstance(BaseConv, type(m)) and hasattr(m, "bn"):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, "bn")  # remove batchnorm
            m.forward = m.fuseforward  # update forward
    return model


def get_model(depth=0.33, width=0.375):
    """
    initialize yolox model
    """
    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth=depth, width=width, in_channels=in_channels) # s: 0.33 0.50
    head = YOLOXHead(num_classes=1, width=width, in_channels=in_channels)
    model = YOLOX(backbone, head)

    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)
    model.eval()
    return model
