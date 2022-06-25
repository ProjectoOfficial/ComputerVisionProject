import math
import torch

def clip_boxes_to_image(boxes, size):
    """
    Clip boxes so that they lie inside an image of size `size`.
    :param boxes: boxes in (x1, y1, x2, y2) format
    :param size: size of the image
    :return: clipped_boxes (Tensor[N, 4])
    """

    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]  # x1, x2
    boxes_y = boxes[..., 1::2]  # y1, y2
    height, width = size

    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)

def nms(boxes, scores, iou_threshold):
    """
     Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU).
    NMS iteratively removes lower scoring boxes which have an IoU greater than iou_threshold with another (higher scoring)
    box.
    :param boxes: Tensor[N, 4]), boxes to perform NMS on. They are expected to be in (x1, y1, x2, y2) format
    :param scores: Tensor[N], scores for each one of the boxes
    :param iou_threshold: float, discards all overlapping boxes with IoU < iou_threshold
    :return: int64 tensor with the indices of the elements that have been kept by NMS, sorted in decreasing order of scores
    """

    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)

def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.
    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories
    :param boxes: Tensor[N, 4], boxes where NMS will be performed. They are expected to be in (x1, y1, x2, y2) format
    :param scores:  Tensor[N], scores for each one of the boxes
    :param idxs: Tensor[N], indices of the categories for each one of the boxes.
    :param iou_threshold: float, discards all overlapping boxes, with IoU < iou_threshold
    :return: int64 tensor with the indices of the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()

    # to(): Performs Tensor dtype and/or device conversion
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep

def encode_boxes(reference_boxes, proposals, weights):
    """
    Encode a set of proposals with respect to some reference boxes
    :param reference_boxes: reference boxes(gt)
    :param proposals: boxes to be encoded(anchors)
    :param weights:
    :return:
    """

    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    # Returns a new tensor with a dimension of size one inserted at the specified position.
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    # parse widths and heights
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1

    # center point
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets

def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    :param boxes:  boxes for which the area will be computed. They
                   are expected to be in (x1, y1, x2, y2) format
    :return: area for each box
    """

    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    """
     Calculate intersection-over-union (Jaccard index) of boxes.
     Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    :param boxes1: boxes1 (Tensor[N, 4])
    :param boxes2: boxes2 (Tensor[M, 4])
    :return: iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def permute_and_flatten(layer, N, A, C, H, W):
    """
    adjust tensor order，and reshape
    :param layer: classification or bboxes parameters
    :param N: batch_size
    :param A: anchors_num_per_position
    :param C: classes_num or bbox coordinate
    :param H: height
    :param W: width
    :return: Tensor after adjusting order and reshaping
    """

    # [batch_size, anchors_num_per_position * (C or 4), height, width]
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, -1, C]
    layer = layer.reshape(N, -1, C)
    return layer

def concat_box_prediction_layers(box_cls, box_regression):
    """
    Adjust box classification and bbox regression parameters order and reshape
    :param box_cls: target prediction score
    :param box_regression: bbox regression parameters
    :return: [N, -1, C]
    """

    box_cls_flattened = []
    box_regression_flattened = []

    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # [batch_size, anchors_num_per_position * classes_num, height, width], class_num is equal 2
        N, AxC, H, W = box_cls_per_level.shape
        # [batch_size, anchors_num_per_position * 4, height, width]
        Ax4 = box_regression_per_level.shape[1]
        # anchors_num_per_position
        A = Ax4 // 4
        # classes_num
        C = AxC // A

        # [N, -1, C]
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        # [N, -1, C]
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)  # start_dim, end_dim
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression

def remove_small_boxes(boxes, min_size):
    """
    Remove boxes which contains at least one side smaller than min_size.
    :param boxes: boxes in (x1, y1, x2, y2) format
    :param min_size: minimum size
    :return: indices of the boxes that have both sides
            larger than min_size
    """

    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    # nonzero(): Returns a tensor containing the indices of all non-zero elements of input
    keep = keep.nonzero().squeeze(1)
    return keep

class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    :param weights: 4-element tuple, represented calculation weights of x, y, h, w
    :param bbox_xform_clip: float, represented maximum of height and width
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        This class is inserted to calculate parameters of regression
        :param reference_boxes: gt bbox
        :param proposals: anchors bbox
        :return: regression parameters
        """

        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)

        # targets_dx, targets_dy, targets_dw, targets_dh
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some reference boxes
        :param reference_boxes: reference boxes
        :param proposals: boxes to be encoded
        :return:
        """

        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes, boxes):
        """
        decode regression parameters
        :param rel_codes: bbox regression parameters
        :param boxes: anchors
        :return:
        """

        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)

        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)

        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        # map regression parameters into anchors to get coordinate
        pred_boxes = self.decode_single(
            rel_codes.reshape(box_sum, -1), concat_boxes
        )
        return pred_boxes.reshape(box_sum, -1, 4)

    def decode_single(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets, get the decoded boxes.
        :param rel_codes: encoded boxes (bbox regression parameters)
        :param boxes: reference boxes (anchors)
        :return:
        """
        boxes = boxes.to(rel_codes.dtype)

        # xmin, ymin, xmax, ymax
        widths = boxes[:, 2] - boxes[:, 0]   # anchor width
        heights = boxes[:, 3] - boxes[:, 1]  # anchor height
        ctr_x = boxes[:, 0] + 0.5 * widths   # anchor center x coordinate
        ctr_y = boxes[:, 1] + 0.5 * heights  # anchor center y coordinate

        wx, wy, ww, wh = self.weights  # default is 1
        dx = rel_codes[:, 0::4] / wx   # predicated anchors center x regression parameters
        dy = rel_codes[:, 1::4] / wy   # predicated anchors center y regression parameters
        dw = rel_codes[:, 2::4] / ww   # predicated anchors width regression parameters
        dh = rel_codes[:, 3::4] / wh   # predicated anchors height regression parameters

        # limit max value, prevent sending too large values into torch.exp()
        # self.bbox_xform_clip=math.log(1000. / 16)
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # xmin
        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymin
        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        # xmax
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymax
        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes

