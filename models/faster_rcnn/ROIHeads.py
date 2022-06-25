import torch
import torch.nn.functional as F

from torch.jit.annotations import List, Dict, Tuple

from box_op import *
from utils import *


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Faster R-CNN.
    :param class_logits: predicted class, shape=[num_anchors, num_classes]
    :param box_regression: predicted bbox regression
    :param labels: true label
    :param regression_targets: true bbox
    :return: classification_loss (Tensor)
             box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)

    labels_pos = labels[sampled_pos_inds_subset]

    # shape=[num_proposal, num_classes]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = smooth_l1_loss(box_regression[sampled_pos_inds_subset, labels_pos],
                              regression_targets[sampled_pos_inds_subset],
                              beta=1 / 9,
                              size_average=False,
                              ) / labels.numel()

    return classification_loss, box_loss


def add_gt_proposals(proposals, gt_boxes):
    """
    concate gt_box and proposals
    :param proposals: bboxes of predicted by rpn
    :param gt_boxes: true bbox
    :return:
    """

    proposals = [
        torch.cat((proposal, gt_box))
        for proposal, gt_box in zip(proposals, gt_boxes)
    ]
    return proposals


def check_targets(targets):
    assert targets is not None
    assert all(["boxes" in t for t in targets])
    assert all(["labels" in t for t in targets])


class RoIHeads(torch.nn.Module):
    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_predictor,

                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,

                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detection_per_img):
        super(RoIHeads, self).__init__()

        self.box_similarity = box_iou

        # assign ground-truth boxes for each proposal
        self.proposal_matcher = Matcher(
            fg_iou_thresh,  # 0.5
            bg_iou_thresh,  # 0.5
            allow_low_quality_matches=False)

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image,  # 512
            positive_fraction)  # 0.25

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detection_per_img = detection_per_img

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        """
        get the matched gt_bbox for every anchors, and set positive/negative samples
        :param proposals:
        :param gt_boxes:
        :param gt_labels:
        :return:
        """

        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:
                # background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                # iou of bbox and anchors
                match_quality_matrix = box_iou(gt_boxes_in_image, proposals_in_image)

                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = 0

                # label ignore proposals (between low and high threshold)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def select_training_samples(self,
                                proposals,
                                targets
                                ):

        check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposal
        proposals = add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)

        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(self,
                               class_logits,
                               box_regression,
                               proposals,
                               image_shapes
                               ):
        """
        Post-processing the network’s prediction data, including
        (1) Calculate the final bbox coordinates according to the proposal and the predicted regression parameters
        (2) Perform softmax processing on the predicted category results
        (3) Crop the predicted boxes information and adjust the out-of-bounds coordinates to the picture boundary
        (4) Remove all background information
        (5) Remove low-probability targets
        (6) Remove small size targets
        (7) Perform nms processing and sort by scores
        (8) Return the top topk targets according to the sort of scores
        Args:
            class_logits: network prediction class probability information
            box_regression: bounding box regression parameters predicted by the network
            proposals: proposals output by rpn
            image_shapes: The width and height of each image before being packaged into a batch
        Returns:
        """
        device = class_logits.device
        # Number of predicted target classes
        num_classes = class_logits.shape[-1]

        # Get the number of predicted bboxes per image
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        # Calculate the final bbox coordinates according to the proposal and the predicted regression parameters
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # Softmax the predicted category results
        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        # Traverse each image prediction information
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            # Crop the predicted boxes information and adjust the out-of-bounds coordinates to the picture boundary
            boxes = clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove prediction with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            # 
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            # Remove low probability targets, self.scores_thresh=0.05
            keep = remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximun suppression, independently done per class
            # Execute nms processing, the results after execution will be sorted and returned according to the scores from large to small
            keep = batched_nms(boxes, scores, labels, self.nms_thresh)

            # keep only topk scoring predictions
            # Get the top topk prediction targets in scores
            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self,
                features,
                proposals,
                image_shapes,
                targets=None
                ):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """

        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                # assert t["labels"].dtype == torch.int64, "target labels must of int64 type"

        if self.training:
            # Divide positive and negative samples, count the labels corresponding to gt and bounding box regression information
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        # Pass the collected samples through the roi_pooling layer
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        # Two-layer fully connected layer after roi_pooling
        box_features = self.box_head(box_features)
        # Then predict the target category and bounding box regression parameters separately
        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, losses