# for regression box -> offset loss
def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
    # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Tensor
    losses = []

    bbox_regression = head_outputs['bbox_regression']

    for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in \
            zip(targets, bbox_regression, anchors, matched_idxs):
        # no matched_idxs means there were no annotations in this image
        # TODO enable support for images without annotations with distributed support
        # if matched_idxs_per_image.numel() == 0:
        #     continue

        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_gt_boxes_per_image = targets_per_image['boxes'][matched_idxs_per_image.clamp(min=0)]

        # determine only the foreground indices, ignore the rest
        foreground_idxs_per_image = matched_idxs_per_image >= 0
        num_foreground = foreground_idxs_per_image.sum()

        # select only the foreground boxes
        matched_gt_boxes_per_image = matched_gt_boxes_per_image[foreground_idxs_per_image, :]
        bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
        anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

        # compute the regression targets
        target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)

        # compute the loss
        losses.append(torch.nn.functional.l1_loss(
            bbox_regression_per_image,
            target_regression,
            size_average=False
        ) / max(1, num_foreground))

    return _sum(losses) / max(1, len(targets))


# classfication loss
def compute_loss(self, targets, head_outputs, matched_idxs):
    # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
    losses = []

    cls_logits = head_outputs['cls_logits']

    for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
        # determine only the foreground
        foreground_idxs_per_image = matched_idxs_per_image >= 0
        num_foreground = foreground_idxs_per_image.sum()
        # no matched_idxs means there were no annotations in this image
        # TODO: enable support for images without annotations that works on distributed
        if False:  # matched_idxs_per_image.numel() == 0:
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            valid_idxs_per_image = torch.arange(cls_logits_per_image.shape[0])
        else:
            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image['labels'][matched_idxs_per_image[foreground_idxs_per_image]]
            ] = 1.0

            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

        # compute the classification loss
        losses.append(sigmoid_focal_loss(
            cls_logits_per_image[valid_idxs_per_image],
            gt_classes_target[valid_idxs_per_image],
            reduction='sum',
        ) / max(1, num_foreground))

    return _sum(losses) / len(targets)