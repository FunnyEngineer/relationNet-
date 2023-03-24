import torch
import torch.nn as nn
from torch import Tensor
from torch.jit.annotations import Dict, List, Tuple
from torch.nn.functional import grid_sample
from .pointhead import PointHead
from .attention import ReAttention
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import boxes as box_ops
from torchvision.ops import sigmoid_focal_loss

def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res

class BVRRetina(nn.Module):
    def __init__(self, num_classes = 91):
        super().__init__()
        self.Retina = retinanet_resnet50_fpn(pretrained=False, num_classes = num_classes)
        # they sould provide the key
        self.centerHead = PointHead(self.Retina.anchor_generator.num_anchors_per_location()[0], 'center')
        self.ULcornerHead = PointHead(self.Retina.anchor_generator.num_anchors_per_location()[0], 'upleftcorner')
        # self.RBcornerHead = PointHead(self.Retina.anchor_generator.num_anchors_per_location()[0], 'rightbotcorner')


        # for calculate strengthen module
        self.centerAttn = ReAttention('center')
        self.ULcornerAttn = ReAttention('upleftcorner')
        # self.RBcornerAttn = ReAttention('rightbotcorner')

    def sim_func(input_fea, key_fea, input_geo, key_geo, mode):
      if mode == 'cls':
        Sa = self.cls_attention(input_fea, key_fea, key_fea)
        Sg = self.cls_sg(nn.CosineSimilarity(input_geo, key_geo))
        T = self.cls_T(key_fea)
      else:
        Sa = self.reg_attention(input_fea, key_fea, key_fea)
        Sg = self.reg_sg(nn.CosineSimilarity(input_geo, key_geo))
        T = self.reg_T(key_fea)
      return torch.dot(nn.Softmax(torch.cat((Sa,Sg), dim = 1)), T)

    # def createMap(original_image_sizes):
    #     for image_size in original_image_sizes:
    #         H, W = image_size[0], image_size[1]
    #         for

    # def fromKeytoFeatureandPosition(key, features, anchors, mode = 'corner'):
    #     if mode == 'corner':
    #         features =
    #         anchors = anchors[:, :2]

    def compute_loss(self, targets, head_outputs, anchors):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Dict[str, Tensor]
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image['boxes'].numel() == 0:
                matched_idxs.append(torch.empty((0,), dtype=torch.int32))
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image['boxes'], anchors_per_image)
            matched_idxs.append(self.Retina.proposal_matcher(match_quality_matrix))
        # self.compute_cls_loss(targets, head_outputs, matched_idxs)
        return self.Retina.head.compute_loss(targets, head_outputs, anchors, matched_idxs), matched_idxs

    def createGridSizes(self, x):
        grid_sizes = []
        for features in x:
            grid_sizes.append([features.shape[-2], features.shape[-1]])

    def compute_cls_loss(self, targets, head_outputs, matched_idxs):
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
                valid_idxs_per_image = matched_idxs_per_image != -2
                print(cls_logits_per_image[valid_idxs_per_image].shape)
                print(gt_classes_target[valid_idxs_per_image].shape)
                loss = sigmoid_focal_loss(
                cls_logits_per_image[valid_idxs_per_image],
                gt_classes_target[valid_idxs_per_image],
                reduction='sum',
            ) / max(1, num_foreground)
                print(loss)
                import pdb; pdb.set_trace()
            # compute the classification loss
            losses.append(loss)

        return _sum(losses) / len(targets)

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))


        # transform the input
        images, targets = self.Retina.transform(images, targets)

        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        # get the features from the backbone
        features = self.Retina.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        features = list(features.values())

        # create the set of anchors
        anchors = self.Retina.anchor_generator(images, features)

        # compute the retinanet heads outputs using the features
        head_outputs = self.Retina.head(features)

        # return all_keys, all_offsets, sel_indices
        centner_keys, center_offsets, center_indices = self.centerHead(features, anchors)
        ULcorner_keys, ULcorner_offsets, ULcorner_indices = self.ULcornerHead(features, anchors)
        # RBcorner_keys, RBcorner_offsets, RBcorner_indices = self.RBcornerHead(features, anchors)

        # make grid size
        grid_sizes = self.createGridSizes(features)

        # attention and get strenthen features
        head_outputs['cls_logits'] = self.centerAttn(head_outputs['cls_logits'], center_indices,  anchors, center_offsets)
        head_outputs['bbox_regression'] = self.ULcornerAttn(head_outputs['bbox_regression'], ULcorner_indices,  anchors, ULcorner_offsets)

        # create strengthen features
        # head_outputs['cls_logits'] = head_outputs['cls_logits'] + sim_func(head_outputs['cls_logits'], cen_val, anchors, cen_val, 'cls')
        # # TODO: Add PointHead network
        # cen_key, cen_val = self.center_ph(features)
        # # geometic -> point
        # # need to prepare those things
        # # input feature, key feature, input geometric, key geometric, mode
        # # head_outputs['cls_logits'].shape = (163206, 91)
        # # key_feature 50
        # new_cls_fea = head_outputs['cls_logits'] + sim_func(head_outputs['cls_logits'], cen_val, anchors, cen_val, 'cls')
        # cor_key, cor_val = self.corner_ph(features)
        # new_reg_fea = head_outputs['bbox_regression'] + sim_func(head_outputs['bbox_regression'], cor_val, anchors, cor_val, 'reg')

        # head_outputs['cls_logits'] = new_cls_fea
        # head_outputs['bbox_regression'] = new_reg_fea

        losses = {}
        detections = torch.jit.annotate(List[Dict[str, Tensor]], [])
        if self.training:
            assert targets is not None

            # compute the losses
            losses, matched_idxs= self.compute_loss(targets, head_outputs, anchors)
            centerKeyLoss, centerOffsetLoss = self.centerHead.compute_loss(targets, centner_keys, center_offsets, anchors, matched_idxs)
            cornerKeyLoss, cornerOffsetLoss = self.ULcornerHead.compute_loss(targets, ULcorner_keys, ULcorner_offsets, anchors, matched_idxs)
            losses['centerKeyLoss'] = centerKeyLoss
            losses['centerOffsetLoss'] = centerOffsetLoss
            losses['cornerKeyLoss'] = cornerKeyLoss
            losses['cornerOffsetLoss'] = cornerOffsetLoss
        else:
            # compute the detections
            detections = self.Retina.postprocess_detections(head_outputs, anchors, images.image_sizes)
            detections = self.Retina.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        if torch.jit.is_scripting():
            if not self.Retina._has_warned:
                warnings.warn("RetinaNet always returns a (Losses, Detections) tuple in scripting")
                self.Retina._has_warned = True
            return (losses, detections)
        return self.Retina.eager_outputs(losses, detections)
