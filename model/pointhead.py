import torch
import torch.nn as nn
from torch import Tensor
from torch.jit.annotations import Dict, List, Tuple
from torch.nn.functional import grid_sample
from .embedding import PositionalEncoding
from torchvision.models.detection import _utils as det_utils
from .utils import encode_boxes_point, encode_boxes_keys
from torchvision.ops import sigmoid_focal_loss

def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res

class PointHead(nn.Module):
    def __init__(self, num_anchors, mode):
        super().__init__()
        # ensure the anchor cal
        self.mode = mode
        # for calculate the score and offset
        self.conv = nn.Sequential(*[conv_block(256, 256), conv_block(256, 256)])
        self.score = nn.Conv2d(256,  num_anchors, 3, padding=1)
        self.offset = nn.Conv2d(256,  num_anchors * 2, 3, padding=1)
        self.score_max = nn.MaxPool2d(3, stride=1, padding=1)
        self.proposal_matcher = det_utils.Matcher

        # for decode the things OR might be not that easy
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    def compute_loss(self, targets, keys, offsets, anchors, matched_idxs):
        offset_losses = []
        key_losses = []
        for targets_per_image, keys_per_image, offsets_per_image, anchors_per_image, matched_idxs_per_image in \
            zip(targets, keys, offsets, anchors, matched_idxs):

            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

             # get the targets corresponding GT for each proposal
            matched_gt_boxes_per_image = targets_per_image['boxes'][matched_idxs_per_image.clamp(min=0)]

            # select only the foreground boxes for ground truth
            matched_gt_boxes_per_image = matched_gt_boxes_per_image[foreground_idxs_per_image, :]
            # select only the foreground boxes for offsets and keys
            offsets_per_image = offsets_per_image[foreground_idxs_per_image, :]
            # select only the foreground boxes for anchors
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # compute the offsets targets
            target_regression = encode_boxes_point(matched_gt_boxes_per_image, anchors_per_image, self.mode)
            # create the point ground truth
            # keys_regression = encode_boxes_keys(matched_gt_boxes_per_image, anchors_per_image, self.mode)
            gt_classes_target = torch.zeros_like(keys_per_image)
            gt_classes_target[foreground_idxs_per_image, :] = 1.0
            valid_idxs_per_image = matched_idxs_per_image != -2
            # compute the loss
            offset_losses.append(torch.nn.functional.smooth_l1_loss(
            offsets_per_image,
            target_regression, 
            reduction = 'mean') / max(1, num_foreground))
            key_loss = sigmoid_focal_loss(
            keys_per_image[valid_idxs_per_image],
            gt_classes_target[valid_idxs_per_image],
            reduction='mean',)  # / max(1, num_foreground)
            key_losses.append(key_loss)

        return (_sum(key_losses) / max(1, len(targets))), (_sum(offset_losses) / max(1, len(targets)))


    def forward(self, x, anchors):
        all_keys = []
        all_offsets = []
        grid_sizes = []
        for feature in x:
            feature = self.conv(feature)
            #add grid size
            grid_sizes.append([feature.shape[-2],feature.shape[-1]])

            #calculate the scores
            # score = torch.sigmoid(self.score(feature)) # with sigmoid
            score = self.score(feature) # without sigmoid
            N = score.shape[0]
            key = self.score_max(score).view(N, -1 , 1)
            offset = torch.sigmoid(self.offset(feature)).view(N, -1, 2)

            all_keys.append(key)
            all_offsets.append(offset)

        # concat together
        all_keys = torch.cat(all_keys, dim=1) # N, H*W, 1
        # mask = torch.zeros(all_keys.shape)
        all_offsets = torch.cat(all_offsets, dim=1)
        _, sel_indices = torch.topk(all_keys, 50, dim = 1)
        # sel_indices.shape = (N, 50, 1)
        # sel_indices = sel_indices
        return all_keys, all_offsets, sel_indices
        # all_keys = []
        # for anchors_per_image, sel_indices_per_image, offsets_per_image in zip(anchors, sel_indices, all_offsets):
        #     key_per_image = anchors_per_image[sel_indices_per_image.flatten()]
        #     offsets = offsets_per_image[sel_indices_per_image.flatten()]
        #     # calculate is it corner or center
        #     if self.mode == 'center':
        #         key_per_image[:, 0] = (key_per_image[:, 0] + key_per_image[:, 2]) / 2
        #         key_per_image[:, 1] = (key_per_image[:, 1] + key_per_image[:, 3]) / 2
        #         all_keys.append(key_per_image[:, :2] + offsets)
        #     elif self.mode == 'upleftcorner':
        #         all_keys.append(key_per_image[:, :2] + offsets)
        #     elif self.mode == 'rightbotcorner':
        #         all_keys.append(key_per_image[:, 2:4] + offsets)

        # #concat together
        # all_keys = torch.tensor([item.cpu().detach().numpy() for item in all_keys]) # shape = (N, 50, 2)
        # #### for testing ####
        # all_keys = torch.randn(all_keys.shape)
        # # anchors shape = (N, 163206, 2)
        # #calculate cosine distance, output should be (163206, 50)

        # print('Before encode pos : {}'.format(all_keys.shape))
        # all_keys = self.pos_encoder(all_keys)
        # print('After encode pos : {}'.format(all_keys.shape))

        # anchors = torch.tensor([item.cpu().detach().numpy() for item in anchors])
        # anchors = anchors[:, :, :2]

        # print('Before encode pos : {}'.format(anchors.shape))
        # anchors = self.pos_encoder(anchors)
        # print('after encode pos : {}'.format(anchors.shape))
        # outputs = self.decoder()


        # mask[:, sel_indices, :] = 1
        # key_points_mask = []
        # start = 0
        # for grid in grid_sizes:
        #     H, W = grid[0], grid[1]
        #     key_points_mask.append(mask[:, start:(start + H*W), :].view(N, 1, H, W))
        #     start += H*W
        # output shape should be (N, Hout, Wout, 2)
        # return key_points_mask

def conv_block(in_channels, out_channels):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
      nn.ReLU(),
    )
