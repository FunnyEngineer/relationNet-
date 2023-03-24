import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import PositionalEncoding

def transferTo2d(anchors, mode):
    if mode == 'center':
        new_anchors = []
        for anchor in anchors:
            anchor[:, 0] = (anchor[:, 0] + anchor[:, 2]) / 2
            anchor[:, 1] = (anchor[:, 1] + anchor[:, 3]) / 2
            new_anchors.append(anchor[:, :2])
        return new_anchors
    elif mode == 'upleftcorner':
        new_anchors = [anchor[:, :2] for anchor in anchors]
        return new_anchors
    elif mode == 'rightbotcorner':
        new_anchors = [anchor[:, 2:4] for anchor in anchors]
        return new_anchors
class ReAttention(nn.Module):

    def __init__(self, mode):
        super(ReAttention, self).__init__()
        self.mode = mode
        self.Sa = nn.MultiheadAttention(91, 1)
        self.Sg = nn.Sequential(
            nn.Linear(8, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
        )
        if mode == 'center':
            self.Tfunc = nn.Linear((91+512), 91)
            self.Sa = nn.MultiheadAttention(91, 1)
        else:
            self.Tfunc = nn.Linear((4+512), 4)
            self.Sa = nn.MultiheadAttention(4, 1)

        # position eocoding
        self.pos_encoder = PositionalEncoding(102)
        self.pos_encoder2 = nn.Sequential(
            nn.Linear(102, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 8),
        )
        self.softmax = nn.Softmax(dim=2)
    def forward(self, features, keys,  anchors, all_offsets):
        ## features = (N , 163206, 91 or 4)
        ## keys = (N, 50, 1) which is index (int))
        N = len(features)

        # Sa part
        key_feature = []
        for features_per_image, keys_per_image in zip(features, keys):
            key_feature.append(features_per_image[keys_per_image.flatten()])
        key_feature = torch.cat(key_feature, dim = 0).view(N, 50, -1) # shape = (N, 50, 91 or 4)

        #adjust shape
        key_feature = key_feature.permute(1, 0, 2) # 50, N, 91or4
        features = features.permute(1, 0, 2) #163206, N,

        # Sq part
        # input position -> (N, 163206, 4) turn into (N, 163206, 2)
        anchors = transferTo2d(anchors, self.mode)
        anchors = torch.cat(anchors, dim=0).view(N, -1, 2)

        # find key position -> (N, 50, 2)
        key_position = []
        for keys_per_image, pos_per_image, offsets_per_image in zip(keys, anchors, all_offsets):
            key_position.append(pos_per_image[keys_per_image.flatten()] - offsets_per_image[keys_per_image.flatten()])
        key_position = torch.cat(key_position, dim=0).view(N, -1, 2)
        key_position = key_position.view(N, 1, -1).repeat(1, anchors.shape[1], 1) # N, 100
        sa_output, _ = self.Sa(features, key_feature, key_feature)

        # calculate G channel geo map
        anchors = torch.cat((anchors, key_position), dim = 2) # N, 163206, 102
        anchors = self.pos_encoder(anchors) # cosine/sine embedding
        anchors = self.pos_encoder2(anchors) # GeoMap -> N, 163206, 8
        key_geo = []
        # for anchors_per_image, keys_per_image in zip(anchors, keys):
        #     key_geo.append(anchors_per_image[keys_per_image.flatten()])
        # key_geo = torch.cat(key_geo, dim = 0) # N, 50, 8
        # bilinear sampling
        anchors = self.Sg(anchors) # N, 163206, 512
        total_S = self.softmax(torch.cat((sa_output.permute(1, 0, 2), anchors), dim = 2))
        enhanced = self.Tfunc(total_S) # M, 163206, 91+512
        return features.permute(1, 0, 2) + enhanced
