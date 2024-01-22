import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .module import *

Align_Corners_Range = False


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()

    def forward(self, cost_reg, depth_values, num_depth, interval, _iter=False):
        if _iter:
            # cost_reg b 1 d h w
            b, d, h, w = cost_reg.squeeze(1).shape
            prob_volume = F.softmax(cost_reg.squeeze(1), dim=1)
            depths = winner_take_all(prob_volume, depth_values.repeat(b, 1, 1, 1))  # b h w
            photometric_confidence, idx = torch.max(prob_volume, dim=1)  # b h w
            photometric_confidence, idx = torch.max(photometric_confidence, dim=0)  # b h w
            depth = torch.gather(depths, 0, idx.unsqueeze(0))



            b,d,h,w = cost_reg.squeeze(1).shape
            prob_volume = F.softmax(cost_reg.squeeze(1), dim=1)
            depths = winner_take_all(prob_volume, depth_values.repeat(b,1,1,1))#b h w
            photometric_confidence, idx = torch.max(prob_volume, dim=1)#b h w
            photometric_confidence, idx = torch.max(photometric_confidence, dim=0)#b h w
            depth = torch.gather(depths, 0, idx.unsqueeze(0))
            photometric_confidence = photometric_confidence[None]


        else:
            prob_volume_pre = cost_reg.squeeze(1)  # (b, d, h, w)
            prob_volume = torch.exp(F.log_softmax(prob_volume_pre, dim=1))  # (b, ndepth, h, w)
            depth = winner_take_all(prob_volume, depth_values)  # (b, h, w)
            photometric_confidence, _ = torch.max(prob_volume, dim=1)

        return {"depth": depth, "photometric_confidence": photometric_confidence, "prob_volume": prob_volume,
                "depth_values": depth_values, "interval": interval}


class CostAgg(nn.Module):
    def __init__(self, in_channels=None):
        super(CostAgg, self).__init__()
        self.prob = nn.ModuleList(nn.Conv3d(in_channels[i], 1, 3, stride=1, padding=1, bias=False) for i in range(len(in_channels)) )
    def forward(self, features, proj_matrices, depth_values, stage_idx):
        """
        :param stage_idx: stage
        :param features: [ref_fea, src_fea1, src_fea2, ...], fea shape: (b, c, h, w)
        :param proj_matrices: (b, nview, ...) [ref_proj, src_proj1, src_proj2, ...]
        :param depth_values: (b, ndepth, h, w)
        :return: matching cost volume (b, c, ndepth, h, w)
        """
        ref_feature, src_features = features[0], features[1:]
        proj_matrices = torch.unbind(proj_matrices, 1)  # to list
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        num_views = len(features)
        num_depth = depth_values.shape[1]

        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)

        volume_adapt = None

        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
            warped_volume = (ref_volume * warped_volume)  # bc dhw
            pro = self.prob[stage_idx](warped_volume)
            if volume_adapt is None:
                volume_adapt = pro
            else:
                volume_adapt = torch.cat([volume_adapt, pro], dim=1)

            del warped_volume

        return volume_adapt


class MVSNet(nn.Module):
    def __init__(self, ndepths, depth_interval_ratio, train_view=None):
        super(MVSNet, self).__init__()

        self.ndepths = ndepths
        self.depth_interval_ratio = depth_interval_ratio
        self.num_stage = len(ndepths)
        self.train_view = train_view

        print("ndepths:", ndepths)
        print("depth_intervals_ratio:", depth_interval_ratio)

        assert len(ndepths) == len(depth_interval_ratio)

        self.feature = FeatureNet(base_channels=8)
        self.cost_aggregation = CostAgg(self.feature.out_channels)

        self.cost_regularization = nn.ModuleList(
            [CostRegNet(in_channels=(self.train_view - 1), base_channels=8) for i in range(self.num_stage)])

        self.DepthNet = DepthNet()

    def forward(self, imgs, proj_matrices, depth_values):
        """
        :param is_flip: augment only for 3D-UNet
        :param imgs: (b, nview, c, h, w)
        :param proj_matrices:
        :param depth_values:
        :return:
        """
        depth_interval = (depth_values[0, -1] - depth_values[0, 0]) / depth_values.size(1)

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):  # imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        ori_shape = imgs[:, 0].shape[2:]  # (H, W)

        outputs = {}
        last_depth = None
        for stage_idx in range(self.num_stage):

            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = 2 ** (3 - stage_idx - 1)

            stage_shape = [ori_shape[0] // int(stage_scale), ori_shape[1] // int(stage_scale)]

            if stage_idx == 0:
                last_depth = depth_values
            else:
                last_depth = last_depth.detach()

            # (B, D, H, W)
            depth_range_samples, interval = get_depth_range_samples(last_depth=last_depth,
                                                                    ndepth=self.ndepths[stage_idx],
                                                                    depth_inteval_pixel=self.depth_interval_ratio[
                                                                                            stage_idx] * depth_interval,
                                                                    shape=stage_shape  # only for first stage
                                                                    )

            if stage_idx > 0:
                depth_range_samples = F.interpolate(depth_range_samples, stage_shape, mode='bilinear', align_corners=Align_Corners_Range)

            # (b, c, d, h, w)
            cost_volume = self.cost_aggregation(features_stage, proj_matrices_stage, depth_range_samples, stage_idx)

            test_view = cost_volume.shape[1] + 1
            #print(test_view, self.train_view)
            _iter = False
            if test_view > self.train_view:
                cost_list = []
                cost_front = cost_volume[:, 0:self.train_view - 2, :, :, :]
                for i in range(test_view - self.train_view + 1):
                    cost_this = torch.cat((cost_front, cost_volume[:, self.train_view - 2 + i, :, :, :].unsqueeze(1)), dim=1)  # 1 train d h w
                    cost_list.append(cost_this)
                cost_volume = torch.cat(cost_list, dim=0)#n c d h w
                _iter = True
            elif test_view < self.train_view:
                res = self.train_view - test_view
                cost_0 = cost_volume[:, 0, None, :, :, :].repeat(1, res, 1, 1, 1)  # b 1 d h w
                cost_volume = torch.cat([cost_volume, cost_0], dim=1)
            cost_reg = self.cost_regularization[stage_idx](cost_volume)

            # depth
            outputs_stage = self.DepthNet(cost_reg,
                                          depth_range_samples,
                                          num_depth=self.ndepths[stage_idx],
                                          interval=interval, _iter=_iter)

            last_depth = outputs_stage['depth']
            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs
