import torch.nn as nn
import torch
from ....ops.pointnet2.pointnet2_batch import pointnet2_utils


class HeightCompression(nn.Module):
    def __init__(self, model_cfg,num_frames, **kwargs):
        super().__init__()
        self.num_frames=num_frames
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_fc = nn.Linear(96, 64, bias=False)
        self.point_cls = nn.Linear(64, 1, bias=False)
        self.point_reg = nn.Linear(64, 3, bias=False)

    def tensor2points(self, tensor, offset=(0., -40., -3.), voxel_size=(.05, .05, .1)):
        indices = tensor.indices.float()
        offset = torch.Tensor(offset).to(indices.device)
        voxel_size = torch.Tensor(voxel_size).to(indices.device)
        indices[:, 1:] = indices[:, [3, 2, 1]] * voxel_size + offset + .5 * voxel_size
        return tensor.features, indices

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        if self.training:
            voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
            # print(voxel_coords.shape)
            N, _ = voxel_features.size()
            bs_num = N - ((voxel_coords[:, 0]).sum()).int()
            
            points_mean_1 = voxel_features[:bs_num, :][:, :3].view(1, -1 ,3).contiguous()
            points_mean_2 = voxel_features[bs_num:, :][:, :3].view(1, -1 ,3).contiguous()
            # # print(voxel_features.shape)
            # # exit()

            vx_feat, vx_nxyz = self.tensor2points(batch_dict['multi_scale_3d_features']['x_conv2'], (0, -40., -3.), voxel_size=(.1, .1, .2))
            
            M, _ = vx_nxyz.size()
            bs_num_M = M - ((vx_nxyz[:, 0]).sum()).int()
            vx_feat_1 = vx_feat[:bs_num_M, :].view(1, 32, -1).contiguous()
            vx_feat_2 = vx_feat[bs_num_M:, :].view(1, 32, -1).contiguous()
            vx_nxyz_1 = vx_nxyz[:bs_num_M, :][:, 1:].view(1, -1 ,3).contiguous()
            vx_nxyz_2 = vx_nxyz[bs_num_M:, :][:, 1:].view(1, -1 ,3).contiguous()
            # # voxel聚合点的特征
            p0_1 = nearest_neighbor_interpolate(points_mean_1, vx_nxyz_1, vx_feat_1).view(1, -1, 32)
            p0_2 = nearest_neighbor_interpolate(points_mean_2, vx_nxyz_2, vx_feat_2).view(1, -1, 32)
          
            
            
            vx_feat, vx_nxyz = self.tensor2points(batch_dict['multi_scale_3d_features']['x_conv3'], (0, -40., -3.), voxel_size=(.2, .2, .4))

            M, _ = vx_nxyz.size()
            bs_num_M = M - ((vx_nxyz[:, 0]).sum()).int()
            vx_feat_1 = vx_feat[:bs_num_M, :].view(1, 32, -1).contiguous()
            vx_feat_2 = vx_feat[bs_num_M:, :].view(1, 32, -1).contiguous()
            vx_nxyz_1 = vx_nxyz[:bs_num_M, :][:, 1:].view(1, -1 ,3).contiguous()
            vx_nxyz_2 = vx_nxyz[bs_num_M:, :][:, 1:].view(1, -1 ,3).contiguous()
            p1_1 = nearest_neighbor_interpolate(points_mean_1, vx_nxyz_1, vx_feat_1).view(1, -1, 32)
            p1_2 = nearest_neighbor_interpolate(points_mean_2, vx_nxyz_2, vx_feat_2).view(1, -1, 32)
          

            # #p1 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)
            
            vx_feat, vx_nxyz = self.tensor2points(batch_dict['multi_scale_3d_features']['x_conv4'], (0, -40., -3.), voxel_size=(.4, .4, .8))
            
            M, _ = vx_nxyz.size()
            bs_num_M = M - ((vx_nxyz[:, 0]).sum()).int()
            vx_feat_1 = vx_feat[:bs_num_M, :].view(1, 32, -1).contiguous()
            vx_feat_2 = vx_feat[bs_num_M:, :].view(1, 32, -1).contiguous()
            vx_nxyz_1 = vx_nxyz[:bs_num_M, :][:, 1:].view(1, -1 ,3).contiguous()
            vx_nxyz_2 = vx_nxyz[bs_num_M:, :][:, 1:].view(1, -1 ,3).contiguous()
            p2_1 = nearest_neighbor_interpolate(points_mean_1, vx_nxyz_1, vx_feat_1).view(1, -1, 32)
            p2_2 = nearest_neighbor_interpolate(points_mean_2, vx_nxyz_2, vx_feat_2).view(1, -1, 32)
            # #for p2_1, 2 denote scale, 1 denote batch_size
            # #p2 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)
            pointwise_1 = self.point_fc(torch.cat([p0_1, p1_1, p2_1], dim=-1))
            pointwise_2 = self.point_fc(torch.cat([p0_2, p1_2, p2_2], dim=-1))
            point_cls_1 = self.point_cls(pointwise_1)
            point_reg_1 = self.point_reg(pointwise_2)
            point_cls_2 = self.point_cls(pointwise_2)
            point_reg_2 = self.point_reg(pointwise_2)
            batch_dict['pred_hm_1'] = point_cls_1
            #batch_dict['point_reg_1'] = point_reg_1
            batch_dict['pred_hm_2'] = point_cls_2
            #batch_dict['point_reg_2'] = point_reg_2
            batch_dict['points_mean_1'] = points_mean_1
            batch_dict['points_mean_2'] = points_mean_2

        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        if self.num_frames>1:
            for i in range(self.num_frames-1):
                if 'encoded_spconv_tensor'+str(-i-1) in batch_dict:
                    encoded_spconv_tensor = batch_dict['encoded_spconv_tensor'+str(-i-1)]
                    spatial_features = encoded_spconv_tensor.dense()
                    N, C, D, H, W = spatial_features.shape
                    spatial_features = spatial_features.view(N, C * D, H, W)
                    batch_dict['spatial_features'+str(-i-1)] = spatial_features


        return batch_dict

def nearest_neighbor_interpolate(unknown, known, known_feats):
    """
    :param pts: (n, 4) tensor of the bxyz positions of the unknown features
    :param ctr: (m, 4) tensor of the bxyz positions of the known features
    :param ctr_feats: (m, C) tensor of features to be propigated
    :return:
        new_features: (n, C) tensor of the features of the unknown features
    """
    dist, idx = pointnet2_utils.three_nn(unknown, known)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm
    interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)

    return interpolated_feats

