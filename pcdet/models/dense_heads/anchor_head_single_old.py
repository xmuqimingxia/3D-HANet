import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate
import torch
import cv2
import numpy as np
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

def get_layer(dim,out_dim,init = None):
    init_func = nn.init.kaiming_normal_
    layers = []
    conv = nn.Conv2d(dim, dim,
                      kernel_size=3, padding=1, bias=True)
    nn.init.normal_(conv.weight, mean=0, std=0.001)
    layers.append(conv)
    layers.append(nn.BatchNorm2d(dim))
    layers.append(nn.ReLU())
    conv2 = nn.Conv2d(dim, out_dim,
                     kernel_size=1, bias=True)

    if init is None:
        nn.init.normal_(conv2.weight, mean=0, std=0.001)
        layers.append(conv2)

    else:
        conv2.bias.data.fill_(init)
        layers.append(conv2)

    return nn.Sequential(*layers)

class AnchorHeadSingleV2(AnchorHeadTemplate):
    def __init__(self, model_cfg, num_frames, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg,num_frames=num_frames, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.grid_size = grid_size  # [1408 1600   40]
        self.range = point_cloud_range

        self.voxel_size = (point_cloud_range[3] - point_cloud_range[0]) / grid_size[0]


        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        shard_c = 64

        self.shared_conv = nn.Sequential(
            nn.Conv2d(input_channels, shard_c,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(shard_c),
            nn.ReLU(inplace=True)
        )

        self.conv_cls = get_layer(shard_c,self.num_anchors_per_location * self.num_class,-4.59)

        self.conv_reg = get_layer(shard_c,self.num_anchors_per_location * 2)
        self.conv_height = get_layer(shard_c,self.num_anchors_per_location * 1)

        self.conv_dim = get_layer(shard_c,self.num_anchors_per_location * 3)

        self.conv_ang = get_layer(shard_c,self.num_anchors_per_location * 1)

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:

            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        #self.init_weights()

        #for child in self.children():
        #    for param in child.parameters():
        #        param.requires_grad = False

    def init_weights(self):

        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def get_anchor_mask(self,data_dict,shape):

        stride = np.round(self.voxel_size*8.*10.)

        minx=self.range[0]
        miny=self.range[1]

        points = data_dict["points"]

        mask = torch.zeros(shape[-2],shape[-1])

        mask_large = torch.zeros(shape[-2]//10,shape[-1]//10)

        in_x = (points[:, 1] - minx) / stride
        in_y = (points[:, 2] - miny) / stride

        in_x = in_x.long().clamp(max=shape[-1]//10-1)
        in_y = in_y.long().clamp(max=shape[-2]//10-1)


        mask_large[in_y,in_x] = 1

        mask_large = mask_large.clone().int().detach().cpu().numpy()

        mask_large_index = np.argwhere( mask_large>0 )

        mask_large_index = mask_large_index*10

        index_list=[]

        for i in np.arange(-10, 10, 1):
            for j in np.arange(-10, 10, 1):
                index_list.append(mask_large_index+[i,j])

        index_list = np.concatenate(index_list,0)

        inds = torch.from_numpy(index_list).cuda().long()

        mask[inds[:,0],inds[:,1]]=1

        return mask.bool()

    def gauss_fun(self, points_mean, gt_boxes):

        
        gt_center = gt_boxes[:3]
        w_gt = gt_boxes[3]
        l_gt = gt_boxes[4]
        h_gt = gt_boxes[5]
        offset_gt = (points_mean - gt_center).view(1,3)
        _COVARIANCE_1 = 4/(w_gt ** 2 + l_gt ** 2)
        _COVARIANCE_2 = 4/(w_gt ** 2 + h_gt ** 2)
        _COVARIANCE_3 = 4/(h_gt ** 2 + l_gt ** 2)

        _COVARIANCE = (torch.tensor([[_COVARIANCE_1, 0., 0.],
                    [0., _COVARIANCE_2, 0.],
                    [0., 0., _COVARIANCE_3]])).cuda()
        value_matric = torch.mm(torch.mm(offset_gt, _COVARIANCE), offset_gt.t())
        #diag_value = torch.diag(value_matric)
        gt_hm = torch.exp(-0.5 * value_matric)

        return gt_hm


    def forward(self, data_dict):

        anchor_mask = self.get_anchor_mask(data_dict, data_dict['st_features_2d'].shape)

        new_anchors = []
        for anchors in self.anchors_root:
            new_anchors.append(anchors[:, anchor_mask, ...])

        self.anchors = new_anchors


        for i in range(self.num_frames):
            if i==0:
                frame_id = ''
            else:
                frame_id = str(-i)
            if 'st_features_2d'+frame_id not in data_dict:
                continue
            st_features_2d = data_dict['st_features_2d'+frame_id]

            shard = self.shared_conv(st_features_2d)

            cls_feat = shard

            reg_feat = shard

            cls_preds = self.conv_cls(cls_feat)

            box_reg = self.conv_reg(reg_feat)
            box_height = self.conv_height(reg_feat)
            box_dim = self.conv_dim(reg_feat)
            box_ang = self.conv_ang(reg_feat)

            box_preds = torch.cat([box_reg,box_height,box_dim,box_ang],dim=1)

            cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]  # [N, H, W, C]
            box_preds = box_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]  # [N, H, W, C]

            self.forward_ret_dict['cls_preds'+frame_id] = cls_preds
            self.forward_ret_dict['box_preds'+frame_id] = box_preds

            if self.conv_dir_cls is not None:
                dir_cls_preds = self.conv_dir_cls(st_features_2d)
                dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]
                self.forward_ret_dict['dir_cls_preds'+frame_id] = dir_cls_preds
            else:
                dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            ###############################################################
            gt_boxes_1 = data_dict['gt_boxes'][0,:,:][:,:7].view(1,-1,7)
            gt_boxes_2 = data_dict['gt_boxes'][1,:,:][:,:7].view(1,-1,7)
            points_mean_1 = data_dict['points_mean_1']
            points_mean_2 = data_dict['points_mean_2']
            
            point_indices_1 = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_mean_1, gt_boxes_1
            ) # (nboxes, npoints) point_indices: the point belone which box
            points_hm_1 = torch.zeros_like(point_indices_1, dtype=torch.float)
            ###
            # np.save('points.npy', points_mean_1.cpu().numpy())
            # np.save('box.npy', gt_boxes_1.cpu().numpy())
            ###
            Num_points_1 = points_mean_1.size(1)
            for i in range(Num_points_1):
                ind = point_indices_1[0, i]
                if ind != -1:
                    points_hm_1[0, i] = self.gauss_fun(points_mean_1[0,i,:], gt_boxes_1[0,ind,:]).clone()
            targets_dict['gt_hm_1'] = points_hm_1

            ###############################################################

            point_indices_2 = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_mean_2, gt_boxes_2
            ) # (nboxes, npoints) point_indices: the point belone which box
            points_hm_2 = torch.zeros_like(point_indices_2, dtype=torch.float)
            ###
            # np.save('points.npy', points_mean_1.cpu().numpy())
            # np.save('box.npy', gt_boxes_1.cpu().numpy())
            ###
            Num_points_2 = points_mean_2.size(1)
            for i in range(Num_points_2):
                ind = point_indices_2[0, i]
                if ind != -1:
                    points_hm_2[0, i] = self.gauss_fun(points_mean_2[0,i,:], gt_boxes_2[0,ind,:]).clone()
            targets_dict['gt_hm_2'] = points_hm_2

            ###############################################################
            targets_dict['pred_hm_1'] = data_dict['pred_hm_1']
            targets_dict['pred_hm_2'] = data_dict['pred_hm_2']
            ###############################################################
            self.forward_ret_dict.update(targets_dict)
            data_dict['gt_ious'] = targets_dict['gt_ious']

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, num_frames, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg,num_frames=num_frames, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.grid_size = grid_size  # [1408 1600   40]
        self.range = point_cloud_range

        self.voxel_size = (point_cloud_range[3] - point_cloud_range[0]) / grid_size[0]


        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )


        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

        #for child in self.children():
        #    for param in child.parameters():
        #        param.requires_grad = False

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def get_anchor_mask(self,data_dict,shape):

        stride = np.round(self.voxel_size*8.*10.)

        minx=self.range[0]
        miny=self.range[1]

        points = data_dict["points"]

        mask = torch.zeros(shape[-2],shape[-1])

        mask_large = torch.zeros(shape[-2]//10,shape[-1]//10)

        in_x = (points[:, 1] - minx) / stride
        in_y = (points[:, 2] - miny) / stride

        in_x = in_x.long().clamp(max=shape[-1]//10-1)
        in_y = in_y.long().clamp(max=shape[-2]//10-1)


        mask_large[in_y,in_x] = 1

        mask_large = mask_large.clone().int().detach().cpu().numpy()

        mask_large_index = np.argwhere( mask_large>0 )

        mask_large_index = mask_large_index*10

        index_list=[]

        for i in np.arange(-10, 10, 1):
            for j in np.arange(-10, 10, 1):
                index_list.append(mask_large_index+[i,j])

        index_list = np.concatenate(index_list,0)

        inds = torch.from_numpy(index_list).cuda().long()

        mask[inds[:,0],inds[:,1]]=1

        return mask.bool()


    def forward(self, data_dict):

        anchor_mask = self.get_anchor_mask(data_dict,data_dict['st_features_2d'].shape)

        new_anchors = []
        for anchors in self.anchors_root:
            new_anchors.append(anchors[:, anchor_mask, ...])

        self.anchors = new_anchors

        for i in range(self.num_frames):
            if i==0:
                st_features_2d = data_dict['st_features_2d']

                cls_preds = self.conv_cls(st_features_2d)
                box_preds = self.conv_box(st_features_2d)

                cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]  # [N, H, W, C]
                box_preds = box_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]  # [N, H, W, C]

                self.forward_ret_dict['cls_preds'] = cls_preds
                self.forward_ret_dict['box_preds'] = box_preds

                if self.conv_dir_cls is not None:
                    dir_cls_preds = self.conv_dir_cls(st_features_2d)
                    dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]
                    self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
                else:
                    dir_cls_preds = None

            else:
                if 'st_features_2d'+str(-i) not in data_dict:
                    continue
                st_features_2d = data_dict['st_features_2d'+str(-i)]

                cls_preds2 = self.conv_cls(st_features_2d)
                box_preds2 = self.conv_box(st_features_2d)


                cls_preds2 = cls_preds2.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
                box_preds2 = box_preds2.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]


                self.forward_ret_dict['cls_preds'+str(-i)] = cls_preds2
                self.forward_ret_dict['box_preds'+str(-i)] = box_preds2

                if self.conv_dir_cls is not None:
                    dir_cls_preds2 = self.conv_dir_cls(st_features_2d)
                    dir_cls_preds2 = dir_cls_preds2.permute(0, 2, 3, 1).contiguous()
                    self.forward_ret_dict['dir_cls_preds'+str(-i)] = dir_cls_preds2
                else:
                    dir_cls_preds2 = None


        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            ###############################################################
            gt_boxes_1 = data_dict['gt_boxes'][0,:,:][:,:7].view(1,-1,7)
            gt_boxes_2 = data_dict['gt_boxes'][1,:,:][:,:7].view(1,-1,7)
            points_mean_1 = data_dict['points_mean_1']
            points_mean_2 = data_dict['points_mean_2']
            
            point_indices_1 = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_mean_1, gt_boxes_1
            ) # (nboxes, npoints) point_indices: the point belone which box
            points_hm_1 = torch.zeros_like(point_indices_1, dtype=torch.float)
            ###
            # np.save('points.npy', points_mean_1.cpu().numpy())
            # np.save('box.npy', gt_boxes_1.cpu().numpy())
            ###
            Num_points_1 = points_mean_1.size(1)
            for i in range(Num_points_1):
                ind = point_indices_1[0, i]
                if ind != -1:
                    points_hm_1[0, i] = self.gauss_fun(points_mean_1[0,i,:], gt_boxes_1[0,ind,:]).clone()
            targets_dict['gt_hm_1'] = points_hm_1

            ###############################################################

            point_indices_2 = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_mean_2, gt_boxes_2
            ) # (nboxes, npoints) point_indices: the point belone which box
            points_hm_2 = torch.zeros_like(point_indices_2, dtype=torch.float)
            ###
            # np.save('points.npy', points_mean_1.cpu().numpy())
            # np.save('box.npy', gt_boxes_1.cpu().numpy())
            ###
            Num_points_2 = points_mean_2.size(1)
            for i in range(Num_points_2):
                ind = point_indices_2[0, i]
                if ind != -1:
                    points_hm_2[0, i] = self.gauss_fun(points_mean_2[0,i,:], gt_boxes_2[0,ind,:]).clone()
            targets_dict['gt_hm_2'] = points_hm_2

            ###############################################################
            targets_dict['pred_hm_1'] = data_dict['pred_hm_1']
            targets_dict['pred_hm_2'] = data_dict['pred_hm_2']
            print('##################################################')
            exit()
            ###############################################################
            self.forward_ret_dict.update(targets_dict)
            data_dict['gt_ious'] = targets_dict['gt_ious']

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        if self.model_cfg.get('NMS_CONFIG', None) is not None:
            self.proposal_layer(
                data_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            )

        return data_dict
