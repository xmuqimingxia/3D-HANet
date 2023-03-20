from functools import partial

import numpy as np

from ...utils import common_utils
from . import augmentor_utils, database_sampler


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None, num_frames=1):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger
        self.num_frames=num_frames

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger,
            num_frames=self.num_frames
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
   
    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        aug_param=[]
        if self.num_frames==1:
            gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
            for cur_axis in config['ALONG_AXIS_LIST']:
                assert cur_axis in ['x', 'y']
                gt_boxes, points, param = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                    gt_boxes, points,
                )
                aug_param.append(int(param))

            data_dict['gt_boxes'] = gt_boxes
            data_dict['points'] = points
            data_dict['aug_param'] = aug_param
        else:
            enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
            gt_boxes, gt_tracklets = data_dict['gt_boxes'],data_dict['gt_tracklets']
            for cur_axis in config['ALONG_AXIS_LIST']:
                assert cur_axis in ['x']
                gt_boxes =  getattr(augmentor_utils, 'random_flip_with_param')(
                    gt_boxes, enable, ax=1)
                gt_boxes = getattr(augmentor_utils, 'random_flip_with_param')(
                        gt_boxes, enable, ax=6)
                data_dict['gt_boxes'] = gt_boxes
                for i in range(self.num_frames):
                    if i==0:
                        gt_tracklets=getattr(augmentor_utils, 'random_flip_with_param')(
                            gt_tracklets, enable, ax=1)
                        gt_tracklets = getattr(augmentor_utils, 'random_flip_with_param')(
                            gt_tracklets, enable, ax=6)
                        points=data_dict['points']
                        points = getattr(augmentor_utils, 'random_flip_with_param')(
                            points, enable, ax=1)
                        data_dict['points']=points
                    else:
                        if 'points'+str(-i) in data_dict:
                            gt_tracklets=getattr(augmentor_utils, 'random_flip_with_param')(
                                gt_tracklets, enable, ax=i*4+4)
                            gt_tracklets = getattr(augmentor_utils, 'random_flip_with_param')(
                                gt_tracklets, enable, ax=i * 4 + 6)
                            points = data_dict['points'+str(-i)]
                            points = getattr(augmentor_utils, 'random_flip_with_param')(
                                points, enable, ax=1)
                            data_dict['points' + str(-i)]=points

                        if 'gt_boxes' + str(-i) in data_dict:
                            data_dict['gt_boxes' + str(-i)] = getattr(augmentor_utils, 'random_flip_with_param')(
                                data_dict['gt_boxes' + str(-i)], enable, ax=1)
                            data_dict['gt_boxes' + str(-i)] = getattr(augmentor_utils, 'random_flip_with_param')(
                                data_dict['gt_boxes' + str(-i)], enable, ax=6)

                data_dict['gt_tracklets']=gt_tracklets


        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        if self.num_frames==1:
            gt_boxes, points, param = augmentor_utils.global_rotation(
                data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
            )

            data_dict['gt_boxes'] = gt_boxes
            data_dict['points'] = points
            if 'aug_param' in data_dict:
                data_dict['aug_param'].append(param)
            else:
                data_dict['aug_param'] = [param]

        else:
            noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
            data_dict=augmentor_utils.global_rotation_with_param(data_dict,noise_rotation,self.num_frames)

        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        if self.num_frames==1:
            gt_boxes, points, param = augmentor_utils.global_scaling(
                data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
            )
            data_dict['gt_boxes'] = gt_boxes
            data_dict['points'] = points
            if 'aug_param' in data_dict:
                data_dict['aug_param'].append(param)
            else:
                data_dict['aug_param'] = [param]

            return data_dict
        else:
            scale_range=config['WORLD_SCALE_RANGE']
            noise_scale = np.random.uniform(scale_range[0], scale_range[1])
            data_dict =  augmentor_utils.global_scaling_with_param(data_dict,noise_scale,self.num_frames)
            return data_dict

    def random_local_pyramid_aug(self, data_dict=None, config=None):
        """
        Refer to the paper:
            SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud
        """
        if data_dict is None:
            return partial(self.random_local_pyramid_aug, config=config)

        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']

        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_dropout(gt_boxes, points, config['DROP_PROB'])
        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_sparsify(gt_boxes, points,
                                                                            config['SPARSIFY_PROB'],
                                                                            config['SPARSIFY_MAX_NUM'],
                                                                            pyramids)
        gt_boxes, points = augmentor_utils.local_pyramid_swap(gt_boxes, points,
                                                              config['SWAP_PROB'],
                                                              config['SWAP_MAX_NUM'],
                                                              pyramids)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...
        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'aug_param' in data_dict:
            data_dict['aug_param'] = np.array(data_dict['aug_param'])
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')

        return data_dict
