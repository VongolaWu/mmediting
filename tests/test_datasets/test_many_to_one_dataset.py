# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.datasets import ManyToOneDataset


def test_manytomanydataset():
    dataset_cfg = dict(
        metainfo=dict(dataset_type='Allsets_air', task_name='restore'),
        data_root='tests/data/image',
        data_prefix=dict(gt='gt', img='lq'),
        search_key='img',
        pipeline=[])
    dataset = ManyToOneDataset(**dataset_cfg)
    path_list = dataset.load_data_list()
    assert len(path_list) == 1
    assert path_list[0] == {
        'gt_path': 'tests/data/image/gt/baboon.png',
        'img_path': 'tests/data/image/lq/baboon_x4.png',
        'key': 'baboon_x4'
    }
