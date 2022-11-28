# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmedit.datasets.transforms import AddNoise


def test_addnoise():
    transformer = AddNoise(
        keys=['gt'], after_key_dict={'gt': 'input'}, sigma=15)

    assert transformer.__repr__() == \
        "AddNoise, keys=['gt'], sigma=15, after_key_dict={'gt': 'input'}"
    inp = np.ones((2, 64, 64, 3))
    out = transformer.transform({'gt': inp})
    assert (out['gt'] == inp).all()
    assert out['input'].shape == inp.shape
    assert out['noise_sigma'] == 15
