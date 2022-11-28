# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
from mmcv.transforms import BaseTransform

from mmedit.registry import TRANSFORMS


@TRANSFORMS.register_module()
class AddNoise(BaseTransform):
    """Add noise to input images for training.

    Args:
        keys (Sequence[str]): The images to be cropped.
        after_key_dict (dict[str]): the corresponding new keys
        sigma (int): Added noise sigma.
    """

    def __init__(self, keys, after_key_dict, sigma):
        self.keys = keys
        self.sigma = sigma
        self.after_key_dict = after_key_dict

    def _add_gaussian_noise(self, clear_img, sigma):
        noise = np.random.randn(*clear_img.shape)
        noisy_img = np.clip(clear_img + noise * sigma, 0, 255).astype(np.uint8)
        return noisy_img

    def transform(self, results):
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        for k in self.keys:
            noisy_data = self._add_gaussian_noise(results[k], self.sigma)
            results[self.after_key_dict[k]] = noisy_data
        results['noise_sigma'] = self.sigma
        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (f', keys={self.keys}, '
                     f'sigma={self.sigma}, '
                     f'after_key_dict={self.after_key_dict}')

        return repr_str
