# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms import BaseTransform

from mmedit.registry import TRANSFORMS


@TRANSFORMS.register_module()
class AvgFrames(BaseTransform):
    """Average several frames into one blurred frame.

    Args:
        keys (list[str]): The frame lists to be reversed.
    """

    def __init__(self, keys) -> None:
        super().__init__()
        self.keys = keys

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            results[f'orig_{key}'] = results[key]
            results[key] = np.asarray(results[key]).mean(0).astype(np.float32)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys})'
        return repr_str
