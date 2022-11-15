# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import List

from mmedit.datasets import BasicImageDataset
from mmedit.registry import DATASETS

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


@DATASETS.register_module()
class ManyToOneDataset(BasicImageDataset):
    """Many to one datasets for synthetic image datasets with: many synthetic
    input images with the same gt image.

    It assumes that the training directory is '/path/to/data/train'.
    During test time, the directory is '/path/to/data/test'.
    '/path/to/data' can be initialized by args 'dataroot'.
    Each sample contains a pair of
    images concatenated in the w dimension (A|B).

    Note:
        This is just a hard coding dataset class.
        We assume input_img.split('_')[0] is the name of gt.

    Args:
        dataroot (str | :obj:`Path`): Path to the folder root of paired images.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(img=None, ann=None).
        pipeline (List[dict | callable]): A sequence of data transformations.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
        filename_tmpl (dict): Template for each filename. Note that the
            template excludes the file extension. Default: dict().

        Note:

        Assume the file structure as the following:

        .. code-block:: none

            mmediting (root)
            ├── mmedit
            ├── tools
            ├── configs
            ├── restore_data
            │   ├── gopro
            |   |   ├── train
            │   │   |   ├── input
            │   │   |   │   ├── image0_x0.png
            │   │   |   │   ├── image0_x1.png
            │   │   |   │   ├── image0_x2.png
            │   │   |   ├── gt
            │   │   |   │   ├── image0.png
            |   |   ├── test
            │   │   |   ├── input
            │   │   |   │   ├── image1.png
            │   │   |   ├── gt
            │   │   |   │   ├── image1.png
    """

    def load_data_list(self) -> List[dict]:
        """Load data list from folder or annotation file.

        Returns:
            list[dict]: A list of annotation.
        """

        path_list = self._get_path_list()
        data_list = []
        for file in path_list:
            basename, ext = osp.splitext(file)
            if basename.startswith(os.sep):
                # Avoid absolute-path-like annotations
                basename = basename[1:]
            data = dict(key=basename)
            for key in self.data_prefix:
                if key == 'gt':
                    name = basename.split('_')[0]
                    path = osp.join(self.data_prefix[key],
                                    (f'{self.filename_tmpl[key].format(name)}'
                                     f'{ext}'))
                else:
                    path = osp.join(
                        self.data_prefix[key],
                        (f'{self.filename_tmpl[key].format(basename)}'
                         f'{ext}'))
                data[f'{key}_path'] = path
            data_list.append(data)

        return data_list
