# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import *
from .custom import CustomDataset
from pycocotools.coco import COCO

@DATASETS.register_module()
class TDDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('Small 1-piece vehicle', 'Large 1-piece vehicle', 'Extra-large 2-piece truck')

    PALETTE = [[128, 0, 0], [0, 128, 0], [0, 0, 128]]

    def __init__(self, **kwargs):
        super(TDDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png',reduce_zero_label=True, **kwargs)
        assert osp.exists(self.img_dir) #and self.split is not None

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        self.coco = COCO(ann_dir)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        if "train_AW.json" in ann_dir:
            self.ann_dir = ann_dir.replace("train_AW.json","segs")
        if "test_AW.json" in ann_dir:
            self.ann_dir = ann_dir.replace("test_AW.json","segs")            
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()

        total_ann_ids = []
        
        for i in self.img_ids:
            
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            
            ann_ids = self.coco.getAnnIds(imgIds=[i])
            
            #if len(ann_ids)==0:
            #    continue
            
            #anns = self.coco.loadAnns(ann_ids)
            
            info['ann'] = dict(seg_map=info['filename']+seg_map_suffix)
            
            img_infos.append(info)

            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_dir}' are not unique!"

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos


@DATASETS.register_module()
class TDDataset4(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('background','Small 1-piece vehicle', 'Large 1-piece vehicle', 'Extra-large 2-piece truck')

    PALETTE = [[0,0,0],[128, 0, 0], [0, 128, 0], [0, 0, 128]]

    def __init__(self, **kwargs):
        super(TDDataset4, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
        assert osp.exists(self.img_dir) #and self.split is not None

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        self.coco = COCO(ann_dir)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        if "train_AW.json" in ann_dir:
            self.ann_dir = ann_dir.replace("train_AW.json","segs")
        if "test_AW.json" in ann_dir:
            self.ann_dir = ann_dir.replace("test_AW.json","segs")            
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()

        total_ann_ids = []
        
        for i in self.img_ids:
            
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            
            ann_ids = self.coco.getAnnIds(imgIds=[i])
            
            #if len(ann_ids)==0:
            #    continue
            
            #anns = self.coco.loadAnns(ann_ids)
            
            info['ann'] = dict(seg_map=info['filename']+seg_map_suffix)
            
            img_infos.append(info)

            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_dir}' are not unique!"

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos