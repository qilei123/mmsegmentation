from pycocotools.coco import COCO
import numpy as np
import os

from PIL import Image

def coco2segs(ann_dir,save_dir):

    coco = COCO(ann_dir)

    img_ids = coco.getImgIds()

    cat_ids = coco.getCatIds()
    
    for i in img_ids:
        
        img = coco.loadImgs([i])[0]

        seg_img = np.zeros((img['height'],img['width']))
        
        ann_ids = coco.getAnnIds(imgIds=[i])
        
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            seg_img = np.maximum(seg_img,coco.annToMask(ann)*ann['category_id'])

        seg_save_dir = os.path.join(save_dir,img['file_name']+".png")
        np.save(seg_save_dir, seg_img.astype(np.uint8))
        #Image.fromarray(seg_img).astype(np.uint8).save(seg_save_dir, 'PNG')

coco2segs("data/td/annotations/train_AW.json","data/td/annotations/segs/")

    