import numpy as np
import pandas as pd
from floods.hard_positive_regions import AuditGuidedHardPositiveCropSupervision, manifest_matching_indices

def test_hard_positive_manifest_crop(tmp_path):
    manifest=tmp_path/'hard_positive_regions.csv'
    pd.DataFrame([{'tile_id':'EMSR1_tile','x0':4,'y0':4,'crop_size':8,'fn_pixels':10,'score':2.0}]).to_csv(manifest,index=False)
    sup=AuditGuidedHardPositiveCropSupervision(manifest,target_size=16,probability=1.0)
    image=np.zeros((20,20,3),dtype=np.float32); mask=np.zeros((20,20),dtype=np.uint8); mask[5:9,5:9]=1
    out_image,out_mask,_=sup(image,mask,'/x/EMSR1_tile.tif')
    assert out_image.shape==(16,16,3)
    assert out_mask.shape==(16,16)
    assert manifest_matching_indices(['/x/EMSR1_tile.tif','/x/other.tif'],manifest)=={0}
