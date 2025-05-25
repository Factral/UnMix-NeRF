"""
Hyperspectral dataset.
"""

from typing import Dict

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset

import torch
import numpy as np

import os
from skimage import segmentation

from unmixnerf.data.utils.vca import vca
from nerfstudio.utils.rich_utils import CONSOLE


from PIL import Image


class HyperspectralDataset(InputDataset):
    """Dataset that returns hyperspectral images.
    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device


    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)

        assert scale_factor == 1, 'Scale factors not yet supported for hyperspectral'
        assert (
            "hs_filenames" in dataparser_outputs.metadata.keys()
            and dataparser_outputs.metadata["hs_filenames"] is not None
        )
        self.hs_filenames = self.metadata["hs_filenames"]
        self.dino_filenames = self.metadata.get("dino_filenames", None)
        self.seg_filenames = self.metadata.get("seg_filenames", None)
        self.num_classes = self.metadata["num_classes"]

    def get_metadata(self, data: Dict) -> Dict:
        filepath = self.hs_filenames[data["image_idx"]]

        hs_image = np.load(filepath) # H, W, B 
        hs_image = torch.tensor(hs_image).float().clamp(0, 1)
    
        if not os.path.exists("vca.npy"):
            try:
                Ae, indice, Yp = vca(hs_image.permute(2,0,1).reshape(hs_image.shape[2], -1).numpy(), self.num_classes)
                np.save("vca.npy", Ae.T)
                #CONSOLE.log("VCA saved to vca.npy")
            except Exception as e:
                pass

        if self.seg_filenames is not None:
            seg_image = Image.open(self.seg_filenames[data["image_idx"]])
            seg_image = torch.tensor(np.array(seg_image))

            return {"hs_image": hs_image, "seg_image": seg_image}


        if self.dino_filenames is not None:
            dino_filepath = self.dino_filenames[data["image_idx"]]
            dino_feat = torch.load(dino_filepath).permute(1, 2, 0) # H, W, C
            # normalize dino_feat along unit sphere
            #dino_feat = dino_feat / torch.norm(dino_feat, dim=-1, keepdim=True)

            return {"hs_image": hs_image, "dino_feat": dino_feat}



        return {"hs_image": hs_image}
