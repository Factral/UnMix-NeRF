"""
Template DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union, Generic
from typing_extensions import TypeVar

import torch

from typing import (
    Any,
    Callable,
    ForwardRef,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
)

from pathlib import Path
import os.path as osp

from nerfstudio.utils.misc import IterableWrapper, get_orig_class
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.pixel_samplers import PatchPixelSamplerConfig, PixelSampler, PixelSamplerConfig
from functools import cached_property

from unmixnerf.data.unmix_dataparser import unmixDataParserConfig


CustomDataParserUnion = Union[unmixDataParserConfig, AnnotatedDataParserUnion]

@dataclass
class unmixDataManagerConfig(VanillaDataManagerConfig):
    """unmix DataManager Config
    """

    _target: Type = field(default_factory=lambda: unmixDataManager)
    dataparser: AnnotatedDataParserUnion = field(default_factory=unmixDataParserConfig)
    patch_size: int = 1

TDataset = TypeVar("TDataset", bound=InputDataset, default=InputDataset)

 
class unmixDataManager(VanillaDataManager, Generic[TDataset]):
    """unmix DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: unmixDataManagerConfig

    def __init__(
        self,
        config: unmixDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        num_classes: int = 5,
        **kwargs,  # pylint: disable=unused-argument
    ):
        config.dataparser.num_classes = num_classes

        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

        #if self.config.patch_size > 70:
        #    images = [self.train_dataset[i]["image"].permute(2, 0, 1)[None, ...] for i in range(len(self.train_dataset))]
        #    images = torch.cat(images)

        #    cache_dir = f"outputs/{self.config.dataparser.data.name}"
        #    dino_cache_path = Path(osp.join(cache_dir, "dino.npy"))



    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)

        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)

        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]

        ray_bundle = self.train_ray_generator(ray_indices)

        return ray_bundle, batch


    @cached_property # this is necessary to add the hs rays to the ray bundle
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[unmixDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is unmixDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is unmixDataManager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is unmixDataManager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default
        