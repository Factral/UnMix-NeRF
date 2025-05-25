"""
Nerfstudio unmix Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import numpy as np
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig, DynamicBatchPipeline
from nerfstudio.utils import profiler

from unmixnerf.data.unmix_datamanager import unmixDataManagerConfig
from unmixnerf.unmix_model import unmixModel, unmixConfig

import torch
import wandb


@dataclass
class unmixPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: unmixPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = unmixDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = unmixConfig()
    """specifies the model config"""
    check_nan: bool = False
    num_classes: int = 5
    
    #target_num_samples: int = 262144  # 1 << 18
    """The target number of samples to use for an entire batch of rays."""
    #max_num_samples_per_ray: int = 1024  # 1 << 10
    """The maximum number of samples to be placed along a ray."""

# based on: https://github.com/nerfstudio-project/nerfstudio/blob/758ea1918e082aa44776009d8e755c2f3a88d2ee/nerfstudio/pipelines/base_pipeline.py#L212
class unmixPipeline(VanillaPipeline):
    """unmix Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """
    dynamic_num_rays_per_batch: int


    def __init__(
        self,
        config: unmixPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 3,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()

        self.config = config
        #self.dynamic_num_rays_per_batch = self.config.target_num_samples // self.config.max_num_samples_per_ray
        #self._update_pixel_samplers()

        if config.check_nan:
            torch.autograd.set_detect_anomaly(True)

        self.test_mode = test_mode
        #try:
        #    wandb.init(project="unmixNeRF", entity="kaust-uis")
        #except Exception as e:
        #    print(f"Failed to initialize wandb: {e}")

        world_size = 1
        self.datamanager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            num_classes=config.num_classes
        )
        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            grad_scaler=grad_scaler,
            num_classes=config.num_classes,
            wavelengths=self.datamanager.train_dataparser_outputs.metadata.get("wavelengths", None),
        )
        self.model.to(device)

        self.world_size = 1
        world_size=1
        if world_size > 1:
            self._model = typing.cast(unmixModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

        # random permutate endmembers in model with torch
        #self.model.field.endmembers

    #autocast is needed when using tcnn https://github.com/NVlabs/tiny-cuda-nn/issues/377
    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        
        with torch.autocast(device_type="cuda", enabled=True):
            self.eval()
            ray_bundle, batch = self.datamanager.next_eval(step)
            model_outputs = self.model(ray_bundle)
            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
            self.train()
        return model_outputs, loss_dict, metrics_dict

    #autocast is needed when using tcnn https://github.com/NVlabs/tiny-cuda-nn/issues/377
    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        with torch.autocast(device_type="cuda", enabled=True):
            self.eval()
            camera, batch = self.datamanager.next_eval_image(step)
            outputs = self.model.get_outputs_for_camera(camera)
            metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
            assert "num_rays" not in metrics_dict
            metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()
            self.train()
        return metrics_dict, images_dict


    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state)
        # perm endmembers, is a matrix of n_classes x bands
        perm_idx = torch.randperm(self.model.field.endmembers.shape[0])
        #self.model.field.endmembers = torch.nn.Parameter(self.model.field.endmembers[perm_idx])
        #self.model.field.endmembers = torch.nn.Parameter(torch.randn_like(self.model.field.endmembers))
        # save the endmembers as npy
        np.save("endmembers_hotdog.npy", self.model.field.endmembers.detach().cpu().numpy())
        #metrics_dict = self.get_average_eval_image_metrics(step=1)
        #print(metrics_dict)
        #writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step)
