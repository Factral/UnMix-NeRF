"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig

from unmixnerf.data.unmix_datamanager import (
    unmixDataManagerConfig, unmixDataManager
)
from unmixnerf.unmix_model import unmixConfig
from unmixnerf.unmix_pipeline import (
    unmixPipelineConfig,
)
from unmixnerf.data.unmix_dataparser import unmixDataParserConfig

from unmixnerf.data.utils.hs_dataloader import HyperspectralDataset

unmix_method = MethodSpecification(
    config=TrainerConfig(
        method_name="unmixnerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        save_only_latest_checkpoint=False,
        pipeline=unmixPipelineConfig(
            datamanager=unmixDataManagerConfig(
                _target=unmixDataManager[HyperspectralDataset],
                dataparser=unmixDataParserConfig(),
                train_num_rays_per_batch=9216*4,
                eval_num_rays_per_batch=4096,
            ),
            model=unmixConfig(
                eval_num_rays_per_chunk=512,
                #grid_levels=1,
                #alpha_thre=0.0,
                #cone_angle=0.0,
                #disable_scene_contraction=True,
                #near_plane=0.01,
                #background_color="black",
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=2e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.00001, max_steps=30000),
            },
        }, 
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="viewer",
    ),
    description="unmix method",
)
