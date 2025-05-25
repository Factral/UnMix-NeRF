"""
Unmix Model File

"""

import torch
from dataclasses import dataclass, field
from typing import Type, Dict, List, Literal, Tuple, Union, Optional
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import nerfacc

from torchmetrics.image import SpectralAngleMapper
from nerfstudio.models.instant_ngp  import NGPModel, InstantNGPModelConfig
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.utils import colormaps
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.model_components.scene_colliders import NearFarCollider, AABBBoxCollider


from nerfstudio.model_components.ray_samplers import VolumetricSampler

from nerfstudio.model_components.losses import (
    MSELoss,
    scale_gradients_by_distance_squared,
)

import wandb

from unmixnerf.unmix_field import unmixField
from unmixnerf.utils.spec_to_rgb import ColourSystem
from unmixnerf.unmix_renderer import SpectralRenderer, get_weights_spectral
from unmixnerf.utils.clusterprobe import ClusterLookup
from nerfstudio.configs.config_utils import to_immutable_dict

from torch.nn import CrossEntropyLoss


@dataclass
class unmixConfig(InstantNGPModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: unmixModel)
    """target class to instantiate"""
    enable_collider: bool = False
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = to_immutable_dict({"near_plane": 2.0, "far_plane": 6.0}) #None
    """Instant NGP doesn't use a collider."""
    grid_resolution: Union[int, List[int]] = 128 # 128
    """Resolution of the grid used for the field."""
    grid_levels: int = 4 # 4
    """Levels of the grid used for the field."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    alpha_thre: float = 0.01
    """Threshold for opacity skipping."""
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: Optional[float] = None
    """Minimum step size for rendering."""
    near_plane: float = 0.05
    """How far along ray to start sampling."""
    far_plane: float = 1e3
    """How far along ray to stop sampling."""
    use_gradient_scaling: bool = True
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    use_appearance_embedding: bool = True
    """Whether to use an appearance embedding."""
    background_color: Literal["random", "black", "white"] = "random"
    """
    The color that is given to masked areas.
    These areas are used to force the density in those regions to be zero.
    """
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""

    implementation: Literal["tcnn", "torch"] = "torch"
    """Which implementation to use for the model."""

    # custom configs
    method: Literal["rgb", "spectral", "rgb+spectral"] = "rgb"

    # weight rgb loss
    rgb_loss_weight: float = 1.0
    # weight spectral loss
    spectral_loss_weight: float = 1.0
    # temperature
    temperature: float = 0.2

    pred_dino: bool = False
    pred_specular: bool = False
    load_vca: bool = False


class unmixModel(NGPModel):
    """unmix Model."""

    config: unmixConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = 0 if self.config.use_appearance_embedding else 32
 
        # just for visualization
        self.class_colors = {
        0: torch.tensor([0.49, 0.29, 0.95]),
        1: torch.tensor([0.29, 0.95, 0.30]),
        2: torch.tensor([0.95, 0.29, 0.47]),
        3: torch.tensor([0.29, 0.66, 0.95]),
        4: torch.tensor([0.86, 0.95, 0.29]),
        5: torch.tensor([0.85, 0.29, 0.95]),
        6: torch.tensor([0.29, 0.95, 0.66]),
        7: torch.tensor([0.95, 0.46, 0.29]),
        8: torch.tensor([0.29, 0.30, 0.95]),
        9: torch.tensor([0.50, 0.95, 0.29]),
        10: torch.tensor([0.95, 0.29, 0.69]),
        11: torch.tensor([0.29, 0.88, 0.95]),
        12: torch.tensor([0.95, 0.82, 0.29]),
        13: torch.tensor([0.63, 0.29, 0.95]),
        14: torch.tensor([0.29, 0.95, 0.43])
        }

        self.i = 0

        if 'spectral' in self.config.method:
            self.renderer_spectral = SpectralRenderer() 
            self.spectral_loss = MSELoss()

        self.step = 0
        self.kwargs = self.kwargs["metadata"]
        self.converter = ColourSystem(bands=self.kwargs["wavelengths"], cs='sRGB')
        
        self.rgb_loss = MSELoss()

        self.sam = SpectralAngleMapper(reduction="none")

        self.field = unmixField(
            aabb=self.scene_box.aabb,
            appearance_embedding_dim=0 if self.config.use_appearance_embedding else 32,
            num_images=self.num_train_data,
            log2_hashmap_size=self.config.log2_hashmap_size,
            max_res=self.config.max_res,
            spatial_distortion=scene_contraction,
            implementation=self.config.implementation,
            method=self.config.method,
            wavelengths=len(self.kwargs["wavelengths"]) if 'spectral' in self.config.method else 0,
            num_classes=self.kwargs["num_classes"],
            temperature=self.config.temperature,
            converter=self.converter,
            pred_dino=self.config.pred_dino,
            pred_specular=self.config.pred_specular,
            load_vca=self.config.load_vca
        ) 

        # Reinitialize the occupancy grid and sampler using the new field.
        self.scene_aabb = torch.nn.Parameter(self.scene_box.aabb.flatten(), requires_grad=False)
        if self.config.render_step_size is None:
            self.config.render_step_size = (((self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2).sum().sqrt().item() / 1000)
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )
        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        #self.collider = AABBBoxCollider(self.scene_box)

        self.cluster_probe = ClusterLookup(len(self.kwargs["wavelengths"]), self.kwargs["num_classes"])


    def label_to_rgb(self, labels):
        device = labels.device
        colors = torch.stack(list(self.class_colors.values())).to(device)
        return colors[labels.long().squeeze(-1)]

    def get_outputs(self, ray_bundle: RayBundle):
        assert self.field is not None
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )

        field_outputs = self.field(ray_samples)

        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0]
        weights = weights[..., None]

        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )

        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)

        outputs = {
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.method == "rgb":
            rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
            outputs["rgb"] = rgb

        if "spectral" in self.config.method:
            spectral = self.renderer_spectral(spectral=field_outputs["spectral"], weights=weights, ray_indices=ray_indices,
                                              num_rays=num_rays)
            outputs["spectral"] = spectral
            for i in range(spectral.shape[-1]):
                outputs[f"wv_{i}"] = spectral[..., i]

            if self.config.pred_specular:
                spectral2 = self.renderer_spectral(spectral=field_outputs["spectral2"], weights=weights, ray_indices=ray_indices,
                                                   num_rays=num_rays)
                outputs["spectral2"] = spectral2

                with torch.no_grad():
                    specular = self.renderer_spectral(spectral=field_outputs["specular"], weights=weights, ray_indices=ray_indices,
                                                      num_rays=num_rays)
                    outputs["specular"] = specular
                for i in range(specular.shape[-1]):
                    outputs[f"residual_{i}"] = specular[..., i]

            # pseudorgb
            if self.config.method == "spectral":
                with torch.no_grad():
                    outputs["rgb"] = self.converter(spectral)
            else:
                outputs["rgb"] = self.converter(spectral)

            outputs["num_samples_per_ray"] = packed_info[:, 1]

            # render abundances
            with torch.no_grad():
                abundaces = self.renderer_spectral(spectral=field_outputs["abundances"], weights=weights, ray_indices=ray_indices,
                                                   num_rays=num_rays)
                outputs["abundances"] = abundaces
                for i in range(field_outputs["abundances"].shape[-1]):
                    outputs[f"abundances_{i}"] = abundaces[:, i]

            inner_products, cluster_probs = self.cluster_probe(spectral,alpha=None, clusters = self.field.endmembers)
            outputs["seg_probs"] = cluster_probs

            with torch.no_grad():
                acc_if = torch.where(accumulation > 0.5, torch.tensor(1.).to(accumulation.device), torch.tensor(0.).to(accumulation.device))
                outputs["seg_raw"] = cluster_probs.argmax(1) * acc_if.squeeze()
                outputs["seg_pred"] = self.label_to_rgb(cluster_probs.argmax(1)) * acc_if

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}

        image = batch["image"].to(self.device)
        if "spectral" in self.config.method:
            gt_spectral = batch["hs_image"].to(self.device)
            pred_spectral = outputs["spectral"]
            pred_rgb = outputs["rgb"]
            gt_rgb = image
            if 'spectral2' in outputs:
                pred_spectral2 = outputs["spectral2"]

            if 'seg_image' in batch:
                seg_image = batch["seg_image"].to(self.device)

        pred_spectral, gt_spectral = self.renderer_spectral.blend_background_for_loss_computation(
            pred_image=outputs["spectral"],
            pred_accumulation=outputs["accumulation"],
            gt_image=gt_spectral,
            rgba_image=image
        )

        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image
        )

        if self.config.method == "rgb":
            loss_dict["rgb_loss"] = self.rgb_loss(pred_rgb, gt_rgb)
        elif self.config.method == "spectral":
            loss_dict["spectral_loss"] = self.spectral_loss(pred_spectral, gt_spectral)
        elif self.config.method == "rgb+spectral":
            loss_dict["spectral_loss"] = self.config.spectral_loss_weight * self.spectral_loss(pred_spectral, gt_spectral)
            loss_dict["rgb_loss"] = self.config.rgb_loss_weight * self.rgb_loss(pred_rgb, gt_rgb)

            #if self.config.pred_specular:
            #     loss_dict["spectral_loss2"] = 1 * self.spectral_loss(pred_spectral2, gt_spectral)

        if self.config.pred_dino:
            loss_dict["dino_mse"] = torch.nn.functional.mse_loss(outputs["dino"], batch["dino_feat"]).nanmean()
            if self.step > 3000:
                loss_dict["cluster_loss"] = -(outputs["cluster_probs"] * outputs["inner_products"]).sum(1).mean()

        return loss_dict

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA

        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        # Calculate RMSE for RGB
        mse_rgb = torch.nn.functional.mse_loss(predicted_rgb, gt_rgb)
        metrics_dict["rmse"] = torch.sqrt(mse_rgb).item()

        if "spectral" in self.config.method:
            gt_spectral = batch["hs_image"].to(self.device)
            metrics_dict["psnr_spectral"] = self.psnr(outputs["spectral"], gt_spectral)
            # Calculate RMSE for spectral
            mse_spectral = torch.nn.functional.mse_loss(outputs["spectral"], gt_spectral)
            metrics_dict["rmse_spectral"] = torch.sqrt(mse_spectral).item()

        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()

        return metrics_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        
        gt_rgb = batch["image"].to(self.device)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)

        predicted_rgb = outputs["rgb"]

        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)
        se_per_pixel = (gt_rgb - predicted_rgb) ** 2
        se_per_pixel = se_per_pixel.squeeze().permute(1, 2, 0).mean(dim=-1).unsqueeze(-1)

        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}
        if "spectral" in self.config.method:
            gt_spectral = batch["hs_image"].to(self.device)
            gt_spectral = torch.moveaxis(gt_spectral, -1, 0)[None, ...]
            predicted_spectral = outputs["spectral"]
            predicted_spectral = torch.moveaxis(predicted_spectral, -1, 0)[None, ...]
            psnr_spectral = self.psnr(gt_spectral, predicted_spectral)
            ssim_spectral = self.ssim(gt_spectral, predicted_spectral)
            
            sam_spectral =  torch.nanmean(self.sam(predicted_spectral, gt_spectral))
            metrics_dict["psnr_spectral"] = float(psnr_spectral.item())
            metrics_dict["ssim_spectral"] = float(ssim_spectral)
            metrics_dict["sam_spectral"] = float(sam_spectral)

            #rmse
            metrics_dict["rmse_spectral"] = torch.sqrt(torch.nn.functional.mse_loss(predicted_spectral, gt_spectral)).item()
    
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth, "se_per_pixel": se_per_pixel}

        return metrics_dict, images_dict

    def compute_sam(self, pred, gt, eps=1e-8):
        """
        Compute the Spectral Angle Mapper (SAM) between the predicted and ground truth spectral images.
        Assumes pred and gt have shape (..., channels).
        Returns the mean spectral angle in degrees.
        """
        dot_product = (pred * gt).sum(dim=-1)
        norm_pred = torch.norm(pred, dim=-1)
        norm_gt = torch.norm(gt, dim=-1)
        cos_angle = dot_product / (norm_pred * norm_gt + eps)
        cos_angle = torch.clamp(cos_angle, -1, 1)
        angle = torch.acos(cos_angle)
        return angle.mean()

    # related to https://github.com/NVlabs/tiny-cuda-nn/issues/377
    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Assumes a ray-based model.

        Args:   
            camera: generates raybundle
        """
        with torch.autocast(device_type="cuda", enabled=True):
            return self.get_outputs_for_camera_ray_bundle(
                camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box)
            )

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if True:
            # anneal the weights of the proposal network before doing PDF sampling

            def update_occupancy_grid(step: int):
                self.step = step
                self.occupancy_grid.update_every_n_steps(
                    step=step,
                    occ_eval_fn=lambda x: self.field.density_fn(x) * self.config.render_step_size,
                )


            if self.config.method != "rgb":
                def clamp_endmembers(step):
                    with torch.no_grad():
                        self.field.endmembers[:] = self.field.endmembers.clamp(0, 1)
                    if self.step % 100 == 0:
                        np.save(f"endmembers.npy", self.field.endmembers.cpu().detach().numpy())

                # callback to clamp between 0 and 1 the endmember parameter
                callbacks.append(
                    TrainingCallback(
                        where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                        update_every_num_iters=1,
                        func=clamp_endmembers
                    )
                )

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=update_occupancy_grid,
                )
            )

        return callbacks

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # move the chunk inputs to the model device
            ray_bundle = ray_bundle.to(self.device)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():
                if not isinstance(output, torch.Tensor):
                    continue
                # move the chunk outputs from the model device back to the device of the inputs.
                outputs_lists[output_name].append(output.to(input_device))
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)
        return outputs



