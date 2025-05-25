"""
unmix Field spectral unmixing.
"""

from typing import Literal, Optional, Any, Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import os

from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from nerfstudio.fields.base_field import get_normalized_directions
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding

from unmixnerf.utils.spec_to_rgb import ColourSystem
import numpy as np
import tinycudann as tcnn


from nerfstudio.data.scene_box import SceneBox


class unmixField(NerfactoField):
    """unmix Field with spectral unmixing."""

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        wavelengths: int = 128,
        method: Literal["rgb", "spectral", "rgb+spectral"] = "rgb",
        num_classes: int = 4,
        feature_dim: int = 256,
        temperature: float = 0.5,
        converter: ColourSystem = None,
        pred_dino: bool = False,
        pred_specular: bool = False,
        load_vca: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(aabb=aabb, num_images=num_images,implementation=implementation, **kwargs)

        self.method = method
        self.num_classes = num_classes
        self.wavelengths = wavelengths
        self.feature_dim = feature_dim
        self.pred_specular = pred_specular
        self.average_init_density = 1

        if self.method == "spectral" or self.method == "rgb+spectral":
            # Semantic field for abundance prediction
            input_dim = self.position_encoding.get_out_dim() + self.geo_feat_dim
            out_dim_feature = self.num_classes + 1 if self.pred_specular else self.num_classes
            self.feature_mlp = MLP(
                in_dim=input_dim,
                num_layers=3,
                layer_width=hidden_dim_color,
                out_dim=out_dim_feature,
                activation=nn.ReLU(),
                out_activation=None, # tanh ?
                implementation=implementation,
            )

            if self.training:
                if load_vca:
                    endmembers = np.load("vca.npy")
                    self.endmembers = nn.Parameter(torch.tensor(endmembers, dtype=torch.float32, requires_grad=True))
                    #self.endmembers = nn.Parameter(torch.randn(self.num_classes, self.wavelengths), requires_grad=True)
                else:
                    self.endmembers = nn.Parameter(torch.randn(self.num_classes, self.wavelengths), requires_grad=True)
            else:
                # will be loaded from the checkpoint
                self.endmembers = torch.randn(self.num_classes, self.wavelengths).cuda()
        
    
            self.mlp_head = MLP(
                in_dim=self.position_encoding.get_out_dim()+ self.geo_feat_dim + self.appearance_embedding_dim,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim=num_classes,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )

            self.mlp_directional = MLP(
               in_dim=self.direction_encoding.get_out_dim() + self.position_encoding.get_out_dim(),
               num_layers=2,
               layer_width=16,
               out_dim=self.wavelengths,
               activation=nn.ReLU(),
               out_activation=nn.Sigmoid(),
               implementation=implementation,
            )

            self.converter = converter
            self.temperature = temperature
            self.pred_dino = pred_dino
            self.use_scalar = True

            
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[Any, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # Get appearance embedding
        embedded_appearance = None
        if self.embedding_appearance is not None:
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                if self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    )

        if "spectral" in self.method:

            positions =  ray_samples.frustums.get_positions()
            positions_flat = self.position_encoding(positions.view(-1, 3))


            if len(positions_flat.shape) == 2:
                positions_flat = positions_flat.unsqueeze(0)
            else:
                positions_flat = positions_flat.view(-1, density_embedding.size(1),  self.position_encoding.get_out_dim() )


            if len(density_embedding.shape) == 2:
                density_embedding = density_embedding.unsqueeze(0)


            h1 = torch.cat(
                [
                    #d,
                    positions_flat.view(-1, self.position_encoding.get_out_dim()),
                    density_embedding.view(-1, self.geo_feat_dim),
                ]
                + (
                    [embedded_appearance.view(-1, self.appearance_embedding_dim)] 
                    if embedded_appearance is not None else []
                ),
                dim=-1,
            ) # direction, density features, appeareance embeddings

            if self.use_scalar:
                scalar = self.mlp_head(h1).view(*outputs_shape, -1, self.num_classes)
                scalar = F.sigmoid(scalar)

            features_input = torch.cat([positions_flat, density_embedding], dim=-1) # positions, density

            size = features_input.size()
            features_input = features_input.view(-1, features_input.size(-1))
            
            features = self.feature_mlp(features_input)
            logits = features.view(*size[:-1], -1)

            if self.pred_specular:
                logits, s1 = torch.split(logits, [self.num_classes, 1], dim=-1)
                s1 = F.sigmoid(s1)

            abundances = F.softmax(logits / self.temperature, dim=-1)

            endmembers = self.endmembers.unsqueeze(0).unsqueeze(0)
    
            endmembers = endmembers.expand(abundances.shape[0], abundances.shape[1], -1, -1).transpose(2,3).squeeze(0)

            if self.use_scalar:
                adapted_endmembers = scalar * endmembers  # (B, ray_sample, wavelengths, num_classes)
            else:
                adapted_endmembers = endmembers

            spec = (adapted_endmembers  @ abundances.unsqueeze(-1)).squeeze() # linear mixing model spec = EA

            if self.pred_specular:

                input_spec = torch.cat(
                [
                    d,
                    positions_flat.view(-1, self.position_encoding.get_out_dim()),
                ],
                dim=-1,
                )

                specular = self.mlp_directional(input_spec).view(*outputs_shape, self.wavelengths) # (B, ray_sample, wavelengths)
                spec2 = spec +  (s1 * specular)

            if self.pred_specular:
                outputs["spectral"] = spec2.to(directions)
                outputs["spectral2"] = spec.to(directions)
                with torch.no_grad():
                    outputs["specular"] = (s1 * specular).to(directions)
            else:
                outputs["spectral"] = spec.to(directions)

            outputs["abundances"] = abundances.to(directions)


        return outputs



    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]

        assert positions.numel() > 0, "positions is empty."

        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)

        assert positions_flat.numel() > 0, "positions_flat is empty."
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = self.average_init_density * trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out