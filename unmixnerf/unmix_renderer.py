import torch
from torch import nn
from typing import Optional, Tuple, Union
from jaxtyping import Float, Int
from torch import Tensor
from nerfstudio.model_components.renderers import SemanticRenderer
from nerfstudio.utils import colors
import nerfacc

class SpectralRenderer(SemanticRenderer):
    """Calculate spectral along the ray."""
    background_color: Optional[str] = "random"

    @classmethod
    def forward(
        cls,
        spectral: Float[Tensor, "*bs num_samples num_classes"],
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs num_classes"]:
        """Calculate spectral along the ray."""
        if spectral.dim() == 3:
            spectral = spectral.squeeze(0)
        elif spectral.dim() == 1:
            spectral = spectral.unsqueeze(0)

        return nerfacc.accumulate_along_rays(
                weights[..., 0], values=spectral, ray_indices=ray_indices, n_rays=num_rays
            )

#torch.Size([2465439, 1])  weights
#torch.Size([1, 2465439, 21])  values     

    @classmethod
    def get_background_color(
        cls, background_color, shape: Tuple[int, ...], device: torch.device
    ) -> Union[Float[Tensor, "3"], Float[Tensor, "*bs 3"]]:
        """Returns the RGB background color for a specified background color.
        Note:
            This function CANNOT be called for background_color being either "last_sample" or "random".

        Args:
            background_color: The background color specification. If a string is provided, it must be a valid color name.
            shape: Shape of the output tensor.
            device: Device on which to create the tensor.

        Returns:
            Background color as RGB.
        """
        assert background_color not in {"last_sample", "random"}
        if isinstance(background_color, str) and background_color in colors.COLORS_DICT:
            background_color = colors.COLORS_DICT[background_color]
        assert isinstance(background_color, Tensor)

        return background_color.expand(shape).to(device)

    def blend_background(
        self,
        image: Tensor,
        rgba: Tensor,
        background_color = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Blends the background color into the image if image is RGBA.
        Otherwise no blending is performed (we assume opacity of 1).

        Args:
            image: RGB/RGBA per pixel.
            opacity: Alpha opacity per pixel.
            background_color: Background color.

        Returns:
            Blended RGB.
        """
        if rgba.size(-1) < 4:
            return image

        _, opacity = rgba[..., :3], rgba[..., 3:]
        if background_color is None:
            background_color = self.background_color
            if background_color in {"last_sample", "random"}:
                background_color = "black"
        background_color = self.get_background_color(background_color, shape=image.shape, device=image.device)
        assert isinstance(background_color, torch.Tensor)
        # save opacity in disk
        return image * opacity + background_color.to(image.device) * (1 - opacity)


    def blend_background_for_loss_computation(
        self,
        pred_image: Tensor,
        pred_accumulation: Tensor,
        gt_image: Tensor,
        rgba_image: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Blends a background color into the ground truth and predicted image for
        loss computation.

        Args:
            gt_image: The ground truth image.
            pred_image: The predicted spectral values (without background blending).
            pred_accumulation: The predicted opacity/ accumulation.
            rgba_image: The RGBA values for the ground truth image.
        Returns:
            A tuple of the predicted and ground truth RGB values.
        """
        background_color = self.background_color
        if background_color == "last_sample":
            background_color = "black"  # No background blending for GT
        elif background_color == "random":
            background_color = torch.rand_like(pred_image)
            pred_image = pred_image + background_color * (1.0 - pred_accumulation)
        gt_image = self.blend_background(gt_image, rgba_image, background_color=background_color)
        return pred_image, gt_image


def get_weights_spectral(deltas, densities):
    """Return weights based on predicted densities

    Args:
        densities: Predicted densities for samples along ray

    Returns:
        Weights for each sample
    """

    delta_density = deltas * densities
    alphas = 1 - torch.exp(-delta_density)

    transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
    transmittance = torch.cat(
        [torch.zeros((*transmittance.shape[:1], 1, transmittance.shape[-1]), device=densities.device), transmittance], dim=-2
    )
    transmittance = torch.exp(-transmittance)

    weights = alphas * transmittance
    weights = torch.nan_to_num(weights)

    return weights