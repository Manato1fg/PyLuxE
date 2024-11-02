from typing import Tuple

import torch

from pyluxe.element import Element
from pyluxe.enum import Medium
from pyluxe.typing import MediumTensor


class LaserWithCollimatorLens(Element):
    """
    Laser with collimator lens source.
    """

    def __init__(
        self,
        wavelength: float,
        output_size: Tuple[int, int] = (1920, 1080),
        intensity: float = 1.0,
        **kwargs,
    ):
        """
        # Parameters
        wavelength: float
            Wavelength of the laser.
        """
        super().__init__(**kwargs)
        self.wavelength = wavelength
        self.output_size = output_size
        self.intensity = torch.nn.Parameter(torch.tensor(intensity))

    def forward(self, _) -> MediumTensor:
        """
        # Parameters
        inputs: Tensor
            Not used.

        # Returns
        Tensor
            Output tensor. It must be of shape (batch_size, *).
            Perform laser with collimator lens on the input tensor.
        """
        return MediumTensor(
            torch.ones(self.output_size) * self.intensity,
            medium=Medium.LIGHT,
            wavelengths=[self.wavelength],
            color_dim=1,
        )


class ColorDisplay(Element):
    """
    Color display source.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, input_tensor: torch.Tensor) -> MediumTensor:
        """
        Emit a light source.
        """
        ndim = input_tensor.ndim
        if ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        med_tensor = MediumTensor(
            input_tensor,
            medium=Medium.LIGHT,
            wavelengths=[0.65, 0.53, 0.47],
            color_dim=1,
        )
        if ndim == 3:
            med_tensor.elem = med_tensor.squeeze(0)
        return med_tensor
