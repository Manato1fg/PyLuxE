import torch

from pyluxe.element import Element
from pyluxe.enum import Medium
from pyluxe.typing import MediumTensor


class FresnelDiffraction(Element):
    """
    Fresnel diffraction element.
    """

    def __init__(self, distance: float, **kwargs):
        """
        # Parameters
        name: str
            Name of the element.
        distance: float
            Distance of the diffraction.
        """
        super().__init__(**kwargs)
        self.distance = distance

    def forward(self, inputs: MediumTensor) -> MediumTensor:
        """
        # Parameters
        inputs: Tensor
            Input tensor

        # Returns
        Tensor
            Output tensor. It must be of shape (batch_size, *).
            Perform Fresnel diffraction on the input tensor.
        """
        assert inputs.medium == Medium.LIGHT, "Medium must be LIGHT."
        # Perform Fresnel diffraction
        return torch.fft.fftn(inputs)

    def __str__(self):
        return f"FresnelDiffraction ({self.name})"
