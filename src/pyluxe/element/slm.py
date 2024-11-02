from typing import Tuple

import torch

from pyluxe.element import Element
from pyluxe.enum import InitializeType
from pyluxe.typing import MediumTensor


class SLM(Element):
    """
    Spatial light modulator element.
    """

    def __init__(
        self,
        pixel_size: float = 1e-6,
        size: Tuple[int, int] = (1920, 1080),
        initialize_type: InitializeType = InitializeType.ZEROS,
        gaussian_mean: float = 0.0,
        gaussian_std: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pixel_size = pixel_size
        self.size = size

        # Initialize the weight matrix
        match initialize_type:
            case InitializeType.ZEROS:
                self.weight = torch.zeros(size)
            case InitializeType.GAUSSIAN:
                self.weight = torch.normal(gaussian_mean, gaussian_std, size)
            case InitializeType.RANDOM:
                self.weight = torch.rand(size)
        self.weight = torch.nn.Parameter(self.weight)

    def forward(self, inputs: MediumTensor) -> MediumTensor:
        """
        # Parameters
        inputs: Tensor
            Input tensor. This must be the same size as the weight matrix.

        # Returns
        MediumTensor
            Output tensor. This is the element-wise multiplication
            of the input and the weight matrix.
        """
        phase = torch.exp(1j * self.weight)
        return inputs * phase

    def __str__(self):
        return f"SLM ({self.name})"
