import torch

from pyluxe.element import Element
from pyluxe.typing import MediumTensor


class CutOut(Element):
    """
    Cut-out element.
    """

    def __init__(self, mask: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.mask = mask

    def forward(self, inputs: MediumTensor) -> MediumTensor:
        """
        # Parameters
        inputs: Tensor
            Input tensor

        # Returns
        Tensor
            Output tensor. It must be of shape (batch_size, *).
            Perform cut-out on the input tensor.
        """
        assert (
            inputs.shape[-2:] == self.mask.shape
        ), "Input tensor and mask must have the same shape."
        return inputs * self.mask

    def __str__(self):
        return f"CutOut ({self.name})"
