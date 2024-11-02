from typing import Tuple

import torch

from pyluxe.element import Element
from pyluxe.enum import Medium
from pyluxe.typing import MediumTensor


class Camera(Element):
    """
    Camera element.
    """

    def __init__(self, shape: Tuple[int, int], **kwargs):
        """
        # Parameters
        name: str
            Name of the element.
        """
        super().__init__(**kwargs)
        self.shape = shape

    def forward(self, inputs: MediumTensor) -> MediumTensor:
        """
        # Parameters
        inputs: Tensor
            Input tensor

        # Returns
        Tensor
            Output tensor. It must be of shape (batch_size, *).
            Perform camera operation on the input tensor.
        """
        assert inputs.medium == Medium.LIGHT, "Medium must be LIGHT."

        sw, sh = self.shape
        iw, ih = inputs.shape[-2:]

        # Camera and inputs are assumed to be in the same plane
        # and perfectly aligned
        _inputs = inputs.elem
        if iw > sw:
            dx = (iw - sw) // 2
            _inputs = _inputs[..., dx : dx + sw, :]
        elif iw < sw:
            dx = (sw - iw) // 2
            _inputs = torch.nn.functional.pad(_inputs, (dx, dx, 0, 0))

        if ih > sh:
            dy = (ih - sh) // 2
            _inputs = _inputs[..., dy : dy + sh, :]
        elif ih < sh:
            dy = (sh - ih) // 2
            _inputs = torch.nn.functional.pad(_inputs, (0, 0, dy, dy))

        # Note:
        # inputs is a complex tensor and camera cannot capture complex tensor
        # The camera can only capture the intensity of the light
        # Also, the camera converts the medium tensor into a medium tensor
        # on Electronic medium
        intensity = torch.sum(_inputs * _inputs.conj())
        return MediumTensor(
            intensity,
            medium=Medium.ELECTRON,
            wavelengths=inputs.wavelengths,
            color_dim=inputs.color_dim,
        )

    def __str__(self):
        return f"Camera ({self.name})"


class SingleDetectorWithLens(Element):
    """
    Single detector with lens element.
    """

    def __init__(self, focal_length: float, **kwargs):
        """
        # Parameters
        name: str
            Name of the element.
        """
        super().__init__(**kwargs)

    def forward(self, inputs: MediumTensor) -> MediumTensor:
        """
        # Parameters
        inputs: Tensor
            Input tensor

        # Returns
        Tensor
            Output tensor. It must be of shape (batch_size, *).
            Perform single detector with lens operation on the input tensor.
        """
        assert inputs.medium == Medium.LIGHT, "Medium must be LIGHT."

        # Perform lens operation
        # Assume the lens is perfect and the detector is placed
        # at the focal point of the lens

        intensity = torch.sum(inputs.elem * inputs.elem.conj(), dim=[-1, -2])
        return MediumTensor(
            intensity,
            medium=Medium.LIGHT,
            wavelengths=inputs.wavelengths,
            color_dim=inputs.color_dim,
        )

    def __str__(self):
        return f"SingleDetectorWithLens ({self.name})"
