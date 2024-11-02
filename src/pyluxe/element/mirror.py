import torch

from pyluxe.element import Element


class HalfMirror(Element):
    """
    Half mirror element.
    """

    def __init__(self, reflectivity: float = 0.5, **kwargs):
        """
        # Parameters
        name: Optional[str]
            Name of the element.
        reflectivity: float
            Reflectivity of the mirror. It must be in the range [0, 1].
        """
        assert (
            0 <= reflectivity <= 1
        ), "Reflectivity must be in the range [0, 1]."  # noqa: E501

        super().__init__(**kwargs)
        self.reflectivity = reflectivity

    def forward(self, inputs: torch.Tensor):
        """
        # Parameters
        inputs: Tensor
            Input tensor

        # Returns
        Tensor
            Output tensor. It must be of shape (batch_size, 2, *).
            Create a pair of copy of the input tensor and multiply it
            by the reflectivity.
        """
        # Clone the input tensor
        # Tensor.clone() is needed to create a new tensor with the same data
        # as well as its gradients
        cloned = inputs.clone()
        return torch.stack(
            [inputs * (1 - self.reflectivity), cloned * self.reflectivity],
            dim=1,
        )

    def __str__(self):
        return f"HalfMirror ({self.name})"


class FullMirror(Element):
    """
    Full mirror element.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs: torch.Tensor):
        """
        # Parameters
        inputs: Tensor
            Input tensor

        # Returns
        Tensor
            Output tensor. It must be of shape(batch_size, *).
            This is identical to the input tensor.
        """
        return inputs

    def __str__(self):
        return f"FullMirror ({self.name})"
