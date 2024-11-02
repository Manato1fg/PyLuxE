from abc import abstractmethod
from typing import Optional

from torch.nn import Module

from pyluxe.typing import MediumTensor


class Element(Module):
    """
    Abstract class for all elements in the ONN model.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        """
        # Parameters
        name: Optional[str]
            Name of the element.
        """
        super().__init__(**kwargs)
        self.name = name

    @abstractmethod
    def forward(self, inputs: MediumTensor) -> MediumTensor:
        """
        Forward pass of the element.
        """
        pass

    def __str__(self):
        return self.name
