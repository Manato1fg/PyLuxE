from typing import List

import torch

from pyluxe.typing import MediumTensor


def concat_wavelengths(tensors: List[MediumTensor]):
    """
    Concatenate the wavelengths of the tensors.
    Raise an error if there are the same wavelengths
    because it may occur unexpected behavior.
    """
    wavelengths = []
    for tensor in tensors:
        wavelengths.extend(tensor.wavelengths)
    assert len(set(wavelengths)) == len(
        wavelengths
    ), "Wavelengths must be unique."

    # assert the specified color dim is valid
    color_dim = tensors[0].color_dim
    for tensor in tensors:
        assert tensor.color_dim == color_dim, "Color dim must be the same."
    # assert the medium is the same
    medium = tensors[0].medium
    for tensor in tensors:
        assert tensor.medium == medium, "Medium must be the same."

    new_tensor = MediumTensor(
        torch.cat([tensor.elem for tensor in tensors], dim=color_dim),
        tensors[0].medium,
        wavelengths,
    )
    return new_tensor
