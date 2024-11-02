import torch

from pyluxe.enum import Medium
from pyluxe.typing import MediumTensor


def test_medium_tensor():  # noqa: C901
    # Test MediumTensor
    tensor = torch.rand(1, 3, 100, 100)
    medium_tensor = MediumTensor(
        tensor, Medium.ELECTRON, wavelengths=[1.0, 2.0, 3.0], color_dim=1
    )
    assert medium_tensor.medium == Medium.ELECTRON
    assert torch.allclose(medium_tensor.elem, tensor)

    assert tensor.shape == medium_tensor.shape

    # Test MediumTensor with wrong parameters
    try:
        MediumTensor(tensor, Medium.ELECTRON, wavelengths=[1.0])
    except AssertionError as e:
        assert (
            str(e)
            == "The number of colors must be the same as "
            + "the number of wavelengths."
        )

    # Test MediumTensor operations
    tensor2 = torch.rand(1, 3, 100, 100)
    medium_tensor2 = MediumTensor(tensor2, Medium.ELECTRON, [1.0, 2.0, 3.0], 1)
    assert torch.allclose(
        (medium_tensor + medium_tensor2).elem, tensor + tensor2
    )
    assert torch.allclose(
        (medium_tensor - medium_tensor2).elem, tensor - tensor2
    )
    assert torch.allclose(
        (medium_tensor * medium_tensor2).elem, tensor * tensor2
    )
    assert torch.allclose(
        (medium_tensor / medium_tensor2).elem, tensor / tensor2
    )
    assert torch.allclose(
        (medium_tensor @ medium_tensor2).elem, tensor @ tensor2
    )

    # Test MediumTensor operations with MediumTensor on different medium
    medium_tensor2 = MediumTensor(tensor2, Medium.LIGHT, [1.0, 2.0, 3.0])
    try:
        medium_tensor + medium_tensor2
    except AssertionError as e:
        assert str(e) == "Medium must be the same."
    try:
        medium_tensor - medium_tensor2
    except AssertionError as e:
        assert str(e) == "Medium must be the same."
    try:
        medium_tensor * medium_tensor2
    except AssertionError as e:
        assert str(e) == "Medium must be the same."
    try:
        medium_tensor / medium_tensor2
    except AssertionError as e:
        assert str(e) == "Medium must be the same."
    try:
        medium_tensor @ medium_tensor2
    except AssertionError as e:
        assert str(e) == "Medium must be the same."

    # Test MediumTensor operations with MediumTensor on different wavelengths
    medium_tensor2 = MediumTensor(tensor2, Medium.ELECTRON, [1.0, 2.0, 4.0])
    try:
        medium_tensor + medium_tensor2
    except AssertionError as e:
        assert str(e) == "Wavelengths must be the same."
    try:
        medium_tensor - medium_tensor2
    except AssertionError as e:
        assert str(e) == "Wavelengths must be the same."
    try:
        medium_tensor * medium_tensor2
    except AssertionError as e:
        assert str(e) == "Wavelengths must be the same."
    try:
        medium_tensor / medium_tensor2
    except AssertionError as e:
        assert str(e) == "Wavelengths must be the same."
    try:
        medium_tensor @ medium_tensor2
    except AssertionError as e:
        assert str(e) == "Wavelengths must be the same."

    # Test MediumTensor operations with Tensor
    assert (
        torch.allclose((medium_tensor + tensor2).elem, tensor + tensor2)
        and medium_tensor.medium == Medium.ELECTRON
    )
    assert (
        torch.allclose((medium_tensor - tensor2).elem, tensor - tensor2)
        and medium_tensor.medium == Medium.ELECTRON
    )
    assert (
        torch.allclose((medium_tensor * tensor2).elem, tensor * tensor2)
        and medium_tensor.medium == Medium.ELECTRON
    )
    assert (
        torch.allclose((medium_tensor / tensor2).elem, tensor / tensor2)
        and medium_tensor.medium == Medium.ELECTRON
    )
    assert (
        torch.allclose((medium_tensor @ tensor2).elem, tensor @ tensor2)
        and medium_tensor.medium == Medium.ELECTRON
    )

    # Test MediumTensor operations with scalar
    scalar = 2.0
    assert torch.allclose((medium_tensor + scalar).elem, tensor + scalar)
    assert torch.allclose((medium_tensor - scalar).elem, tensor - scalar)
    assert torch.allclose((medium_tensor * scalar).elem, tensor * scalar)
    assert torch.allclose((medium_tensor / scalar).elem, tensor / scalar)

    # Test MediumTensor operations with scalar
    assert torch.allclose((scalar + medium_tensor).elem, scalar + tensor)
    assert torch.allclose((scalar - medium_tensor).elem, scalar - tensor)
    assert torch.allclose((scalar * medium_tensor).elem, scalar * tensor)
    assert torch.allclose((scalar / medium_tensor).elem, scalar / tensor)

    # Squeeze
    squeezed_tensor = medium_tensor.squeeze(0)
    assert (
        torch.allclose(squeezed_tensor.elem, tensor.squeeze(0))
        and squeezed_tensor.medium == Medium.ELECTRON
        and squeezed_tensor.wavelengths == [1.0, 2.0, 3.0]
        and squeezed_tensor.color_dim == 0
    )

    squeezed_tensor = torch.squeeze(medium_tensor, 0)
    assert (
        torch.allclose(squeezed_tensor.elem, tensor.squeeze(0))
        and squeezed_tensor.medium == Medium.ELECTRON
        and squeezed_tensor.wavelengths == [1.0, 2.0, 3.0]
        and squeezed_tensor.color_dim == 0
    )

    # Unsqueeze
    unsqueezed_tensor = medium_tensor.unsqueeze(0)
    assert (
        torch.allclose(unsqueezed_tensor.elem, tensor.unsqueeze(0))
        and unsqueezed_tensor.medium == Medium.ELECTRON
        and unsqueezed_tensor.wavelengths == [1.0, 2.0, 3.0]
        and unsqueezed_tensor.color_dim == 2
    )

    unsqueezed_tensor = torch.unsqueeze(medium_tensor, 0)
    assert (
        torch.allclose(unsqueezed_tensor.elem, tensor.unsqueeze(0))
        and unsqueezed_tensor.medium == Medium.ELECTRON
        and unsqueezed_tensor.wavelengths == [1.0, 2.0, 3.0]
        and unsqueezed_tensor.color_dim == 2
    )

    # Transpose
    transposed_tensor = medium_tensor.transpose(1, 2)
    assert (
        torch.allclose(transposed_tensor.elem, tensor.transpose(1, 2))
        and transposed_tensor.medium == Medium.ELECTRON
        and transposed_tensor.wavelengths == [1.0, 2.0, 3.0]
        and transposed_tensor.color_dim == 2
    )

    transposed_tensor = torch.transpose(medium_tensor, 1, 2)
    assert (
        torch.allclose(transposed_tensor.elem, tensor.transpose(1, 2))
        and transposed_tensor.medium == Medium.ELECTRON
        and transposed_tensor.wavelengths == [1.0, 2.0, 3.0]
        and transposed_tensor.color_dim == 2
    )
