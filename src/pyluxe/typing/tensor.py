from typing import List, Union

import torch.utils._pytree as pytree
from torch import Tensor

from pyluxe.enum import Medium


class MediumTensor(Tensor):
    """
    ONN tensor class.
    This is a subclass of torch.Tensor, which has an additional attribute
    called medium.

    see: https://github.com/albanD/subclass_zoo
         https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/custom_tensor.py
    """

    def __new__(
        cls,
        elem: Tensor,
        medium: Medium,
        wavelengths: List[float],
        color_dim: int = 1,
        *args,
        **kwargs,
    ):
        assert elem.size(color_dim) == len(wavelengths), (
            "The number of colors must be the "
            + "same as the number of wavelengths."
        )
        obj = Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]  # noqa: E501
            cls,
            size=elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
        )
        obj.elem = elem
        obj.medium = medium
        obj.wavelengths = wavelengths
        obj.color_dim = color_dim
        return obj

    def __init__(
        self,
        elem: Tensor,
        medium: Medium,
        wavelengths: List[float],
        color_dim: int = 1,
    ):
        self.elem = elem
        self.medium = medium
        self.wavelengths = wavelengths
        self.color_dim = color_dim

    def __repr__(self):
        inner_repr = repr(self.elem)
        return f"MediumTensor({inner_repr})"

    def __tensor_flatten__(self):
        return ["elem"], (self.medium, self.wavelengths, self.color_dim)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta):
        print(meta)
        return MediumTensor(*inner_tensors, *meta)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_inner = pytree.tree_map_only(MediumTensor, lambda x: x.elem, args)

        kwargs_inner = pytree.tree_map_only(
            MediumTensor, lambda x: x.elem, kwargs
        )

        result = func(*args_inner, **kwargs_inner)
        color_dim = args[0].color_dim

        if func.__name__ == "squeeze.dim":
            color_dim = args[1] if len(args) > 1 else kwargs.get("dim", None)
            current_color_dim = args[0].color_dim
            if color_dim < current_color_dim:
                color_dim = current_color_dim - 1

        if func.__name__ == "unsqueeze.default":
            color_dim = args[1]
            current_color_dim = args[0].color_dim
            if color_dim <= current_color_dim:
                color_dim = current_color_dim + 1

        if func.__name__ == "transpose.int":
            dim0, dim1 = args[1], args[2]
            if dim0 == color_dim:
                color_dim = dim1
            elif dim1 == color_dim:
                color_dim = dim0

        return MediumTensor(
            result, args[0].medium, args[0].wavelengths, color_dim
        )

    def __add__(self, other: Union[Tensor, "MediumTensor", float, int]):
        if isinstance(other, MediumTensor):
            assert self.medium == other.medium, "Medium must be the same."
            assert (
                self.wavelengths == other.wavelengths
            ), "Wavelengths must be the same."
            return MediumTensor(
                self.elem + other.elem, self.medium, self.wavelengths
            )
        return MediumTensor(self.elem + other, self.medium, self.wavelengths)

    def __sub__(self, other: Union[Tensor, "MediumTensor", float, int]):
        if isinstance(other, MediumTensor):
            assert self.medium == other.medium, "Medium must be the same."
            assert (
                self.wavelengths == other.wavelengths
            ), "Wavelengths must be the same."
            return MediumTensor(
                self.elem - other.elem, self.medium, self.wavelengths
            )
        return MediumTensor(self.elem - other, self.medium, self.wavelengths)

    def __mul__(self, other: Union[Tensor, "MediumTensor", float, int]):
        if isinstance(other, MediumTensor):
            assert self.medium == other.medium, "Medium must be the same."
            assert (
                self.wavelengths == other.wavelengths
            ), "Wavelengths must be the same."
            return MediumTensor(
                self.elem * other.elem, self.medium, self.wavelengths
            )
        return MediumTensor(self.elem * other, self.medium, self.wavelengths)

    def __truediv__(self, other: Union[Tensor, "MediumTensor", float, int]):
        if isinstance(other, MediumTensor):
            assert self.medium == other.medium, "Medium must be the same."
            assert (
                self.wavelengths == other.wavelengths
            ), "Wavelengths must be the same."
            return MediumTensor(
                self.elem / other.elem, self.medium, self.wavelengths
            )
        return MediumTensor(self.elem / other, self.medium, self.wavelengths)

    def __matmul__(self, other: Union[Tensor, "MediumTensor", float, int]):
        if isinstance(other, MediumTensor):
            assert self.medium == other.medium, "Medium must be the same."
            assert (
                self.wavelengths == other.wavelengths
            ), "Wavelengths must be the same."
            return MediumTensor(
                self.elem @ other.elem, self.medium, self.wavelengths
            )
        return MediumTensor(self.elem @ other, self.medium, self.wavelengths)

    def __radd__(self, other: Union[Tensor, "MediumTensor", float, int]):
        if isinstance(other, MediumTensor):
            assert self.medium == other.medium, "Medium must be the same."
            assert (
                self.wavelengths == other.wavelengths
            ), "Wavelengths must be the same."
            return MediumTensor(
                other.elem + self.elem, self.medium, self.wavelengths
            )
        return MediumTensor(other + self.elem, self.medium, self.wavelengths)

    def __rsub__(self, other: Union[Tensor, "MediumTensor", float, int]):
        if isinstance(other, MediumTensor):
            assert self.medium == other.medium, "Medium must be the same."
            assert (
                self.wavelengths == other.wavelengths
            ), "Wavelengths must be the same."
            return MediumTensor(
                other.elem - self.elem, self.medium, self.wavelengths
            )
        return MediumTensor(other - self.elem, self.medium, self.wavelengths)

    def __rmul__(self, other: Union[Tensor, "MediumTensor", float, int]):
        if isinstance(other, MediumTensor):
            assert self.medium == other.medium, "Medium must be the same."
            assert (
                self.wavelengths == other.wavelengths
            ), "Wavelengths must be the same."
            return MediumTensor(
                other.elem * self.elem, self.medium, self.wavelengths
            )
        return MediumTensor(other * self.elem, self.medium, self.wavelengths)

    def __rtruediv__(self, other: Union[Tensor, "MediumTensor", float, int]):
        if isinstance(other, MediumTensor):
            assert self.medium == other.medium, "Medium must be the same."
            assert (
                self.wavelengths == other.wavelengths
            ), "Wavelengths must be the same."
            return MediumTensor(
                other.elem / self.elem, self.medium, self.wavelengths
            )
        return MediumTensor(other / self.elem, self.medium, self.wavelengths)

    def __rmatmul__(self, other: Union[Tensor, "MediumTensor", float, int]):
        if isinstance(other, MediumTensor):
            assert self.medium == other.medium, "Medium must be the same."
            assert (
                self.wavelengths == other.wavelengths
            ), "Wavelengths must be the same."
            return MediumTensor(
                other.elem @ self.elem, self.medium, self.wavelengths
            )
        return MediumTensor(other @ self.elem, self.medium, self.wavelengths)

    def __iadd__(self, other: Union[Tensor, "MediumTensor", float, int]):
        if isinstance(other, MediumTensor):
            assert self.medium == other.medium, "Medium must be the same."
            assert (
                self.wavelengths == other.wavelengths
            ), "Wavelengths must be the same."
            self.elem += other.elem
        else:
            self.elem += other
        return self

    def __isub__(self, other: Union[Tensor, "MediumTensor", float, int]):
        if isinstance(other, MediumTensor):
            assert self.medium == other.medium, "Medium must be the same."
            assert (
                self.wavelengths == other.wavelengths
            ), "Wavelengths must be the same."
            self.elem -= other.elem
        else:
            self.elem -= other
        return self

    def __imul__(self, other: Union[Tensor, "MediumTensor", float, int]):
        if isinstance(other, MediumTensor):
            assert self.medium == other.medium, "Medium must be the same."
            assert (
                self.wavelengths == other.wavelengths
            ), "Wavelengths must be the same."
            self.elem *= other.elem
        else:
            self.elem *= other
        return self

    def __imatmul__(self, other: Union[Tensor, "MediumTensor", float, int]):
        if isinstance(other, MediumTensor):
            assert self.medium == other.medium, "Medium must be the same."
            assert (
                self.wavelengths == other.wavelengths
            ), "Wavelengths must be the same."
            self.elem @= other.elem
        else:
            self.elem @= other
        return self

    def __neg__(self):
        return MediumTensor(-self.elem, self.medium, self.wavelengths)

    def __pos__(self):
        return MediumTensor(+self.elem, self.medium, self.wavelengths)

    def __abs__(self):
        return MediumTensor(abs(self.elem), self.medium, self.wavelengths)
