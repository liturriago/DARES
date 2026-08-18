"""Shared helpers for the DARES backbone encoders."""

from abc import ABC, abstractmethod
from typing import OrderedDict, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.LayerNorm):
    """Layer normalization over the channel dimension of NCHW tensors.

    Args:
        normalized_shape (int | list[int]): Number of channels (or shape) to
            normalize over.
        eps (float): Small value added to the variance for numerical stability.
        elementwise_affine (bool): Whether to learn per-channel weight and bias.
        bias (bool): Whether to learn a per-channel bias (torch >= 2.1).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Encoder(nn.Module, ABC):
    """Abstract base class for all DARES backbone encoders.

    Every concrete encoder maps an input ``x`` of shape ``(B, C_in, H, W)`` to
    an ``OrderedDict[str, torch.Tensor]`` with exactly the keys ``"S2"``,
    ``"S4"``, ``"S8"``, ``"S16"`` and ``"S32"``, corresponding to feature maps
    at spatial strides 2, 4, 8, 16 and 32.

    Attributes:
        out_channels (dict[str, int]): Number of channels per feature level.
        strides (dict[str, int]): Spatial stride per feature level.
        pretrained (bool): Whether ImageNet-pretrained weights were loaded.
    """

    out_channels: dict[str, int]
    strides: dict[str, int]
    pretrained: bool

    @abstractmethod
    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Runs the encoder and returns multi-scale feature maps.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, C_in, H, W)`` with
                ``H`` and ``W`` multiples of 32.

        Returns:
            OrderedDict[str, torch.Tensor]: Feature maps keyed by spatial stride
                (``"S2"`` ... ``"S32"``), each of shape
                ``(B, C, H // stride, W // stride)``.
        """
        raise NotImplementedError

    @property
    def reference_params(self) -> list[nn.Parameter]:
        """Parameters of the deepest shared encoder block.

        Used by the DARES trust-region gradient balancing to anchor the
        auxiliary alignment gradient to the segmentation gradient (GradNorm-lite,
        see ``Docs/KimiReport.txt`` Section 4b). Concrete encoders override it to
        return the last stage / bottleneck block (e.g. ``layer4`` of a ResNet).
        """
        return list(self.parameters())


def adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Adapts a 3-channel pretrained convolution to ``in_channels``.

    The first three input-channel kernels are copied verbatim from ``conv`` and
    any additional channels are initialized with the mean of the three
    pretrained channels, preserving the original scale.

    Args:
        conv (nn.Conv2d): Source convolution (typically 3 input channels).
        in_channels (int): Desired number of input channels.

    Returns:
        nn.Conv2d: A new convolution with the same kernel size, stride, padding,
            dilation, groups and bias as ``conv``, with ``in_channels`` inputs.
            If ``conv.in_channels == in_channels`` the original module is
            returned unchanged.
    """
    if conv.in_channels == in_channels:
        return conv
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        cast(tuple[int, int] | int, conv.kernel_size),
        cast(tuple[int, int] | int, conv.stride),
        cast(str | int | tuple[int, int], conv.padding),
        cast(tuple[int, int] | int, conv.dilation),
        conv.groups,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = conv.weight
        if in_channels > 3:
            new_conv.weight[:, 3:] = conv.weight.mean(dim=1, keepdim=True)
        if conv.bias is not None:
            assert new_conv.bias is not None
            new_conv.bias.copy_(conv.bias)
    return new_conv


def _make_stride2_stem(
    in_channels: int, out_channels: int, activation: str = "gelu"
) -> nn.Sequential:
    """Builds a lightweight parallel stem producing the ``S2`` feature level.

    The stem is a ``Conv2d(in_channels, out_channels, 3, stride=2, padding=1)``
    followed by normalization and activation. It runs in parallel to the native
    backbone and feeds only the ``S2`` skip connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels of the stem.
        activation (str): ``"relu"`` for a ResNet-style stem (BatchNorm2d +
            ReLU) or ``"gelu"`` for a ConvNeXt/Swin-style stem (LayerNorm2d +
            GELU).

    Returns:
        nn.Sequential: The stride-2 stem module.

    Raises:
        ValueError: If ``activation`` is not ``"relu"`` or ``"gelu"``.
    """
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    ]
    if activation == "gelu":
        layers.append(LayerNorm2d(out_channels))
        layers.append(nn.GELU())
    elif activation == "relu":
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
    else:
        raise ValueError(
            f"Unknown stem activation {activation!r}; expected 'relu' or 'gelu'"
        )
    return nn.Sequential(*layers)
