"""
Model package for the DARES framework.

Architecture convention
-----------------------
Every segmentation model is a composition of a pluggable **backbone** (encoder)
and a pluggable **head** (decoder):

* ``dares.models.backbones``  ->  multi-scale feature extractors (ResNet50,
  ConvNeXt, Swin Transformer). Each returns an ``OrderedDict`` of feature maps
  with keys ``"S2"``, ``"S4"``, ``"S8"``, ``"S16"``, ``"S32"`` corresponding to
  spatial strides 2, 4, 8, 16 and 32.
* ``dares.models.heads``  ->  dense decoders (ResUNet, DeepLabV3+). Each consumes
  the backbone's ``OrderedDict`` and returns a tuple ``(features, logits)`` where
  both are at the full input resolution.
* ``dares.models.segmentation``  ->  the ``SegmentationModel`` wrapper combining
  backbone + head with dual forward modes ('class' / 'feature' / 'both').
"""
from dares.models.segmentation import SegmentationModel, build_model

__all__ = ["SegmentationModel", "build_model"]
