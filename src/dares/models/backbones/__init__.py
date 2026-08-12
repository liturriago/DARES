"""
Backbone encoders for DARES.

Each backbone is a ``torch.nn.Module`` that maps an input tensor ``x`` of shape
``(B, C_in, H, W)`` to an ``OrderedDict[str, torch.Tensor]`` of multi-scale
feature maps. The dict keys are the spatial strides of each level:

* ``"S2"``  ->  feature map at 1/2 resolution
* ``"S4"``  ->  feature map at 1/4 resolution
* ``"S8"``  ->  feature map at 1/8 resolution
* ``"S16"`` ->  feature map at 1/16 resolution
* ``"S32"`` ->  feature map at 1/32 resolution

Every backbone exposes:

* ``self.out_channels: dict[str, int]``  ->  number of channels per level.
* ``self.strides: dict[str, int]``       ->  spatial stride per level.
* ``self.pretrained: bool``              ->  whether ImageNet weights were loaded.

All backbones accept an arbitrary number of input channels (default 4 for the
DARES spectral configuration B2/B3/B4/B8). Pretrained 3-channel weights are
adapted to ``C_in`` by copying the pretrained kernel for the first 3 channels
and initializing the remaining channel(s) with the mean of the first three.
"""
