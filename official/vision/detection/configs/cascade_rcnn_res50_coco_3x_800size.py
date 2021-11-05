# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from megengine import hub

from official.vision.detection import models

def cascade_rcnn_res50_coco_3x_800size(**kwargs):
    r"""
    Cascade-RCNN FPN trained from COCO dataset.
    `"Cascade-RCNN" <https://arxiv.org/abs/1712.00726>`_
    `"FPN" <https://arxiv.org/abs/1612.03144>`_
    `"COCO" <https://arxiv.org/abs/1405.0312>`_
    """
    cfg = models.CascadeRCNNConfig()
    cfg.backbone_pretrained = False
    return models.CascadeRCNN(cfg, **kwargs)


Net = models.CascadeRCNN
Cfg = models.CascadeRCNNConfig
