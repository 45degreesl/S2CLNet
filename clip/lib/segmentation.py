import torch
import torch.nn as nn
from .mask_predictor import SimpleDecoding
from .backbone_clip import MMModifiedResNet
from ._utils import LAVT, LAVTOne


__all__ = ['lavt', 'lavt_one']


# LAVT
def _segm_lavt(pretrained, args):
    # initialize the SwinTransformer backbone with the specified version
    vision_layers = [3, 4, 23, 3]
    embed_dim = 512
    vision_heads = 32
    input_resolution = 480
    backbone_CLIP = MMModifiedResNet(vision_layers, embed_dim, vision_heads, input_resolution)
    backbone_CLIP.init_weights(pretrained=pretrained)

    model_map = [SimpleDecoding, LAVT]

    classifier = model_map[0](2048)
    base_model = model_map[1]

    model = base_model(backbone_CLIP, classifier)
    return model


def _load_model_lavt(pretrained, args):
    model = _segm_lavt(pretrained, args)
    return model


def lavt(pretrained='', args=None):
    return _load_model_lavt(pretrained, args)


###############################################
# LAVT One: put BERT inside the overall model #
###############################################
def _segm_lavt_one(pretrained, args):
    # initialize the SwinTransformer backbone with the specified version
    vision_layers = [3, 4, 23, 3]
    embed_dim = 512
    vision_heads = 32
    input_resolution = 480
    backbone_CLIP = MMModifiedResNet(vision_layers, embed_dim, vision_heads, input_resolution)
    backbone_CLIP.init_weights(pretrained=pretrained)

    # todo: embed_dim?
    model_map = [SimpleDecoding, LAVTOne]
    classifier = model_map[0](2048)  # todo: related to the optimizer!
    base_model = model_map[1]

    model = base_model(backbone_CLIP, classifier, args)

    return model


def _load_model_lavt_one(pretrained, args):
    model = _segm_lavt_one(pretrained, args)
    return model


def lavt_one(pretrained='', args=None):
    return _load_model_lavt_one(pretrained, args)
