import torch
import torch.nn as nn
from .mask_predictor import SimpleDecoding
from .backbone_swin import MultiModalSwinTransformer
from ._utils import LAVT, LAVTOne

__all__ = ['lavt_one']

###############################################
# LAVT One: put BERT inside the overall model #
###############################################
def _segm_lavt_one(pretrained, args):
    embed_dim = 128
    backbone = MultiModalSwinTransformer(
        pretrain_img_size=480,
        patch_size=4,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_path_rate=0.2,
        fusion_drop=args.fusion_drop
    )
    backbone.init_weights(args.pretrained_swin_weights)

    model_map = [SimpleDecoding, LAVTOne]
    classifier = model_map[0](8 * embed_dim)
    base_model = model_map[1]

    model = base_model(backbone, classifier, args)
    return model

def _load_model_lavt_one(pretrained, args):
    model = _segm_lavt_one('', args)
    return model

def lavt_one(pretrained='', args=None):
    return _load_model_lavt_one('', args)
