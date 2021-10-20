from detectron2.utils.registry import Registry

BACKBONE_BOTTOMUP_FUSE_REGISTRY = Registry("BACKBONE_BOTTOMUP_FUSE")


def build_bottom_up_fuse(cfg):
    bottom_up_fuse_name = cfg.MODEL.BOTTOMUP_FUSE.NAME
    bottom_up_fuse = BACKBONE_BOTTOMUP_FUSE_REGISTRY.get(bottom_up_fuse_name)(cfg)
    return bottom_up_fuse



