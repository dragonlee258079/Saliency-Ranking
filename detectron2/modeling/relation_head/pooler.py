import torch
from torch import nn
from detectron2.layers import ROIAlign, cat


def convert_boxes_to_pooler_format(box_lists):
    def fmt_box_list(box_tensor, batch_index):
        repeated_index = torch.full(
            (len(box_tensor), 1), batch_index, dtype=box_tensor.dtype, device=box_tensor.device
        )
        return cat((repeated_index, box_tensor), dim=1)

    pooler_fmt_boxes = cat(
        [fmt_box_list(box_list.tensor, i) for i, box_list in enumerate(box_lists)], dim=0
    )

    return pooler_fmt_boxes


class Pooler(nn.Module):
    def __init__(
            self,
            output_size,
            scales,
            sampling_ratio,
    ):
        super().__init__()
