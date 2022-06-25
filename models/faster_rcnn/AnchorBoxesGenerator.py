from email.mime import base
import torch
from torch import nn

class AnchorBoxesGenerator(nn.Module):

    def __init__(self, sizes:tuple = (128, 256, 512), aspect_ratios:tuple = (0.5, 1.0, 2.0)) -> None:
        super(AnchorBoxesGenerator, self).__init__()

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    @staticmethod
    def generate_anchors(sizes, aspect_ratios, dtype = torch.float32, device = "cpu"):
        sizes = torch.as_tensor(sizes, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1.0 / h_ratios

        ws = (w_ratios[:, None] * sizes[None, :]).view(-1)
        hs = (h_ratios[:, None] * sizes[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

        return base_anchors.round()

    def set_cell_anchors(self, dtype, device):
        self.cell_anchors = [self.generate_anchors(sizes, aspect_ratios, dtype, device) for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)]

    def num_anchors_per_location(self):
        # calculate the number of anchors per feature map, for k in origin paper
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, feature_map_sizes, strides):
        anchors = []
        assert self.cell_anchors is not None

        for size, stride, base_anchor in zip(feature_map_sizes, stride, self.cell_anchors):
            f_p_height, f_p_width = size
            stride_height, stride_width = stride
            device = base_anchor.device

            shifts_x = torch.arange(0, f_p_width, dtype=torch.float32, device=device) * stride_width
            shifts_y = torch.arange(0, f_p_height, dtype=torch.float32, device=device) * stride_height

            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack([shifts_x, shifts_y, shifts_x, shifts_y], dim=1)

            shift_anchors = shifts.view(-1, 1, 4) + base_anchor.view(1, -1, 4)
            anchors.append(shift_anchors.reshape(-1,4))
        return anchors


    def cached_grid_anchors(self, feature_map_sizes, strides):
        key = str(feature_map_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(feature_map_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, images, feature_maps):
        feature_map_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]

        image_size = images.tensors.shape[-2:]

        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        strides = [[torch.tensor(image_size[0] / g[0], dtype=dtype, device=device),
                    torch.tensor(image_size[1] / g[1], dtype=dtype, device=device)] for g in feature_map_sizes]

        self.set_cell_anchors(dtype, device)

        anchors_over_all_feature_maps = self.cached_grid_anchors(feature_map_sizes, strides)
        anchors = []

        for i, (_, _) in enumerate(images.image_sizes):
            anchor_in_image = []

            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchor_in_image.append(anchors_per_feature_map)
            anchors.append(anchor_in_image)
        
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        self._cache.clear()
        return anchors
