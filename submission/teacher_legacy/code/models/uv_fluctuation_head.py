"""Small spatial corrections to frozen point fluctuations in physical u/v units."""
import torch
from torch import nn
from torch.nn import functional as F


class UVFluctuationHead(nn.Module):
    def __init__(self, matrix: bool, input_channels: int = 210):
        super().__init__()
        self.matrix = matrix
        self.projection = nn.Conv2d(input_channels, 4 if matrix else 1, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, prediction, feature, target_std):
        coefficient = .25 * torch.tanh(self.projection(feature))
        if coefficient.shape[-2:] != prediction.shape[2:4]:
            coefficient = F.interpolate(coefficient, size=prediction.shape[2:4],
                                        mode="bilinear", align_corners=False)
        uv = prediction[..., :2]
        std = target_std[:2].to(uv)
        fluctuation = (uv - uv.mean(dim=1, keepdim=True)) * std
        if self.matrix:
            batch, _, height, width = coefficient.shape
            matrix = coefficient.reshape(batch, 2, 2, height, width).permute(0, 3, 4, 1, 2)
            physical_delta = torch.einsum("bhwij,bthwj->bthwi", matrix, fluctuation)
        else:
            physical_delta = coefficient.permute(0, 2, 3, 1).unsqueeze(1) * fluctuation
        # An unconditional residual addition preserves the initial head gradient.
        return torch.cat((uv + physical_delta / std, prediction[..., 2:]), dim=-1)


def frozen_point_and_feature(model, inputs):
    """Capture the controller's existing decoder input from one frozen forward."""
    captured = []
    hook = model.energy_controller.register_forward_pre_hook(
        lambda _module, arguments: captured.append(arguments[0].detach()))
    try:
        with torch.no_grad():
            prediction = model(inputs).detach()
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError("expected exactly one frozen energy-controller call")
    return prediction, captured[0]
