"""Class-specific Grad-CAM for the existing E6 ConvNeXt-Tiny model only."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image

from .model_adapter import CANONICAL_CLASSES, E6ModelAdapter


def generate_gradcam_overlay(
    adapter: E6ModelAdapter,
    image: Image.Image,
    target_class: str,
    *,
    alpha: float = 0.42,
) -> Image.Image:
    """Create an AI-attention overlay using the selected E6 class logit.

    `model.features[-1]` is the final ConvNeXt convolutional feature stage.  The
    model itself is never changed; activation and gradient hooks exist only for
    this one explanation call.
    """
    if target_class not in CANONICAL_CLASSES:
        raise ValueError(f"Unsupported E6 target class: {target_class}")
    if not adapter.is_connected:
        raise RuntimeError(adapter.status.message)

    model = adapter.require_model()
    target_layer = model.features[-1]
    captured: dict[str, torch.Tensor] = {}

    def capture_activation(_module, _inputs, output: torch.Tensor) -> None:
        output.retain_grad()
        captured["activation"] = output

    handle = target_layer.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        inputs = adapter.prepare_image(image)
        logits = model(inputs)
        logits[0, CANONICAL_CLASSES.index(target_class)].backward()
        activation = captured.get("activation")
        if activation is None or activation.grad is None:
            raise RuntimeError("Grad-CAM could not capture final ConvNeXt activations and gradients.")
        gradients = activation.grad
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam = cam.detach().float().cpu().numpy()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

    maximum = float(cam.max())
    if maximum > 0.0:
        cam = cam / maximum
    else:
        cam = np.zeros_like(cam)

    original = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    heatmap = colormaps["jet"](cam)[..., :3]
    blended = np.clip((1.0 - alpha) * original + alpha * heatmap, 0.0, 1.0)
    return Image.fromarray((blended * 255).astype(np.uint8), mode="RGB")
