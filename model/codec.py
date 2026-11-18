"""
AIR as an image codec: the predicted Gaussians, optionally optimised within a time or step budget, quantised with
AIR's quantiser and written as a bitstream (model/bitstream.py) that renders without the network.

AIRNet predicts the Gaussians of a picture in one feed-forward pass and its FeedForwardGaussianCodec quantises
them, but AIR only estimates the coded size. This module codes a picture end to end:

    encode   predict(model, image)        AIRNet.forward's stage loop in inference: the raw Gaussians of every kept
                                          candidate, their patch centres, the stage masks and the class token
             quantiser(model, cls)        the per-picture quantiser: steps and offsets from the class token
             optimize(...)                optional: Adam on the raw Gaussians, through the quantiser
             to_bitstream(...)            activation, quantisation, bitstream
    decode   decode(data)                 bitstream -> dequantised Gaussians -> AIR's rasteriser: RGB [3, H, W]

The optimisation runs Adam on AIR's raw (pre-activation) Gaussian parameters, so positions stay within a patch of
their centre and scales within AIR's range, with AIR's renderer and loss (0.7 L1 + 0.3 (1 - SSIM), as
render_pseudo), through AIR's quantiser with straight-through gradients (its steps and offsets fixed at the
prediction's), and divides the learning rate by `decay_ratio` at `decay_time_ratio` of its time. Without the
quantiser in the loop the parameters drift off the quantiser's grid and the coded picture gets worse after a few
hundred steps. The Gaussian count stays the prediction's.

    data, info = encode(model, image, max_time=2.0)   # image [1, 3, H, W] in [0, 1]; 2 s for the whole encode
    img = decode(data)                                # [3, H, W] float on the GPU, not clamped

The budget of encode is wall-clock time for everything, the prediction included: the optimisation gets what the
prediction leaves (budget 0 = the predicted Gaussians as they are). Rendering is reproducible when the rasteriser
sorts the tile intersections with a stable sort (all depths are 0, so the order within a tile, and with it which
256 Gaussians a crowded tile renders, is the Gaussian index).
"""

import math
import time
from typing import NamedTuple

import numpy as np
import torch
from fused_ssim import fused_ssim
from gsplat import project_gaussians_2d_scale_rot, rasterize_gaussians_sum

from . import bitstream
from .quantize import rgb_to_ycrcb, ycrcb_to_rgb
from .utils import pixel_to_xy

BLOCK = 16  # the rasteriser's tile size
LOSS = "0.7 l1 + 0.3 (1 - ssim)"


class Prediction(NamedTuple):
    raw: torch.Tensor      # [N, 8] raw Gaussians of all kept candidates, in stage order then patch order
    centers: torch.Tensor  # [N, 2] their patch centres (pixels)
    masks: np.ndarray      # [stages, L] bool: the kept candidates of each stage on the patch grid
    cls: torch.Tensor      # [1, C] class token of the last stage (drives the quantiser)


def predict(model, image: torch.Tensor, stages: int = None, autocast_dtype=None) -> Prediction:
    """AIR's stage loop in inference (AIRNet.forward without POD and without its quantised re-render). Stage
    Control keeps the candidates of the patches below 35 dB PSNR or 0.95 SSIM; a stage that would keep none keeps
    all, as AIRNet.forward does. `autocast_dtype` (torch.bfloat16 / float16) runs the heads under autocast."""
    H, W = image.shape[-2:]
    stages = stages or model.head_num
    if not 1 <= stages <= model.head_num:
        raise ValueError(f"stages must be in 1..{model.head_num}, got {stages}")
    render_img = torch.zeros_like(image)
    gs, cs, masks = [], [], []
    cls = None
    with torch.inference_mode():
        for s in range(stages):
            mse_err = model._mse_error(image, render_img)
            ssim_err = model._ssim_error(image, render_img)
            mask = torch.logical_or(mse_err > model.mse_threshold, ssim_err > model.ssim_threshold).flatten(1)[0]
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                g, c, cls = model.gaussian_heads[s](image - render_img)
            if not bool(mask.any()):
                mask = torch.ones_like(mask)
            gs.append(g[0][mask].float())
            cs.append(c[0][mask].float())
            masks.append(mask.cpu().numpy())
            if s + 1 < stages:  # the render only feeds the next stage (Stage Control and the residual)
                render_img = model._gaussian_to_image([torch.cat(gs)], [torch.cat(cs)], H, W, render_mode="image")["image"]
    return Prediction(torch.cat(gs).clone(), torch.cat(cs).clone(), np.stack(masks), cls.float().clone())


def quantiser(model, cls: torch.Tensor) -> dict:
    """The per-picture quantiser of FeedForwardGaussianCodec (UniformQuantizer.forward's step and offset): step,
    offset, qmax [8] tensors and bits [8] ints, in the order of bitstream.CHANNELS."""
    F = torch.nn.functional
    codec = model.feed_forward_gaussian_codec
    steps, offsets, qmax, bits = [], [], [], []
    with torch.inference_mode():
        for qz in (codec.xy_quantizer, codec.scaling_quantizer, codec.rotation_quantizer, codec.color_quantizer):
            off = 0.5 * torch.tanh(qz.offset_net(cls).reshape(1, qz.num_channels, 2))
            steps.append(F.softplus(qz.raw_scale.to(off.device) + off[0, :, 0]) + qz.eps)
            offsets.append(qz.beta.to(off.device) + off[0, :, 1])
            qmax.append(qz.qmax_per_channel.to(off.device).float())
            bits += [int(b) for b in qz.bits_per_channel.tolist()]
            if float(qz.qmin_per_channel.min()) != 0.0:
                raise ValueError("AIR's quantisers are unsigned")
    cat = lambda xs: torch.cat(xs).float().clone()  # noqa: E731
    return {"step": cat(steps), "offset": cat(offsets), "qmax": cat(qmax), "bits": bits}


def activate(raw: torch.Tensor, patch_size) -> torch.Tensor:
    """Raw Gaussians [N, 8] -> the quantised domain of FeedForwardGaussianCodec: offset from the patch centre
    (pixels), scale (pixels), rotation (radians), YCrCb colour."""
    ps = float(patch_size)
    xy = torch.tanh(raw[:, 0:2]) * ps
    scale = torch.sigmoid(raw[:, 2:4]) * ps + 0.5
    rot = torch.sigmoid(raw[:, 4:5]) * (2.0 * math.pi)
    color = rgb_to_ycrcb(torch.tanh(raw[:, 5:8]))
    return torch.cat([xy, scale, rot, color], dim=1)


def quantize(vals: torch.Tensor, q: dict) -> torch.Tensor:
    """Values in the quantised domain -> symbols (float, integral), as UniformQuantizer rounds them."""
    return torch.round(torch.clamp((vals - q["offset"]) / q["step"], min=0.0).minimum(q["qmax"]))


def render_values(vals: torch.Tensor, centers: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Gaussians in the quantised domain [N, 8] (dequantised or not) -> image [3, H, W], as AIR renders its
    quantised Gaussians (FeedForwardGaussianCodec.forward + utils.render)."""
    xy = pixel_to_xy(vals[:, 0:2] + centers, H, W)
    color = ycrcb_to_rgb(vals[:, 5:8])
    tile_bounds = ((W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK, 1)
    xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
        xy.contiguous(), vals[:, 2:4].contiguous(), vals[:, 4:5].contiguous(), H, W, tile_bounds)
    opacity = torch.ones_like(vals[:, 4:5])
    img = rasterize_gaussians_sum(xys, depths, radii, conics, num_tiles_hit, color.contiguous(), opacity, H, W, BLOCK, BLOCK)
    return img.view(H, W, 3).permute(2, 0, 1)


def optimize(raw, centers, target, q, patch_size, max_time=0.0, max_steps=0, lr=1e-2, decay_ratio=10.0,
             decay_time_ratio=0.8333, quant_aware=True):
    """Adam on the raw Gaussians [N, 8] against `target` [3, H, W] for `max_time` seconds of wall-clock, or
    `max_steps` steps when > 0. A time budget takes no step that the duration of the previous one says would end
    past it. Returns (raw Gaussians, steps taken)."""
    H, W = target.shape[-2:]
    u = raw.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    target4 = target[None].contiguous()
    budget = max_steps if max_steps > 0 else max_time
    decay_at = decay_time_ratio * budget
    decayed = decay_ratio == 1.0
    step, last = 0, 0.0
    t_start = time.perf_counter()
    while True:
        t_step = time.perf_counter()
        used = step if max_steps > 0 else t_step - t_start
        if used + (0 if max_steps > 0 else last) >= budget:
            break
        if not decayed and used >= decay_at:
            for group in opt.param_groups:
                group["lr"] = lr / decay_ratio
            decayed = True
        vals = activate(u, patch_size)
        if quant_aware:
            deq = quantize(vals, q) * q["step"] + q["offset"]
            vals = vals + (deq - vals).detach()
        img = render_values(vals, centers, H, W)
        loss = 0.7 * (img - target).abs().mean() + 0.3 * (1.0 - fused_ssim(img[None], target4))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        last = time.perf_counter() - t_step  # the rasteriser synchronises within a step, so this is GPU time too
    return u.detach(), step


def to_bitstream(raw, masks, q, patch_size, width, height) -> bytes:
    """Quantise raw Gaussians [N, 8] (in the order of the masks) and write the bitstream."""
    with torch.no_grad():
        symbols = quantize(activate(raw, patch_size), q).to(torch.int64).cpu().numpy()
    return bitstream.encode(width, height, int(patch_size), q["bits"], q["step"].cpu().numpy(),
                            q["offset"].cpu().numpy(), masks, symbols)


def encode(model, image, max_time=0.0, max_steps=0, stages=None, autocast_dtype=None, t_start=None,
           reserve_per_gaussian=0.0, **optimize_kw):
    """Code a picture [1, 3, H, W] in [0, 1] within a wall-clock budget: `max_time` seconds for the whole encode
    (prediction, optimisation, bitstream), counted from `t_start` (time.perf_counter(); default: this call). The
    optimisation gets what the prediction leaves, less the expected time of writing the bitstream
    (`reserve_per_gaussian` seconds per Gaussian, e.g. measured on an earlier picture); budget 0, or a prediction
    that takes the whole budget, codes the predicted Gaussians as they are. `max_steps` > 0 runs that many
    optimisation steps instead, whatever the time. `optimize_kw` go to optimize (lr, decay_ratio, ...).
    Returns (bytes, info): num_gaussians, steps, stage_gaussians, candidates_per_stage, and the seconds
    t_predict, t_optimize, t_bitstream and t_total (from t_start)."""
    t0 = time.perf_counter() if t_start is None else t_start
    H, W = image.shape[-2:]
    pred = predict(model, image, stages, autocast_dtype)
    q = quantiser(model, pred.cls)
    torch.cuda.synchronize(image.device)
    t1 = time.perf_counter()
    raw, steps = pred.raw, 0
    if max_steps > 0:
        raw, steps = optimize(raw, pred.centers, image[0], q, model.patch_size, max_steps=max_steps, **optimize_kw)
    elif max_time > 0:
        left = max_time - (t1 - t0) - reserve_per_gaussian * raw.shape[0]
        if left > 0:
            raw, steps = optimize(raw, pred.centers, image[0], q, model.patch_size, max_time=left, **optimize_kw)
    torch.cuda.synchronize(image.device)
    t2 = time.perf_counter()
    data = to_bitstream(raw, pred.masks, q, model.patch_size, W, H)
    t3 = time.perf_counter()
    return data, {"num_gaussians": int(raw.shape[0]), "steps": int(steps), "stage_gaussians": [int(m.sum()) for m in pred.masks],
                  "candidates_per_stage": int(pred.masks.shape[1]), "t_predict": t1 - t0, "t_optimize": t2 - t1,
                  "t_bitstream": t3 - t2, "t_total": t3 - t0}


def patch_centers(masks, width, height, patch_size, device="cuda") -> torch.Tensor:
    """The centres [N, 2] of the Gaussians of the stage masks (stage order, then patch order), as AIR's
    PatchEmbedding places them: the centre of the patch, clamped to the picture (float32)."""
    rows, cols = bitstream.grid(width, height, patch_size)
    idx = torch.from_numpy(np.concatenate([np.flatnonzero(m) for m in masks])).to(device)
    r, c = idx // cols, idx % cols
    cx = (c.float() * patch_size + (patch_size - 1) / 2.0).clamp(max=max(width - 1, 0))
    cy = (r.float() * patch_size + (patch_size - 1) / 2.0).clamp(max=max(height - 1, 0))
    return torch.stack([cx, cy], dim=-1)


def decode(data: bytes, device="cuda") -> torch.Tensor:
    """Read and render a bitstream: float RGB [3, H, W] on `device`, not clamped. Needs no network."""
    f = bitstream.decode(data)
    H, W, ps = f["height"], f["width"], f["patch_size"]
    centers = patch_centers(f["masks"], W, H, ps, device)
    sym = torch.from_numpy(f["symbols"]).to(device).float()
    step = torch.from_numpy(f["steps"]).to(device)
    offset = torch.from_numpy(f["offsets"]).to(device)
    with torch.no_grad():
        return render_values(sym * step + offset, centers, H, W)
