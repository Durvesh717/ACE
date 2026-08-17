#!/usr/bin/env python3
"""Joint denoising + 2x super-resolution — submission entry point.

    python run.py <input-dir> <output-dir>

Reads every .npy in <input-dir>, restores it, and writes one .npy per input to
<output-dir> under the same filename. Each output is a grayscale float32 array
of shape (2H, 2W) with values in [0, 1] and no NaN/Inf.

Runs offline on an NVIDIA GPU when one is available and falls back to CPU
otherwise. No downloads, no API keys, no interactive prompts, no configuration:
the weights ship in models/ and every path is resolved relative to this file.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from restormer_arch import Restormer  # noqa: E402  (vendored, see file header)

WEIGHTS = os.path.join(HERE, "models", "v4_restormer.pt")
SCALE = 2


class RestormerSR(nn.Module):
    """Restormer body with a PixelShuffle x2 tail.

    Restormer restores at the input's own resolution, so the x2 head is ours. A
    global bilinear skip means the network predicts only a residual on top of a
    plain upsample, and PixelShuffle avoids the checkerboard artifacts that
    transposed convolution produces.

    The input offset is a buffer inside the model, so callers pass the raw
    degraded array and receive output already in the [0, 1] ground-truth domain.
    Being fully convolutional in extent, one set of weights serves 128->256 and
    256->512 alike.
    """

    def __init__(self, dim=32, num_blocks=(4, 6, 6, 8), heads=(1, 2, 4, 8),
                 ffn_expansion_factor=2.66, offset=0.5, feat=48):
        super().__init__()
        self.register_buffer("offset", torch.tensor(float(offset)))
        self.body = Restormer(
            inp_channels=1, out_channels=feat, dim=dim, num_blocks=list(num_blocks),
            num_refinement_blocks=4, heads=list(heads),
            ffn_expansion_factor=ffn_expansion_factor, bias=False,
            LayerNorm_type="WithBias",
        )
        self.tail = nn.Sequential(
            nn.Conv2d(feat, feat * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(feat, 1, 3, padding=1),
        )
        self.padding = 8  # the UNet downsamples three times

    def _pad(self, x):
        _, _, h, w = x.shape
        p = self.padding
        ph, pw = (p - h % p) % p, (p - w % p) % p
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, h, w

    def forward(self, x):
        x = x - self.offset
        x, h, w = self._pad(x)
        base = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.tail(self.body(x)) + base
        return out[:, :, : h * 2, : w * 2] + self.offset


def load_model(device):
    if not os.path.isfile(WEIGHTS):
        raise SystemExit(f"model weights not found at {WEIGHTS}")
    ck = torch.load(WEIGHTS, map_location=device, weights_only=False)
    dim = (ck.get("args") or {}).get("dim", 32)
    model = RestormerSR(dim=dim).to(device)
    model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
    model.eval()
    return model


def to_2d(arr, name):
    """Accept (H,W), (H,W,1) or (1,H,W); return (H,W) float32."""
    a = np.asarray(arr)
    if a.ndim == 3:
        if a.shape[-1] == 1:
            a = a[..., 0]
        elif a.shape[0] == 1:
            a = a[0]
        else:
            raise ValueError(f"{name}: expected grayscale, got shape {a.shape}")
    elif a.ndim != 2:
        raise ValueError(f"{name}: expected a 2-D array, got shape {a.shape}")
    return a.astype(np.float32, copy=False)


@torch.no_grad()
def restore(model, arr, device, use_amp):
    """arr: (H,W) degraded, raw and unclipped. Returns (2H,2W) float32 in [0,1].

    The input is deliberately NOT clipped: speckle is multiplicative, so a large
    fraction of pixels legitimately sit outside [0,1] and that overshoot is the
    model's best evidence of where the noise is. Only the OUTPUT is clamped,
    which is valid because the ground truth is per-image min-max normalized to
    exactly that range.
    """
    x = torch.from_numpy(np.ascontiguousarray(arr))[None, None].to(device)
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        y = model(x)
    y = y.float().clamp_(0.0, 1.0)
    y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    return y[0, 0].cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser(
        description="Restore degraded .npy images: denoise + 2x super-resolution."
    )
    ap.add_argument("input_dir", help="directory containing degraded .npy files")
    ap.add_argument("output_dir", help="directory for restored .npy files (created if absent)")
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"input directory not found: {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if not files:
        raise SystemExit(f"no .npy files found in {args.input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model = load_model(device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"device : {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")
    print(f"model  : {n_params/1e6:.2f}M parameters")
    print(f"input  : {len(files)} .npy files from {args.input_dir}")
    print(f"output : {args.output_dir}\n")

    t0 = time.perf_counter()
    failures = 0
    for i, path in enumerate(files, 1):
        name = os.path.basename(path)
        try:
            arr = to_2d(np.load(path), name)
            out = restore(model, arr, device, use_amp)

            # Contract checks, per file, before anything is written.
            expected = (arr.shape[0] * SCALE, arr.shape[1] * SCALE)
            if out.shape != expected:
                raise ValueError(f"expected {expected}, produced {out.shape}")
            if not np.isfinite(out).all():
                raise ValueError("output contains NaN or Inf")
            if out.min() < 0.0 or out.max() > 1.0:
                raise ValueError(f"output outside [0,1]: [{out.min()}, {out.max()}]")

            np.save(os.path.join(args.output_dir, name), out)
        except Exception as e:  # keep going; report at the end
            failures += 1
            print(f"  FAILED {name}: {e}", file=sys.stderr)
            continue

        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)} restored", flush=True)

    dt = time.perf_counter() - t0
    done = len(files) - failures
    print(f"\nrestored {done}/{len(files)} files in {dt:.1f}s "
          f"({dt/max(1, len(files))*1000:.1f} ms/image including I/O)")

    if failures:
        print(f"{failures} file(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
