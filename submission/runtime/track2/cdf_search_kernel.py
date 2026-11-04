"""Exact 64-bin interval-utility search without a large quadratic work tensor."""
import torch
import triton
import triton.language as tl


@triton.jit
def _search(Cdf, Sharp, Parent, Lower, Upper):
    group = tl.program_id(0)
    parent = tl.load(Parent + group)
    left = tl.arange(0, 64)
    right = tl.arange(0, 64) + 1
    low_cdf = tl.load(Cdf + group * 65 + left)
    high_cdf = tl.load(Cdf + group * 65 + right)
    distance = right[None, :] - left[:, None]
    sharp = tl.load(Sharp + parent * 65 + tl.maximum(distance, 0))
    score = (high_cdf[None, :] - low_cdf[:, None]) * sharp
    score = tl.where(distance > 0, score, -1.)
    best = tl.argmax(tl.reshape(score, (4096,)), 0, tie_break_left=True)
    tl.store(Lower + group, -4. + .125 * (best // 64))
    tl.store(Upper + group, -4. + .125 * (best % 64 + 1))


def choose(cdf, sharp, parent):
    lower = torch.empty(cdf.shape[0], device=cdf.device, dtype=torch.float32)
    upper = torch.empty_like(lower)
    _search[(cdf.shape[0],)](cdf, sharp, parent, lower, upper, num_warps=4, enable_fp_fusion=False)
    return lower, upper
