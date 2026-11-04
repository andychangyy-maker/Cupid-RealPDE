"""Render observed public RealPDE fields, not predictions. Requires h5py, numpy, Pillow."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('trajectory', type=Path, help='Official flat 10125_10.h5')
parser.add_argument('--font', required=True, type=Path, help='Local TrueType font')
parser.add_argument('--output', type=Path, default=Path('docs/assets'))
args = parser.parse_args()
indices = list(range(0, 180, 4))
with h5py.File(args.trajectory, 'r') as f:
    fields = [f[k][indices] for k in ('u', 'v')]
    times = f['t'][indices].tolist()
    # Stored y coordinates decrease down the array; retain that orientation.
    assert np.all(np.diff(f['x'][0]) > 0) and np.all(np.diff(f['y'][:, 0]) < 0)
assert all(a.shape == (45, 64, 128) and np.isfinite(a).all() for a in fields)
limits = [(-0.03, 0.19), (-0.065, 0.065)]  # Fixed across every frame; no clipping.
assert all(a.min() >= lo and a.max() <= hi for a, (lo, hi) in zip(fields, limits))
colors = [
    ['#102336', '#18526b', '#23a7ad', '#a8ddbc', '#f2e5b4', '#ff9b78'],
    ['#53e3d1', '#246e7d', '#102336', '#8b4f4b', '#ffa788'],
]


def colorize(a, stops):
    rgb = np.array([tuple(bytes.fromhex(c[1:])) for c in stops])
    x = np.linspace(0, 1, len(rgb))
    return np.stack([np.interp(a, x, rgb[:, c]) for c in range(3)], axis=-1).astype('uint8')


font = lambda size: ImageFont.truetype(str(args.font), size)
# One shared palette avoids changing colors between frames and keeps the GIF small.
palette_rgb = np.concatenate([colorize(np.linspace(0, 1, 112), c) for c in colors])
palette_rgb = np.vstack([palette_rgb, [[11, 28, 42], [238, 245, 236], [154, 200, 200], [196, 220, 218], [169, 192, 193]]])
palette = Image.new('P', (1, 1))
palette.putpalette(palette_rgb.ravel().tolist() + [0] * (768 - palette_rgb.size))
frames = []
for n, index in enumerate(indices):
    canvas = Image.new('RGB', (1100, 438), '#0b1c2a')
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 20), 'A REAL FLOW, IN MOTION', font=font(25), fill='#eef5ec')
    draw.text((780, 25), f'Observed frame {index:03d} / t = {times[n]:.2f}', font=font(17), fill='#9ac8c8')
    for j, (a, (lo, hi), stops) in enumerate(zip(fields, limits, colors)):
        left = 28 + j * 548
        draw.text((left, 64), ['Streamwise velocity  u', 'Transverse velocity  v'][j], font=font(20), fill='#c4dcda')
        tile = Image.fromarray(colorize((a[n] - lo) / (hi - lo), stops))
        canvas.paste(tile.resize((496, 248), Image.Resampling.BILINEAR), (left, 99))
        bar = Image.fromarray(colorize(np.linspace(0, 1, 496)[None, :], stops))
        canvas.paste(bar.resize((496, 6)), (left, 359))
        draw.text((left, 371), f'{lo:.3f}', font=font(15), fill='#a9c0c1')
        draw.text((left + 450, 371), f'{hi:.3f}', font=font(15), fill='#a9c0c1')
    draw.text((28, 410), 'RealPDE public data / 10125_10.h5 / fixed scales, dataset units / visualization only, not model predictions', font=font(15), fill='#a9c0c1')
    frames.append(canvas.quantize(palette=palette, dither=Image.Dither.NONE))
args.output.mkdir(parents=True, exist_ok=True)
frames[0].save(args.output / 'real-flow-preview.png', optimize=True)
frames[0].save(args.output / 'real-flow-preview.gif', save_all=True,
               append_images=frames[1:], duration=110, loop=0, optimize=True, disposal=1)
provenance = {
    'kind': 'observed public data visualization; no model inference',
    'source': 'https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data',
    'upstream_credit': 'AI4Science-WestlakeU / RealPDE and RealPDEBench contributors',
    'data_license': 'CC-BY-NC-4.0', 'license_url': 'https://creativecommons.org/licenses/by-nc/4.0/',
    'trajectory': args.trajectory.name,
    'trajectory_sha256': hashlib.file_digest(args.trajectory.open('rb'), 'sha256').hexdigest(),
    'frame_indices': indices, 'stored_time_values': times,
    'fields': ['u', 'v'], 'native_grid': [64, 128], 'display_grid_per_panel': [248, 496],
    'color_limits': limits, 'color_stops': colors,
    'transform': 'Map stored values to fixed color scales; bilinear image enlargement; no field filtering.',
    'orientation': 'x increases rightwards; y decreases downwards, matching stored arrays.',
    'playback': '110 ms per rendered frame; illustrative playback, not real-time.',
}
(args.output / 'flow-preview-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'frames': len(frames), 'gif_bytes': (args.output / 'real-flow-preview.gif').stat().st_size}))
