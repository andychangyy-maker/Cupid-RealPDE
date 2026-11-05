"""Render a captured Cupid forecast, its errors and intervals. No inference here."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nice_upper(value):
    value = max(value, 1e-8)
    unit = 10 ** np.floor(np.log10(value))
    return float(np.ceil(value / unit * 2) * unit / 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--font', type=Path, required=True, help='Local TrueType font; not redistributed')
    parser.add_argument('--output', type=Path, default=Path('docs/assets'))
    args = parser.parse_args()
    record = json.loads((args.capture / 'capture.json').read_text())
    assert record['complete'] and all(record['compile_status'].values())
    path = args.capture / 'forecast.npz'
    assert sha(path) == record['array_sha256']
    with np.load(path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    truth, pred, lower, upper = [arrays[key][..., :2] for key in ('truth', 'prediction', 'lower', 'upper')]
    assert all(a.shape == (20, 32, 64, 2) and np.isfinite(a).all() for a in (truth, pred, lower, upper))
    assert (upper >= lower).all()
    error, width = np.abs(pred - truth), upper - lower
    fields = [truth, pred, error, width]
    limits = []
    for channel in range(2):
        joint = np.stack([truth[..., channel], pred[..., channel]])
        if channel == 1:
            bound = nice_upper(float(np.abs(joint).max()))
            velocity = (-bound, bound)
        else:
            lo, hi = float(joint.min()), float(joint.max())
            velocity = (float(np.floor(lo * 100) / 100), float(np.ceil(hi * 100) / 100))
        limits.append([velocity, velocity, (0., nice_upper(float(error[..., channel].max()))),
                       (0., nice_upper(float(width[..., channel].max())))])
    maps = [['viridis', 'viridis', 'magma', 'cividis'], ['RdBu_r', 'RdBu_r', 'magma', 'cividis']]
    title = ['Ground truth', 'Cupid forecast', 'Absolute error', 'Interval width']
    font = lambda size: ImageFont.truetype(str(args.font), size)
    bg, fg, muted = '#0b1c2a', '#eef5ec', '#a9c0c1'

    def colored(a, channel, column):
        lo, hi = limits[channel][column]
        assert a.min() >= lo - 1e-7 and a.max() <= hi + 1e-7
        return Image.fromarray(colormaps[maps[channel][column]]((a - lo) / (hi - lo), bytes=True)[..., :3])

    # One palette across all horizons prevents temporal color quantization flicker.
    colors = np.concatenate([colormaps[name](np.linspace(0, 1, 56), bytes=True)[:, :3]
                             for name in ('viridis', 'RdBu_r', 'magma', 'cividis')])
    colors = np.vstack([colors, [[11, 28, 42], [238, 245, 236], [169, 192, 193], [83, 227, 209]]])
    palette = Image.new('P', (1, 1))
    palette.putpalette(colors.ravel().tolist() + [0] * (768 - colors.size))
    args.output.mkdir(parents=True, exist_ok=True)

    for mobile in (False, True):
        frames = []
        canvas_width, canvas_height = (480, 1110) if mobile else (1280, 780)
        tile_w = 208 if mobile else 282
        tile_h = tile_w // 2
        for horizon in range(20):
            canvas = Image.new('RGB', (canvas_width, canvas_height), bg)
            draw = ImageDraw.Draw(canvas)
            draw.text((22, 20), 'TWENTY FUTURE FRAMES' if mobile else 'ONE FORECAST. TWENTY FUTURE FRAMES.',
                      font=font(26 if mobile else 34), fill=fg)
            draw.text((22, 61 if mobile else 73), f'Lead {horizon + 1:02d} / 20   |   target frame {record["target_frame_indices"][horizon]}',
                      font=font(19 if mobile else 22), fill='#53e3d1')
            if mobile:
                draw.text((22, 92), f'{record["trajectory"]}  /  32 x 64 grid', font=font(17), fill=muted)
            else:
                draw.text((660, 75), f'{record["trajectory"]}  /  32 x 64  /  fixed color scales', font=font(19), fill=muted)
            for channel in range(2):
                row_top = (174 + channel * 458) if mobile else (198 + channel * 300)
                draw.text((22, row_top - 46 if mobile else row_top - 70),
                          ['u  /  streamwise velocity', 'v  /  transverse velocity'][channel],
                          font=font(19 if mobile else 22), fill=fg)
                for column, field in enumerate(fields):
                    x = 22 + (column % 2) * 230 if mobile else 22 + column * 316
                    y = row_top + (column // 2) * 218 if mobile else row_top
                    draw.text((x, y - 21 if mobile else y - 32), title[column],
                              font=font(18 if mobile else 21), fill=muted)
                    tile = colored(field[horizon, ..., channel], channel, column)
                    canvas.paste(tile.resize((tile_w, tile_h), Image.Resampling.BILINEAR), (x, y + (6 if mobile else 0)))
                    bar_y = y + tile_h + (20 if mobile else 16)
                    lo, hi = limits[channel][column]
                    bar = colored(np.linspace(lo, hi, tile_w)[None, :], channel, column)
                    canvas.paste(bar.resize((tile_w, 7)), (x, bar_y))
                    draw.text((x, bar_y + 12), f'{lo:.3f}', font=font(16 if mobile else 18), fill=muted)
                    draw.text((x + tile_w, bar_y + 12), f'{hi:.3f}', anchor='ra',
                              font=font(16 if mobile else 18), fill=muted)
            footer_y = 1048 if mobile else 718
            draw.text((22, footer_y), 'Public-training example / not a held-out score',
                      font=font(17 if mobile else 20), fill=muted)
            draw.text((22, footer_y + 28), 'Interval width = upper - lower / dataset units',
                      font=font(17 if mobile else 20), fill=muted)
            frames.append(canvas.quantize(palette=palette, dither=Image.Dither.NONE))
        stem = 'forecast-comparison-mobile' if mobile else 'forecast-comparison'
        frames[-1].save(args.output / f'{stem}.png', optimize=True)
        frames[0].save(args.output / f'{stem}.gif', save_all=True, append_images=frames[1:],
                       duration=200, loop=0, optimize=True, disposal=1)

    # The spatial point was fixed before inference; this is not a best-case trace.
    row, column = 16, 48
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), layout='constrained')
    horizon = np.arange(1, 21)
    for channel, ax in enumerate(axes):
        ax.fill_between(horizon, lower[:, row, column, channel], upper[:, row, column, channel],
                        color='#16a6a0', alpha=.2, label='Model lower–upper interval')
        ax.plot(horizon, pred[:, row, column, channel], color='#008b83', lw=2, label='Cupid forecast')
        ax.plot(horizon, truth[:, row, column, channel], color='#dd674b', lw=2, marker='.', label='Ground truth')
        ax.set(xlim=(1, 20), ylabel=f'{"uv"[channel]} (dataset units)', xticks=[1, 5, 10, 15, 20])
        ax.grid(alpha=.16)
    axes[0].set_title('One fixed grid point, all 20 horizons  |  row 16, column 48 (zero-based)', loc='left', fontsize=12)
    axes[0].legend(loc='best', frameon=False, fontsize=9)
    axes[1].set_xlabel('Forecast lead (frames)')
    fig.savefig(args.output / 'forecast-interval-trace.png', dpi=160, facecolor='white')
    plt.close(fig)

    metrics = {name: {'mae': float(error[..., c].mean()), 'rmse': float(np.sqrt(np.mean((pred[..., c] - truth[..., c]) ** 2))),
                     'mean_interval_width': float(width[..., c].mean()),
                     'empirical_pointwise_coverage': float(((truth[..., c] >= lower[..., c]) & (truth[..., c] <= upper[..., c])).mean())}
               for c, name in enumerate(('u', 'v'))}
    provenance = {
        'kind': 'Qualitative inference on public training data; not a held-out or official score',
        'rendered_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': 'https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data',
        'credit': 'AI4Science-WestlakeU / RealPDE and RealPDEBench contributors; Cupid model by Team Cupid',
        'data_and_derived_visuals_license': 'CC-BY-NC-4.0',
        'selection': 'Same trajectory as homepage observation preview; first clean81 tail origin, selected before inference',
        'capture': record, 'color_limits_per_channel_per_column': limits, 'colormaps': maps,
        'display': 'Bilinear display enlargement only; fixed scales across all 20 horizons; no clipping or field filtering',
        'orientation': 'Stored array orientation: x increases rightwards; y decreases downwards',
        'playback_ms_per_frame': 200, 'playback_scope': 'Illustrative, not real-time',
        'static_horizon': 20, 'trace_pixel_zero_based': [row, column], 'descriptive_metrics': metrics,
        'metrics_scope': 'Single window, 20 x 32 x 64 points per channel; stored units, no official scoring',
        'renderer_sha256': sha(Path(__file__)),
        'render_environment': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
                               'pillow': PIL.__version__, 'font_file': args.font.name,
                               'font_sha256': sha(args.font)},
        'assets': {p.name: sha(p) for p in sorted(args.output.glob('forecast-*.png'))}
    }
    execution = args.capture / 'execution.json'
    if execution.is_file():
        provenance['execution'] = json.loads(execution.read_text())
    provenance['assets'].update({p.name: sha(p) for p in sorted(args.output.glob('forecast-*.gif'))})
    (args.output / 'forecast-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
