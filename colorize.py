"""Experimental colorizer using richer FFT/context features and edge-aware refinement.

Examples:
  python colorize.py --input data/test/image_0001.jpg --output outputs/enhanced.jpg
  python colorize.py --input data/test_original/image_0001.jpg --compare
  python colorize.py --input data/test --compare
"""
from pathlib import Path
import argparse
import cv2
import joblib
import numpy as np

from utils import (load_rgb, save_rgb, resize_width, make_yuv, segment_y,
                   extract_features, neighbor_graph, icm_smooth, guided_filter,
                   yuv_to_rgb, LUMA_COL)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def colorize_array(rgb, u_model, v_model, work_width=500, gamma=0.25,
                   gain=1.5, radius=10, eps=1e-3):
    """Colorize RGB input. `gain` boosts predicted chroma; compare carefully."""
    small = resize_width(rgb, work_width)
    y_small = make_yuv(small)[:, :, 0]
    labels = segment_y(y_small)
    features, _ = extract_features(y_small, labels)
    u_pred = u_model.predict(features)
    v_pred = v_model.predict(features)

    # Weighted adjacency reduces smoothing across strong luminance edges.
    graph = neighbor_graph(labels, features[:, LUMA_COL])
    u_seg = icm_smooth(u_pred, graph, gamma=gamma)
    v_seg = icm_smooth(v_pred, graph, gamma=gamma)

    # SVR predictions can average plausible colors; gain is an experimental control.
    u_seg = u_seg * gain
    v_seg = v_seg * gain

    # Edge-aware spread of segment colors to pixels.
    u_map = guided_filter(y_small, u_seg[labels], radius=radius, eps=eps)
    v_map = guided_filter(y_small, v_seg[labels], radius=radius, eps=eps)

    # Keep the original-resolution luminance; resize only the predicted chroma fields.
    y_full = make_yuv(rgb)[:, :, 0]
    height, width = y_full.shape
    if u_map.shape != (height, width):
        u_map = cv2.resize(u_map, (width, height), interpolation=cv2.INTER_CUBIC)
        v_map = cv2.resize(v_map, (width, height), interpolation=cv2.INTER_CUBIC)
    u_map = np.clip(u_map, -0.436, 0.436)
    v_map = np.clip(v_map, -0.615, 0.615)
    return yuv_to_rgb(y_full, u_map, v_map)


def is_color(rgb):
    return float(np.abs(rgb[..., 0].astype(np.int16) - rgb[..., 1].astype(np.int16)).mean()) > 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="image file or folder")
    parser.add_argument("--output", default=None, help="output file for one input image")
    parser.add_argument("--gain", type=float, default=1.5, help="chroma gain; 1.0 means no boost")
    parser.add_argument("--gamma", type=float, default=0.25, help="ICM smoothness strength for this variant")
    parser.add_argument("--radius", type=int, default=10, help="guided-filter radius")
    parser.add_argument("--compare", action="store_true", help="save grayscale | output | original (when input is color)")
    args = parser.parse_args()

    u_path, v_path = MODEL_DIR / "u_svr.joblib", MODEL_DIR / "v_svr.joblib"
    if not u_path.is_file() or not v_path.is_file():
        raise FileNotFoundError("Enhanced model files are missing. Run python train.py first.")
    u_model, v_model = joblib.load(u_path), joblib.load(v_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    src = Path(args.input)
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    else:
        files = [src]
    if not files:
        raise FileNotFoundError(f"No supported images found in {src}")

    for path in files:
        rgb = load_rgb(path)
        output = colorize_array(rgb, u_model, v_model, gamma=args.gamma, gain=args.gain, radius=args.radius)
        if args.output and not src.is_dir():
            out_path = Path(args.output)
            if not out_path.is_absolute():
                out_path = PROJECT_DIR / out_path
        else:
            out_path = OUTPUT_DIR / f"{path.stem}_enhanced.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_rgb(out_path, output)

        if args.compare:
            y = make_yuv(rgb)[:, :, 0]
            panels = [np.repeat(y[:, :, None], 3, axis=2), output]
            if is_color(rgb):
                panels.append(rgb.astype(np.float32) / 255.0)
            compare_path = OUTPUT_DIR / f"{path.stem}_compare.jpg"
            save_rgb(compare_path, np.hstack(panels))
            print(f"Saved comparison: {compare_path}")
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
