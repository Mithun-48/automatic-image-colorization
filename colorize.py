from pathlib import Path
import argparse
import joblib
import cv2
import numpy as np

from utils import (
    load_rgb,
    resize_width,
    make_yuv,
    segment_y,
    extract_features,
    neighbor_graph,
    icm_smooth,
    yuv_to_rgb,
    save_rgb,
)


MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

ICM_GAMMA = 2.0
ICM_SIGMA = 1.0
ICM_ITERATIONS = 10

# Controls how smoothly predicted colors spread between superpixels.
CHROMA_SMOOTH_SIGMA = 12.0
NEIGHBOR_THRESHOLD=5.0

def smooth_segment_chroma(
    values,
    labels,
    centers,
    sigma=CHROMA_SMOOTH_SIGMA
):
    """
    Convert one chroma value per superpixel into a smooth
    pixel-level chroma field.

    Instead of assigning one constant color to every SLIC
    region, each superpixel prediction is placed at its
    centroid and smoothly interpolated across the image.
    """

    height, width = labels.shape

    value_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    weight_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # Put each predicted chroma value at its superpixel centroid.
    for index, (cy, cx) in enumerate(centers):

        y = int(round(cy))
        x = int(round(cx))

        y = max(0, min(height - 1, y))
        x = max(0, min(width - 1, x))

        value_map[y, x] = values[index]
        weight_map[y, x] = 1.0

    # Spread centroid information smoothly.
    value_blurred = cv2.GaussianBlur(
        value_map,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    weight_blurred = cv2.GaussianBlur(
        weight_map,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    # Normalized interpolation.
    smooth = value_blurred / (
        weight_blurred + 1e-6
    )

    return smooth.astype(np.float32)


def colorize(input_path, output_path):

    u_model_path = MODEL_DIR / "u_svr.joblib"
    v_model_path = MODEL_DIR / "v_svr.joblib"

    if not u_model_path.exists():
        raise FileNotFoundError(
            f"Missing model: {u_model_path}"
        )

    if not v_model_path.exists():
        raise FileNotFoundError(
            f"Missing model: {v_model_path}"
        )

    print("Loading trained SVR models...")

    u_model = joblib.load(u_model_path)
    v_model = joblib.load(v_model_path)

    print("Loading input image...")

    rgb = load_rgb(input_path)

    rgb = resize_width(
        rgb,
        500
    )

    yuv = make_yuv(rgb)

    y = yuv[:, :, 0]

    print("Creating SLIC superpixels...")

    labels = segment_y(y)

    print("Extracting FFT features...")

    features, centers = extract_features(
        y,
        labels
    )

    print(
        f"Number of superpixels: {len(centers)}"
    )

    print("Predicting U chrominance...")

    u_pred = u_model.predict(
        features
    )

    print("Predicting V chrominance...")

    v_pred = v_model.predict(
        features
    )

    print("Building superpixel neighbor graph...")

    graph = neighbor_graph(
        labels,
        features,
        threshold=NEIGHBOR_THRESHOLD
    )

    print("Applying MRF / ICM smoothing...")

    u_smooth = icm_smooth(
        u_pred.copy(),
        u_pred,
        graph,
        gamma=ICM_GAMMA,
        sigma=ICM_SIGMA,
        iterations=ICM_ITERATIONS
    )

    v_smooth = icm_smooth(
        v_pred.copy(),
        v_pred,
        graph,
        gamma=ICM_GAMMA,
        sigma=ICM_SIGMA,
        iterations=ICM_ITERATIONS
    )

    print("Creating smooth chrominance fields...")

    u_img = smooth_segment_chroma(
        u_smooth,
        labels,
        centers
    )

    v_img = smooth_segment_chroma(
        v_smooth,
        labels,
        centers
    )

    # Keep chrominance within a safe range.
    u_img = np.clip(
        u_img,
        -0.5,
        0.5
    )

    v_img = np.clip(
        v_img,
        -0.5,
        0.5
    )

    print("Converting YUV to RGB...")

    result = yuv_to_rgb(
        y,
        u_img,
        v_img
    )

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    save_rgb(
        output_path,
        result
    )

    print()
    print(
        f"Saved colorized image to: {output_path}"
    )


def main():

    parser = argparse.ArgumentParser(
        description="Automatic image colorization"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to grayscale input image"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output colorized image path"
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    if args.output:

        output_path = Path(
            args.output
        )

    else:

        output_path = (
            OUTPUT_DIR /
            f"{input_path.stem}_colorized.jpg"
        )

    colorize(
        input_path,
        output_path
    )


if __name__ == "__main__":
    main()