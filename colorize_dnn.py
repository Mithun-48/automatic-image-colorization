from pathlib import Path
import argparse

import cv2
import numpy as np


MODEL_DIR = Path("models/opencv_colorization")

PROTOTXT = MODEL_DIR / "colorization_deploy_v2.prototxt"
WEIGHTS = MODEL_DIR / "colorization_release_v2.caffemodel"
POINTS = MODEL_DIR / "pts_in_hull.npy"


def colorize(input_path, output_path):
    for path in (PROTOTXT, WEIGHTS, POINTS):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Missing or empty model file: {path}"
            )

    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read input: {input_path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    height, width = image.shape[:2]

    net = cv2.dnn.readNetFromCaffe(
        str(PROTOTXT), str(WEIGHTS)
    )

    # Load the model's color-bin centers.
    points = np.load(str(POINTS)).transpose().reshape(
        2, 313, 1, 1
    ).astype(np.float32)

    class_layer = net.getLayerId("class8_ab")
    prior_layer = net.getLayerId("conv8_313_rh")

    net.getLayer(class_layer).blobs = [points]
    net.getLayer(prior_layer).blobs = [
        np.full((1, 313), 2.606, dtype=np.float32)
    ]

    # Convert the input to Lab and predict its color channels.
    image_float = image.astype(np.float32) / 255.0
    lab = cv2.cvtColor(image_float, cv2.COLOR_BGR2LAB)
    small_lab = cv2.resize(lab, (224, 224))

    luminance = small_lab[:, :, 0]
    net.setInput(cv2.dnn.blobFromImage(luminance - 50.0))

    predicted_ab = net.forward()[0].transpose(1, 2, 0)
    predicted_ab = cv2.resize(predicted_ab, (width, height))

    # Preserve input luminance and insert the predicted colors.
    
    # Mildly sharpen luminance while preserving predicted color channels.
    luminance_full = lab[:, :, 0]

    luminance_blur = cv2.GaussianBlur(
        luminance_full,
        (0, 0),
        sigmaX=1.0,
        sigmaY=1.0
    )

    luminance_sharp = cv2.addWeighted(
        luminance_full, 1.25,
        luminance_blur, -0.25,
        0
    )

    luminance_sharp = np.clip(luminance_sharp, 0, 100)

    output_lab = np.concatenate(
        (luminance_sharp[:, :, np.newaxis], predicted_ab),
        axis=2
    )

    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    output_bgr = np.clip(output_bgr, 0.0, 1.0)
    output_bgr = (output_bgr * 255).astype(np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), output_bgr):
        raise RuntimeError(f"Could not save output: {output_path}")

    print(f"Input:  {input_path}")
    print(f"Output: {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Colorize a grayscale image with a pretrained DNN."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/dnn_colorized.jpg")
    args = parser.parse_args()

    colorize(args.input, args.output)
