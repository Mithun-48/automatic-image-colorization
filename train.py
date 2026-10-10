from pathlib import Path
import argparse
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    load_rgb,
    resize_width,
    make_yuv,
    segment_y,
    extract_features,
    segment_targets,
)


TRAIN_DIR = Path("data/train")
MODEL_DIR = Path("models")

SVR_EPSILON = 0.0625
SVR_C = 0.125
SVR_GAMMA = 2.0

# Maximum number of superpixel samples used for RBF-SVR training.
# The complete 500-image dataset is still used for feature extraction.
MAX_SVR_SAMPLES = 25000


def collect_training_data():

    image_paths = sorted([
        p for p in TRAIN_DIR.iterdir()
        if p.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        }
    ])

    if not image_paths:
        raise RuntimeError(
            "No training images found. Put COLOR images in data/train/"
        )

    all_x = []
    all_u = []
    all_v = []

    print(f"Found {len(image_paths)} training images.")

    for path in tqdm(
        image_paths,
        desc="Extracting training features"
    ):

        try:

            rgb = resize_width(
                load_rgb(path),
                500
            )

            yuv = make_yuv(rgb)

            y = yuv[:, :, 0]

            labels = segment_y(y)

            x, _ = extract_features(
                y,
                labels
            )

            u, v = segment_targets(
                yuv,
                labels
            )

            all_x.append(x)
            all_u.append(u)
            all_v.append(v)

        except Exception as exc:

            print(
                f"Skipping {path.name}: {exc}"
            )

    if not all_x:
        raise RuntimeError(
            "Could not extract features from any training image."
        )

    return (
        np.vstack(all_x),
        np.concatenate(all_u),
        np.concatenate(all_v),
    )


def select_training_samples(x, target):

    if len(target) <= MAX_SVR_SAMPLES:

        return x, target

    print()
    print(
        f"Full dataset contains {len(target)} superpixel samples."
    )

    print(
        f"Selecting {MAX_SVR_SAMPLES} representative samples "
        f"for faster RBF-SVR training..."
    )

    # Fixed seed makes the selection reproducible.
    rng = np.random.default_rng(42)

    indices = rng.choice(
        len(target),
        size=MAX_SVR_SAMPLES,
        replace=False
    )

    x_train = x[indices]
    target_train = target[indices]

    return x_train, target_train


def train_model(x, target, name):

    x_train, target_train = select_training_samples(
        x,
        target
    )

    model = make_pipeline(
    StandardScaler(),
    SVR(
        kernel="rbf",
        C=SVR_C,
        epsilon=SVR_EPSILON,
        gamma="scale",
            ),
            )

    print()
    print(
        f"Training {name} SVR on "
        f"{len(target_train)} superpixels..."
    )

    model.fit(
        x_train,
        target_train
    )

    MODEL_DIR.mkdir(
        exist_ok=True
    )

    model_path = (
        MODEL_DIR /
        f"{name}_svr.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Saved {name} model to: "
        f"{model_path}"
    )

    return model


def main():

    global SVR_EPSILON
    global SVR_C
    global SVR_GAMMA

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epsilon",
        type=float,
        default=SVR_EPSILON
    )

    parser.add_argument(
        "--C",
        type=float,
        default=SVR_C
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=SVR_GAMMA
    )

    args = parser.parse_args()

    SVR_EPSILON = args.epsilon
    SVR_C = args.C
    SVR_GAMMA = args.gamma

    print()
    print("=" * 60)
    print("AUTOMATIC IMAGE COLORIZATION")
    print("SVR TRAINING")
    print("=" * 60)

    print()
    print("SVR configuration:")
    print("  Kernel  : RBF")
    print(f"  C       : {SVR_C}")
    print(f"  Epsilon : {SVR_EPSILON}")
    print(f"  Gamma   : {SVR_GAMMA}")
    print(f"  Max samples: {MAX_SVR_SAMPLES}")
    print()

    x, u, v = collect_training_data()

    print()
    print("Feature matrix:", x.shape)
    print("U targets:", u.shape)
    print("V targets:", v.shape)

    print()
    print("U target statistics:")
    print("  Min :", float(u.min()))
    print("  Max :", float(u.max()))
    print("  Mean:", float(u.mean()))
    print("  Std :", float(u.std()))

    print()
    print("V target statistics:")
    print("  Min :", float(v.min()))
    print("  Max :", float(v.max()))
    print("  Mean:", float(v.mean()))
    print("  Std :", float(v.std()))

    print()

    train_model(
        x,
        u,
        "u"
    )

    train_model(
        x,
        v,
        "v"
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print()
    print("Saved:")
    print("  models/u_svr.joblib")
    print("  models/v_svr.joblib")


if __name__ == "__main__":
    main()