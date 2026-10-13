"""Train experimental U/V SVRs with richer features and target scaling.

This is an experimental alternative to the report-aligned baseline.
It uses per-image train/validation splitting to avoid segment-level leakage.
"""
from pathlib import Path
import argparse
import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

from utils import load_rgb, resize_width, make_yuv, segment_y, extract_features, segment_targets

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_DIR / "data" / "train"
MODEL_DIR = PROJECT_DIR / "models"
MIN_CHROMA = 0.02  # reject grayscale or nearly colorless training images
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def process_image(path):
    try:
        rgb = resize_width(load_rgb(path), 500)
        yuv = make_yuv(rgb)
        if np.sqrt(yuv[:, :, 1] ** 2 + yuv[:, :, 2] ** 2).mean() < MIN_CHROMA:
            return None
        y = yuv[:, :, 0]
        labels = segment_y(y)
        x, _ = extract_features(y, labels)
        u, v = segment_targets(yuv, labels)
        return x, u, v
    except Exception as exc:
        print(f"Skipping {Path(path).name}: {exc}")
        return None


def make_model(C, epsilon, gamma):
    svr = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma))
    return TransformedTargetRegressor(regressor=svr, transformer=StandardScaler())


def stack(items, max_samples, rng):
    x = np.vstack([item[0] for item in items])
    u = np.concatenate([item[1] for item in items])
    v = np.concatenate([item[2] for item in items])
    if len(u) > max_samples:
        idx = rng.choice(len(u), max_samples, replace=False)
        x, u, v = x[idx], u[idx], v[idx]
    return x, u, v


def report(name, model, x, target, baseline):
    prediction = model.predict(x)
    print(f"  {name}: MAE={mean_absolute_error(target, prediction):.4f} "
          f"(predict-the-mean MAE={baseline:.4f})  R2={r2_score(target, prediction):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train experimental enhanced SVR colorization models")
    parser.add_argument("--C", type=float, default=3.0)
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon in standardized target units")
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--max-samples", type=int, default=12000,
                        help="maximum superpixel training samples (keep modest for RBF-SVR runtime)")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--tune", action="store_true", help="run optional grid search; can be slow")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    gamma = args.gamma if args.gamma == "scale" else float(args.gamma)

    paths = sorted(p for p in TRAIN_DIR.iterdir() if p.suffix.lower() in EXTS)
    if not paths:
        raise SystemExit(f"No training images in {TRAIN_DIR}. Run prepare_dataset.py first.")
    print(f"Found {len(paths)} training images; extracting features in parallel...")
    results = Parallel(n_jobs=args.jobs, verbose=5)(delayed(process_image)(p) for p in paths)
    results = [item for item in results if item is not None]
    print(f"Usable color images: {len(results)} (rejected {len(paths) - len(results)})")
    if len(results) < 20:
        raise SystemExit("Too few usable images. Use at least 20; a few hundred is preferable.")

    rng = np.random.default_rng(42)
    order = rng.permutation(len(results))
    n_val = max(1, int(len(results) * args.val_fraction))
    val_items = [results[i] for i in order[:n_val]]
    train_items = [results[i] for i in order[n_val:]]
    x_train, u_train, v_train = stack(train_items, args.max_samples, rng)
    x_val, u_val, v_val = stack(val_items, min(10000, args.max_samples), rng)
    print(f"Train superpixels: {len(u_train)} | validation superpixels from unseen images: {len(u_val)}")
    print(f"Feature dimensions: {x_train.shape[1]} | U std={u_train.std():.4f} | V std={v_train.std():.4f}")

    C = args.C
    if args.tune:
        print("Tuning U on a subset (this may take a while)...")
        sub = rng.choice(len(u_train), min(3000, len(u_train)), replace=False)
        grid = GridSearchCV(
            make_model(C, args.epsilon, gamma),
            {"regressor__svr__C": [1, 3, 10],
             "regressor__svr__gamma": [0.2 / x_train.shape[1], 1 / x_train.shape[1], 4 / x_train.shape[1]]},
            cv=3, scoring="neg_mean_absolute_error", n_jobs=args.jobs)
        grid.fit(x_train[sub], u_train[sub])
        C = grid.best_params_["regressor__svr__C"]
        gamma = grid.best_params_["regressor__svr__gamma"]
        print("Best settings:", grid.best_params_)

    print("\nValidation metrics (held-out images):")
    for name, train_target, val_target in (("U", u_train, u_val), ("V", v_train, v_val)):
        model = make_model(C, args.epsilon, gamma).fit(x_train, train_target)
        baseline = np.abs(val_target - train_target.mean()).mean()
        report(name, model, x_val, val_target, baseline)

    print("\nRefitting final models on all selected images...")
    x_all, u_all, v_all = stack(results, args.max_samples, rng)
    MODEL_DIR.mkdir(exist_ok=True)
    for name, target in (("u", u_all), ("v", v_all)):
        model = make_model(C, args.epsilon, gamma).fit(x_all, target)
        joblib.dump(model, MODEL_DIR / f"{name}_svr.joblib")
        print(f"Saved {MODEL_DIR / f'{name}_svr.joblib'}")
    print("TRAINING COMPLETE")


if __name__ == "__main__":
    main()
