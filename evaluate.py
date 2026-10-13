"""Evaluate the experimental colorizer on paired grayscale/color test images."""
from pathlib import Path
import csv
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import joblib

from utils import load_rgb, save_rgb
from colorize import colorize_array

PROJECT_DIR = Path(__file__).resolve().parent
TEST_DIR = PROJECT_DIR / "data" / "test"
ORIGINAL_DIR = PROJECT_DIR / "data" / "test_original"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "evaluation"
COMPARE_DIR = PROJECT_DIR / "outputs" / "comparisons"
RESULTS_DIR = PROJECT_DIR / "results"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    u_path, v_path = MODEL_DIR / "u_svr.joblib", MODEL_DIR / "v_svr.joblib"
    if not u_path.is_file() or not v_path.is_file():
        raise FileNotFoundError("Enhanced SVR models are missing. Run train.py first.")
    if not TEST_DIR.is_dir() or not ORIGINAL_DIR.is_dir():
        raise FileNotFoundError("Expected data/test and data/test_original folders.")

    u_model, v_model = joblib.load(u_path), joblib.load(v_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_paths = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() in EXTS)
    rows = []

    for test_path in test_paths:
        original_path = ORIGINAL_DIR / test_path.name
        if not original_path.is_file():
            print(f"Skipping {test_path.name}: no paired original")
            continue

        gray_rgb = load_rgb(test_path)
        original = load_rgb(original_path).astype(np.float32) / 255.0
        pred = colorize_array(gray_rgb, u_model, v_model)
        if original.shape[:2] != pred.shape[:2]:
            original = cv2.resize(original, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)

        mae = float(np.mean(np.abs(original - pred)))
        mse = float(np.mean((original - pred) ** 2))
        psnr = float("inf") if mse == 0 else float(10 * np.log10(1 / mse))
        ssim_value = float(ssim(original, pred, channel_axis=2, data_range=1.0))
        rows.append((test_path.name, mae, mse, psnr, ssim_value))

        out_path = OUTPUT_DIR / test_path.name
        save_rgb(out_path, pred)
        comparison = np.hstack([gray_rgb.astype(np.float32) / 255.0, pred, original])
        save_rgb(COMPARE_DIR / test_path.name, comparison)
        print(f"{test_path.name}: MAE={mae:.4f} MSE={mse:.4f} PSNR={psnr:.2f} dB SSIM={ssim_value:.4f}")

    if not rows:
        raise SystemExit("No paired test images were evaluated.")
    csv_path = RESULTS_DIR / "enhanced_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "mae", "mse", "psnr", "ssim"])
        writer.writerows(rows)
        writer.writerow(["AVERAGE", np.mean([r[1] for r in rows]), np.mean([r[2] for r in rows]),
                         np.mean([r[3] for r in rows]), np.mean([r[4] for r in rows])])
    print("\nAVERAGES")
    print("Images evaluated:", len(rows))
    print("MAE :", round(float(np.mean([r[1] for r in rows])), 4))
    print("MSE :", round(float(np.mean([r[2] for r in rows])), 4))
    print("PSNR:", round(float(np.mean([r[3] for r in rows])), 2), "dB")
    print("SSIM:", round(float(np.mean([r[4] for r in rows])), 4))
    print("Metrics:", csv_path)
    print("Visual comparisons:", COMPARE_DIR)


if __name__ == "__main__":
    main()
