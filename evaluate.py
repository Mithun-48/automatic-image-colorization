import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from colorize import colorize


test_folder = "data/test"
original_folder = "data/test_original"

output_folder = "outputs/evaluation"
comparison_folder = "outputs/comparisons"
results_folder = "results"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(comparison_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

results = []

images = os.listdir(test_folder)

print("Number of test images:", len(images))
print()

for image_name in images:

    test_path = os.path.join(test_folder, image_name)
    original_path = os.path.join(original_folder, image_name)

    if not os.path.exists(original_path):
        continue

    print("Processing:", image_name)

    output_path = os.path.join(
        output_folder,
        image_name
    )

    colorize(test_path, output_path)

    original = cv2.imread(original_path)
    predicted = cv2.imread(output_path)

    original = cv2.resize(
        original,
        (predicted.shape[1], predicted.shape[0])
    )

    original_float = original.astype(float) / 255
    predicted_float = predicted.astype(float) / 255

    mae = np.mean(
        np.abs(original_float - predicted_float)
    )

    mse = np.mean(
        (original_float - predicted_float) ** 2
    )

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10(1 / mse)

    ssim_value = ssim(
        original_float,
        predicted_float,
        channel_axis=2,
        data_range=1
    )

    results.append([
        image_name,
        mae,
        mse,
        psnr,
        ssim_value
    ])

    print("MAE :", round(mae, 4))
    print("MSE :", round(mse, 4))
    print("PSNR:", round(psnr, 2), "dB")
    print("SSIM:", round(ssim_value, 4))
    print()

    grayscale = cv2.imread(test_path)

    grayscale = cv2.resize(
        grayscale,
        (predicted.shape[1], predicted.shape[0])
    )

    comparison = np.hstack([
        original,
        grayscale,
        predicted
    ])

    comparison_path = os.path.join(
        comparison_folder,
        image_name
    )

    cv2.imwrite(
        comparison_path,
        comparison
    )


results = np.array(results, dtype=object)

average_mae = np.mean(
    results[:, 1].astype(float)
)

average_mse = np.mean(
    results[:, 2].astype(float)
)

average_psnr = np.mean(
    results[:, 3].astype(float)
)

average_ssim = np.mean(
    results[:, 4].astype(float)
)

print("=" * 40)
print("FINAL RESULTS")
print("=" * 40)

print("Images evaluated:", len(results))
print("Average MAE :", round(average_mae, 4))
print("Average MSE :", round(average_mse, 4))
print("Average PSNR:", round(average_psnr, 2), "dB")
print("Average SSIM:", round(average_ssim, 4))

with open("results/metrics.csv", "w") as file:
    file.write("image,mae,mse,psnr,ssim\n")

    for result in results:
        file.write(
            f"{result[0]},{result[1]},{result[2]},"
            f"{result[3]},{result[4]}\n"
        )

print()
print("Results saved to results/metrics.csv")