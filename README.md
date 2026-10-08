# Automatic Image Colorization

A practical implementation of an automatic image colorization system based on the methodology described in the project report.

The system takes a grayscale landscape image and predicts its color automatically using image features, Support Vector Regression (SVR), and Markov Random Field (MRF) based smoothing.

## Methodology

The colorization pipeline follows these steps:

**RGB image**
→ **YUV color space**
→ **Y (luminance) channel**
→ **SLIC superpixel segmentation**
→ **10×10 luminance patches**
→ **2D FFT features**
→ **Two SVR models for U and V chrominance**
→ **MRF / ICM smoothing**
→ **YUV to RGB conversion**
→ **Colorized image**

The input to the model is the luminance (Y) channel. The two SVR models independently predict the U and V chrominance values for each superpixel.

MRF-based smoothing is then applied to encourage consistent colors between similar neighboring superpixels.

---

## Project Structure

```text
automatic_image_colorization/
│
├── data/
│   ├── train/              # Color training images
│   ├── test/               # Grayscale test images
│   └── test_original/      # Original color versions for evaluation
│
├── models/
│   ├── u_svr.joblib        # Trained U chrominance model
│   └── v_svr.joblib        # Trained V chrominance model
│
├── outputs/
│   └── evaluation/         # Generated colorized test images
│
├── results/
│   └── metrics.csv         # Evaluation metrics
│
├── colorize.py             # Colorizes a grayscale image
├── train.py                # Trains the U and V SVR models
├── evaluate.py             # Evaluates the model
├── utils.py                # Image processing and feature extraction
├── download_dataset.py     # Downloads and prepares the dataset
├── requirements.txt        # Python dependencies
└── README.md