# Automatic Image Colorization

A practical implementation of the methodology described in the project report:
YUV color space -> SLIC superpixels -> 10x10 luminance patches -> 2D FFT features
-> two Support Vector Regressors (U and V) -> MRF-style smoothing with ICM
-> RGB output.

## Project structure

- data/train/ : training COLOR images
- data/test/  : grayscale test images
- models/     : trained SVR models
- outputs/    : generated colorized images
- train.py    : train the two SVR models
- colorize.py : colorize a grayscale image
- utils.py    : shared image/feature functions

## Windows setup

```powershell
py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1. Add data

Put color training images in:

data/train/

Put grayscale images to test in:

data/test/

For a first demo, even 20-50 landscape images are enough to verify the pipeline.
For meaningful results, use a larger and visually consistent dataset.

## 2. Train

```powershell
python train.py
```

## 3. Colorize

```powershell
python colorize.py --input data/test/example.jpg
```

The result is written to outputs/.

## Parameters based on the report

SVR epsilon = 0.0625
SVR C = 0.125
ICM gamma = 2.0

The original report used 98 training and 118 test Yellowstone landscape images.
This implementation does not assume those files are available.
