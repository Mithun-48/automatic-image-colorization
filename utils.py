"""Shared functions: YUV -> SLIC superpixels -> FFT/statistical features -> SVR -> MRF/ICM -> RGB."""
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2yuv, yuv2rgb

PATCH_SIZE = 10          # fine-scale FFT patch (as in the report)
CONTEXT_PATCH = 30       # coarse-scale patch, resized to 10x10 -> scene context
N_SEGMENTS = 600         # more superpixels = finer colour detail (was 250)
COMPACTNESS = 0.08       # FIX: was 10.0. Y is in [0,1], so 10.0 made SLIC ignore
                         # image edges and produce a plain square grid.
LUMA_COL = 0             # column of the feature matrix holding mean luminance


# ----------------------------------------------------------------- image I/O
def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(path, rgb):
    arr = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def resize_width(rgb, width=500):
    h, w = rgb.shape[:2]
    if w == width:
        return rgb
    new_h = max(1, int(round(h * width / w)))
    interp = cv2.INTER_AREA if width < w else cv2.INTER_CUBIC
    return cv2.resize(rgb, (width, new_h), interpolation=interp)


def make_yuv(rgb):
    return rgb2yuv(rgb.astype(np.float32) / 255.0).astype(np.float32)


def yuv_to_rgb(y, u, v):
    rgb = yuv2rgb(np.stack([y, u, v], axis=2).astype(np.float32))
    return np.clip(rgb, 0.0, 1.0)


# ------------------------------------------------------------- superpixels
def segment_y(y):
    labels = slic(
        y.astype(np.float64),
        n_segments=N_SEGMENTS,
        compactness=COMPACTNESS,
        sigma=1.0,
        start_label=0,
        channel_axis=None,
    )
    # make labels contiguous 0..n-1
    _, labels = np.unique(labels, return_inverse=True)
    return labels.reshape(y.shape).astype(np.int32)


def centroids(labels):
    n = int(labels.max()) + 1
    rows, cols = np.indices(labels.shape)
    flat = labels.ravel()
    cnt = np.maximum(np.bincount(flat, minlength=n), 1).astype(np.float64)
    cy = np.bincount(flat, rows.ravel(), n) / cnt
    cx = np.bincount(flat, cols.ravel(), n) / cnt
    return np.stack([cy, cx], axis=1).astype(np.float32)


# ---------------------------------------------------------------- features
def _fft_mag(patch):
    patch = patch - patch.mean()                 # DC is stored separately (mean Y)
    return np.log1p(np.abs(np.fft.rfft2(patch))).astype(np.float32).ravel()


def extract_features(y, labels):
    """Per-superpixel features:
       8 statistics (mean Y, std Y, gradient, position, size, global context)
       + FFT of a 10x10 patch + FFT of a 30x30 patch resized to 10x10."""
    h, w = y.shape
    n = int(labels.max()) + 1
    centers = centroids(labels)
    flat = labels.ravel()

    counts = np.bincount(flat, minlength=n).astype(np.float64)
    cnt = np.maximum(counts, 1)
    yf = y.ravel().astype(np.float64)
    mean_y = np.bincount(flat, yf, n) / cnt
    var_y = np.bincount(flat, yf * yf, n) / cnt - mean_y ** 2
    std_y = np.sqrt(np.maximum(var_y, 0))

    gx = cv2.Sobel(y.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mean_g = np.bincount(flat, np.sqrt(gx ** 2 + gy ** 2).ravel(), n) / cnt

    g_mean = float(y.mean())
    stats = np.stack([
        mean_y, std_y, mean_g,
        centers[:, 0] / h,            # vertical position: sky is on top, ground below
        centers[:, 1] / w,
        counts / (h * w) * n,
        np.full(n, g_mean),
        mean_y - g_mean,
    ], axis=1).astype(np.float32)

    pad = CONTEXT_PATCH // 2
    padded = np.pad(y, pad, mode="reflect")
    fine, coarse = [], []
    for cy, cx in centers:
        py, px = int(round(cy)) + pad, int(round(cx)) + pad
        h1 = PATCH_SIZE // 2
        p1 = padded[py - h1:py - h1 + PATCH_SIZE, px - h1:px - h1 + PATCH_SIZE]
        h2 = CONTEXT_PATCH // 2
        p2 = padded[py - h2:py + h2, px - h2:px + h2]
        p2 = cv2.resize(p2, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
        fine.append(_fft_mag(p1))
        coarse.append(_fft_mag(p2))

    features = np.hstack([stats, np.vstack(fine), np.vstack(coarse)])
    return features.astype(np.float32), centers


def segment_targets(yuv, labels):
    n = int(labels.max()) + 1
    flat = labels.ravel()
    cnt = np.maximum(np.bincount(flat, minlength=n), 1).astype(np.float64)
    u = np.bincount(flat, yuv[:, :, 1].ravel(), n) / cnt
    v = np.bincount(flat, yuv[:, :, 2].ravel(), n) / cnt
    return u.astype(np.float32), v.astype(np.float32)


# --------------------------------------------------------------- MRF / ICM
def neighbor_graph(labels, luma, luma_sigma=0.10):
    """Adjacency between touching superpixels. Edge weight is small when the
    two regions differ strongly in brightness, so colour does not leak across
    real object boundaries (edge-aware smoothing)."""
    n = int(labels.max()) + 1
    codes = []
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        m = a != b
        lo = np.minimum(a[m], b[m]).astype(np.int64)
        hi = np.maximum(a[m], b[m]).astype(np.int64)
        codes.append(lo * n + hi)
    codes = np.unique(np.concatenate(codes))
    nbrs = [[] for _ in range(n)]
    wts = [[] for _ in range(n)]
    for i, j in zip(codes // n, codes % n):
        w = float(np.exp(-((luma[i] - luma[j]) ** 2) / (2 * luma_sigma ** 2)))
        nbrs[i].append(j); wts[i].append(w)
        nbrs[j].append(i); wts[j].append(w)
    return ([np.array(a, dtype=np.int64) for a in nbrs],
            [np.array(a, dtype=np.float64) for a in wts])


def icm_smooth(local_prediction, graph, gamma=0.25, sigma=1.0, iterations=15):
    """Quadratic ICM. Energy_i = (c_i-mu_i)^2/(2 s^2) + gamma * sum_j w_ij (c_i-c_j)^2.
    Closed-form update = weighted average of own SVR prediction and neighbours."""
    nbrs, wts = graph
    mu = local_prediction.astype(np.float64)
    c = mu.copy()
    inv_s2 = 1.0 / (sigma * sigma)
    for _ in range(iterations):
        old = c.copy()
        for i in range(len(c)):
            if len(nbrs[i]) == 0:
                continue
            num = inv_s2 * mu[i] + 2 * gamma * np.dot(wts[i], c[nbrs[i]])
            den = inv_s2 + 2 * gamma * wts[i].sum()
            c[i] = num / den
        if np.max(np.abs(c - old)) < 1e-6:
            break
    return c.astype(np.float32)


# ------------------------------------------------ edge-aware pixel-level chroma
def guided_filter(guide, src, radius=10, eps=1e-3):
    """He et al. guided filter: spreads the superpixel colours to pixels while
    following the edges of the luminance image."""
    k = (2 * radius + 1, 2 * radius + 1)
    box = lambda x: cv2.boxFilter(x, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
    I = guide.astype(np.float32)
    p = src.astype(np.float32)
    mI, mp = box(I), box(p)
    cov = box(I * p) - mI * mp
    var = box(I * I) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return box(a) * I + box(b)
