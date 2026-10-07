import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2yuv, yuv2rgb


PATCH_SIZE = 10
N_SEGMENTS = 250
COMPACTNESS = 10.0


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(path, rgb):
    arr = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def resize_width(rgb, width=500):
    h, w = rgb.shape[:2]
    if w == width:
        return rgb
    new_h = max(1, int(round(h * width / w)))
    return cv2.resize(rgb, (width, new_h), interpolation=cv2.INTER_AREA)


def make_yuv(rgb):
    rgb_float = rgb.astype(np.float32) / 255.0
    return rgb2yuv(rgb_float)


def segment_y(y):
    # SLIC expects an image-like array. We use luminance only,
    # matching the report's grayscale/luminance-driven segmentation.
    labels = slic(
        y,
        n_segments=N_SEGMENTS,
        compactness=COMPACTNESS,
        start_label=0,
        channel_axis=None,
    )
    return labels.astype(np.int32)


def centroids(labels):
    n = int(labels.max()) + 1
    centers = np.zeros((n, 2), dtype=np.float32)
    counts = np.zeros(n, dtype=np.int32)

    rows, cols = np.indices(labels.shape)
    flat_labels = labels.ravel()
    np.add.at(centers[:, 0], flat_labels, rows.ravel())
    np.add.at(centers[:, 1], flat_labels, cols.ravel())
    np.add.at(counts, flat_labels, 1)
    centers /= np.maximum(counts[:, None], 1)
    return centers


def fft_feature_patch(y, cy, cx, patch_size=PATCH_SIZE):
    # Extract a fixed patch centered at the superpixel centroid.
    # Reflect padding avoids losing features near image borders.
    half = patch_size // 2
    padded = np.pad(y, ((half, half), (half, half)), mode="reflect")
    py = int(round(cy)) + half
    px = int(round(cx)) + half

    patch = padded[
        py - half:py - half + patch_size,
        px - half:px - half + patch_size,
    ]

    if patch.shape != (patch_size, patch_size):
        patch = cv2.resize(patch, (patch_size, patch_size))

    fft = np.fft.fft2(patch)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))
    return magnitude.astype(np.float32).ravel()


def extract_features(y, labels):
    centers = centroids(labels)
    features = np.vstack([
        fft_feature_patch(y, cy, cx)
        for cy, cx in centers
    ])
    return features, centers


def segment_targets(yuv, labels):
    n = int(labels.max()) + 1
    u = np.zeros(n, dtype=np.float32)
    v = np.zeros(n, dtype=np.float32)

    flat = labels.ravel()
    np.add.at(u, flat, yuv[:, :, 1].ravel())
    np.add.at(v, flat, yuv[:, :, 2].ravel())

    counts = np.bincount(flat, minlength=n).astype(np.float32)
    u /= np.maximum(counts, 1)
    v /= np.maximum(counts, 1)
    return u, v


def assign_segments(values, labels):
    return values[labels]


def neighbor_graph(labels, features, threshold=None):
    """
    Build an adjacency graph between touching superpixels.
    If threshold is supplied, neighboring segments are connected only
    when their feature distance is below the threshold, as described
    in the report.
    """
    n = int(labels.max()) + 1
    edges = set()

    # Horizontal and vertical boundaries are enough to recover
    # touching superpixels.
    for a, b in [
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1, :], labels[1:, :]),
    ]:
        mask = a != b
        pairs = np.stack([a[mask], b[mask]], axis=1)
        for x, y in pairs:
            x, y = int(x), int(y)
            if x > y:
                x, y = y, x
            edges.add((x, y))

    graph = [[] for _ in range(n)]
    for i, j in edges:
        if threshold is None:
            graph[i].append(j)
            graph[j].append(i)
        else:
            d = np.linalg.norm(features[i] - features[j])
            if d < threshold:
                graph[i].append(j)
                graph[j].append(i)
    return graph


def icm_smooth(initial, local_prediction, graph, gamma=2.0, sigma=1.0, iterations=10):
    """
    Continuous quadratic ICM-style update.

    Energy per segment:
      (c_i - mu_i)^2 / (2 sigma^2)
      + gamma * sum_j (c_i - c_j)^2

    Minimizing with neighbors fixed gives a weighted average update.
    """
    c = initial.astype(np.float64).copy()
    mu = local_prediction.astype(np.float64)

    denom_local = 1.0 / (sigma * sigma)

    for _ in range(iterations):
        old = c.copy()

        for i, nbrs in enumerate(graph):
            if not nbrs:
                c[i] = mu[i]
                continue

            neighbor_sum = sum(c[j] for j in nbrs)
            denom = denom_local + 2.0 * gamma * len(nbrs)
            numer = denom_local * mu[i] + 2.0 * gamma * neighbor_sum
            c[i] = numer / denom

        if np.max(np.abs(c - old)) < 1e-5:
            break

    return c.astype(np.float32)


def yuv_to_rgb(y, u, v):
    yuv = np.stack([y, u, v], axis=2).astype(np.float32)
    rgb = yuv2rgb(yuv)
    return np.clip(rgb, 0.0, 1.0)
