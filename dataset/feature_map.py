import random
from typing import Tuple

import numpy as np
import richdem as rd
from scipy.ndimage import gaussian_filter
from skimage.feature import canny
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk, remove_small_objects, square


def compute_slope(dem_norm: np.ndarray) -> np.ndarray:
    dy, dx = np.gradient(dem_norm)
    return np.sqrt(dx**2 + dy**2)


def flow_accumulation(dem: np.ndarray) -> np.ndarray:
    rda = rd.rdarray(dem.astype(np.float64), no_data=-9999)
    rd.FillDepressions(rda, epsilon=True, in_place=True)
    acc = rd.FlowAccumulation(rda, method="D8")
    return np.array(acc, dtype=np.float64)


def strahler_approx(flow_acc: np.ndarray, n_orders: int = 6) -> np.ndarray:
    max_acc = flow_acc.max()
    if max_acc == 0:
        return np.zeros_like(flow_acc, dtype=np.uint8)
    thresholds = np.logspace(
        np.log10(max_acc * 0.00004),
        np.log10(max_acc * 0.5),
        num=n_orders,
    )
    order = np.zeros(flow_acc.shape, dtype=np.uint8)
    for o, thresh in enumerate(thresholds, start=1):
        order[flow_acc >= thresh] = o
    return order


def dropout(
    mask: np.ndarray,
    order: np.ndarray,
    min_size: int = 10,
    min_dropout=0.2,
    max_dropout=0.5,
) -> np.ndarray:
    unique_orders = np.unique(order[mask > 0])[::-1]
    res = mask.copy()
    max_order = unique_orders.max()
    for o in unique_orders:
        dropout_prob = min_dropout + (1 - o / (max_order + 1e-9)) * (
            max_dropout - min_dropout
        )
        dropout_mask = (order == o) & (mask > 0)
        higher_order_mask = (order > o) & (res > 0)
        labeled = label(dropout_mask)
        for region in regionprops(labeled):
            reg = labeled == region.label
            if region.area < min_size:
                res[reg] = 0
                continue
            if (
                o != max_order
                and not np.any(dilation(reg, square(3)) & higher_order_mask)
            ):  # If no higher order branches exist we drop it so there are no floating branches
                res[reg] = 0
                continue
            if np.random.rand() < dropout_prob:
                res[reg] = 0
                continue
            res[reg] = 1
    return res


def extract_terrain_features(
    dem: np.ndarray,
    gaussian_sigma: float = 1.5,
    flow_acc_percentile: float = 99.0,
    n_strahler_orders: int = 6,
    canny_sigma: float = 1.5,
    canny_low_threshold: float = 0.05,
    canny_high_threshold: float = 0.15,
    cliff_slope_percentile: float = 75.0,
    flat_slope_percentile: float = 20.0,
    thickness: int = 4,
    min_feature_size: int = 80,
    line_sigma: float = 0.2,
    min_dropout_rate: float = 0.2,
    max_dropout_rate: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    dem = dem.astype(np.float64)
    dem_blur = gaussian_filter(dem, sigma=gaussian_sigma)
    lo, hi = dem_blur.min(), dem_blur.max()
    dem_norm = (dem_blur - lo) / (hi - lo + 1e-9)
    slope = compute_slope(dem_norm)

    flat_thresh = np.percentile(slope, flat_slope_percentile)
    flat_mask = (slope < flat_thresh).astype(np.float32)

    # Valley
    flow_acc = flow_accumulation(dem_blur)
    valley_thresh = np.percentile(flow_acc, flow_acc_percentile)
    valley_mask = flow_acc >= valley_thresh
    valley_order = strahler_approx(flow_acc, n_orders=n_strahler_orders)

    # Ridge
    dem_inv = dem_blur.max() - dem_blur
    flow_acc_inv = flow_accumulation(dem_inv)
    ridge_thresh = np.percentile(flow_acc_inv, flow_acc_percentile)
    ridge_mask = flow_acc_inv >= ridge_thresh
    ridge_order = strahler_approx(flow_acc_inv, n_orders=n_strahler_orders)

    # Cliff
    edges = canny(
        dem_norm,
        sigma=canny_sigma,
        low_threshold=canny_low_threshold,
        high_threshold=canny_high_threshold,
    )
    steep_thresh = np.percentile(slope, cliff_slope_percentile)
    steep_mask = dilation(slope > steep_thresh, disk(2))
    cliff_mask = edges & steep_mask

    cliff_order = np.zeros(dem.shape, dtype=np.uint8)
    for rank, pct in enumerate([50, 65, 75, 85, 95], start=1):
        cliff_order[cliff_mask & (slope >= np.percentile(slope, pct))] = rank

    for ch, order in (
        (valley_mask, valley_order),
        (ridge_mask, ridge_order),
        (cliff_mask, cliff_order),
    ):
        if np.any(ch):
            ch *= dropout(
                ch,
                order,
                min_size=min_feature_size,
                min_dropout=min_dropout_rate,
                max_dropout=max_dropout_rate,
            )
            ch += dilation(ch, disk(thickness))
            ch *= remove_small_objects(ch, min_size=min_feature_size)
            ch += gaussian_filter(ch, sigma=line_sigma)

    map = np.stack(
        [
            np.clip(valley_mask, 0, 1),  # R — valleys
            np.clip(ridge_mask, 0, 1),  # G — ridges
            np.clip(cliff_mask, 0, 1),  # B — cliffs
        ],
        axis=-1,
    ).astype(np.float32)

    return map, flat_mask


def get_map_combined(
    dem,
    dem_size=30,
    canny_sigma=2,
    blur_sigma=2,
    slope_threshold_steep=20.0,
    slope_threshold_flat=10,
    high_percentile=99.8,
    low_percentile=80.0,
    max_features=10,
    line_sigma=0.2,
):
    gaussian_sigma = random.uniform(1.0, 2.5)
    flow_acc_percentile = random.uniform(98.0, 99.9)
    n_strahler_orders = random.randint(5, 8)
    canny_sigma = random.uniform(1.0, 2.5)
    canny_low_threshold = random.uniform(0.03, 0.07)
    canny_high_threshold = random.uniform(0.12, 0.18)
    cliff_slope_percentile = random.uniform(70.0, 80.0)
    flat_slope_percentile = random.uniform(15.0, 25.0)
    thickness = random.randint(3, 6)
    min_feature_size = random.randint(60, 120)
    line_sigma = random.uniform(0.15, 0.3)
    min_dropout_rate = random.uniform(0.15, 0.25)
    max_dropout_rate = random.uniform(0.7, 0.9)

    maps, flat = extract_terrain_features(
        dem,
        # gaussian_sigma=gaussian_sigma,
        # flow_acc_percentile=flow_acc_percentile,
        # n_strahler_orders=n_strahler_orders,
        # canny_sigma=canny_sigma,
        # canny_low_threshold=canny_low_threshold,
        # canny_high_threshold=canny_high_threshold,
        # cliff_slope_percentile=cliff_slope_percentile,
        # flat_slope_percentile=flat_slope_percentile,
        thickness=thickness,
        min_feature_size=min_feature_size,
        line_sigma=line_sigma,
        min_dropout_rate=min_dropout_rate,
        max_dropout_rate=max_dropout_rate,
    )
    return maps, flat
