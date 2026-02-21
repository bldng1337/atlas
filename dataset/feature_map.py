import os
import sys

import numpy as np
import richdem as rd
from scipy import ndimage
from skan.csr import Skeleton, summarize
from skimage import feature, filters, morphology
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


def suppress_system_output(func):
    def wrapper(*args, **kwargs):
        sys.stdout.flush()
        sys.stderr.flush()

        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)

        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            return func(*args, **kwargs)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()

            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)

            os.close(devnull_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)

    return wrapper


def num_features(m):
    m = m * 255
    m = ndimage.gaussian_filter(m, sigma=0.7)
    m = m > 0

    labeled_array, num_features = ndimage.label(
        m, structure=ndimage.generate_binary_structure(m.ndim, 2)
    )
    pixel_counts = np.bincount(labeled_array.ravel())
    feature_pixel_counts = pixel_counts[1:]
    return num_features, feature_pixel_counts


def clean(m, max_features=5):
    skeleton = m.astype(bool)

    labeled = label(skeleton)
    regions = regionprops(labeled)

    regions_sorted = sorted(regions, key=lambda x: x.area, reverse=True)

    result = np.zeros_like(m, dtype=bool)
    features_to_keep = regions_sorted[:max_features]

    for region in features_to_keep:
        result[labeled == region.label] = True

    return result.astype(np.uint8)


def skeleton(m, blur_sigma=2):
    m = ndimage.gaussian_filter(m * 15, sigma=blur_sigma) + m
    m = skeletonize(m)
    return m


def process_flow(flow, high_percentile=99.5, low_percentile=95, blur_sigma=1.5):
    high_thresh = np.percentile(flow, high_percentile)
    low_thresh = np.percentile(flow, low_percentile)
    hysteresis_binary = filters.apply_hysteresis_threshold(
        flow, low=low_thresh, high=high_thresh
    )
    return hysteresis_binary


def get_maps(
    dem,
    dem_size=30,
    canny_sigma=1.5,
    blur_sigma=1.5,
    slope_threshold_steep=20.0,
    slope_threshold_flat=2.0,
    high_percentile=99.7,
    low_percentile=99,
    min_branch_length=100,
    max_features=2,
    max_total_pixels=5000,
    line_sigma=1.5,
):
    cell_size = dem_size / dem.shape[1]
    geotr = (0.0, cell_size, 0.0, 0.0, 0.0, -cell_size)
    dem = dem.reshape(dem.shape[1], dem.shape[1])
    dem = ndimage.gaussian_filter(dem, sigma=blur_sigma)
    dem_inv = dem.max() - dem
    dem = rd.rdarray(dem, no_data=-9999)
    dem.geotransform = geotr
    dem_inv = rd.rdarray(dem_inv, no_data=-9999)
    dem_inv.geotransform = geotr
    # rd.FillDepressions(dem, in_place=True)
    # rd.FillDepressions(dem_inv, in_place=True)
    flow = rd.FlowAccumulation(dem, method="D8")
    flow_inv = rd.FlowAccumulation(dem_inv, method="D8")

    valleys = process_flow(flow, high_percentile, low_percentile)
    valleys = skeleton(valleys)
    valleys = clean(
        valleys,
        max_features=max_features,
    )
    valleys = ndimage.gaussian_filter(valleys * 10, sigma=line_sigma) + valleys

    slope = rd.TerrainAttribute(dem, attrib="slope_degrees")
    flat_regions = slope < slope_threshold_flat
    steep_terrain = slope > slope_threshold_steep
    dem_norm = dem / dem.max()
    cliff_edges = feature.canny(dem_norm, sigma=canny_sigma)
    cliff_edges = cliff_edges & steep_terrain
    cliff_edges = skeleton(cliff_edges)
    cliff_edges = clean(
        cliff_edges,
        max_features=max_features,
    )
    cliff_edges = (
        ndimage.gaussian_filter(cliff_edges * 10, sigma=line_sigma) + cliff_edges
    )

    ridges = process_flow(flow_inv, high_percentile, low_percentile)
    ridges = ridges * (flat_regions == 0)
    ridges = skeleton(ridges)
    ridges = clean(
        ridges,
        max_features=max_features,
    )
    ridges = ndimage.gaussian_filter(ridges * 10, sigma=line_sigma) + ridges

    return valleys, ridges, cliff_edges, flat_regions


def get_map_combined(
    dem,
    dem_size=30,
    canny_sigma=2,
    blur_sigma=2,
    slope_threshold_steep=20.0,
    slope_threshold_flat=5,
    high_percentile=99.1,
    low_percentile=94.0,
    max_features=10,
    line_sigma=0.2,
):
    valleys, ridges, cliff, flat_regions = get_maps(
        dem,
        dem_size=dem_size,
        canny_sigma=canny_sigma,
        blur_sigma=blur_sigma,
        slope_threshold_steep=slope_threshold_steep,
        slope_threshold_flat=slope_threshold_flat,
        high_percentile=high_percentile,
        low_percentile=int(low_percentile),
        max_features=max_features,
    )

    map_combined = np.zeros((dem.shape[0], dem.shape[1], 3))
    valleys = valleys > 0
    ridges = ridges > 0
    cliff = cliff > 0
    map_combined[:, :, 2] = valleys
    map_combined[:, :, 0] = ridges
    map_combined[:, :, 1] = cliff
    return map_combined, flat_regions
