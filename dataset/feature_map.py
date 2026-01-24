import richdem as rd
from skimage import feature,filters
from scipy import ndimage
from skimage.morphology import skeletonize
from skan.csr import skeleton_to_csgraph, Skeleton, summarize
import numpy as np
import sys, os


def num_features(m):
    m=m*255
    m=ndimage.gaussian_filter(m, sigma=0.7)
    m=m>0

    labeled_array, num_features = ndimage.label(m,structure=ndimage.generate_binary_structure(m.ndim,2))
    pixel_counts = np.bincount(labeled_array.ravel())
    feature_pixel_counts = pixel_counts[1:]
    return num_features, feature_pixel_counts

def clean(m, min_length=50, max_features=3, max_total_pixels=None):
    """
    Clean a binary skeleton image by removing short terminal branches
    and keeping only the largest connected components.
    
    Parameters:
        m: Binary image to clean
        min_length: Minimum branch length to keep (in pixels)
        max_features: Maximum number of features to keep
        max_total_pixels: If set, will thin out features until below this pixel count
    
    Returns:
        Cleaned skeleton image
    """
    m = m > 0
    if not np.any(m):
        return m
    
    m = skeletonize(m)
    current_skeleton_image = m.copy()
    
    escape_hatch=0
    max_iterations=5000
    prev_pixel_count = None
    no_progress_count = 0
    
    while True:
        escape_hatch+=1
        if escape_hatch>max_iterations:
            break
        
        # Skip if empty
        if not np.any(current_skeleton_image):
            break
        
        # Check for progress - if pixel count hasn't changed for several iterations, break
        current_pixel_count = np.sum(current_skeleton_image)
        if prev_pixel_count is not None and current_pixel_count == prev_pixel_count:
            no_progress_count += 1
            if no_progress_count > 5:
                break
        else:
            no_progress_count = 0
        prev_pixel_count = current_pixel_count
            
        try:
            skel = Skeleton(current_skeleton_image)
            branch_df = summarize(skel,separator="-")
        except ValueError:
            break
        
        if branch_df.empty:
            break
        
        # Get column names dynamically (skan versions differ)
        cols = branch_df.columns.tolist()
        node_src_col = 'node-id-src' if 'node-id-src' in cols else 'node_id_src'
        node_dst_col = 'node-id-dst' if 'node-id-dst' in cols else 'node_id_dst'
        branch_dist_col = 'branch-distance' if 'branch-distance' in cols else 'branch_distance'
            
        # Get the degree of each node (junctions and endpoints)
        degrees = skel.degrees
        
        # Identify terminal branches (connected to an endpoint with degree 1)
        is_terminal_src = degrees[branch_df[node_src_col].astype(int)] == 1
        is_terminal_dst = degrees[branch_df[node_dst_col].astype(int)] == 1
        is_terminal_branch = is_terminal_src | is_terminal_dst
        
        # Identify short, terminal branches
        short_branches = branch_df[branch_dist_col] < min_length
        branches_to_remove_mask = short_branches & is_terminal_branch
        
        if not branches_to_remove_mask.any():
            break
            
        # Collect the pixel IDs for the branches to remove
        pixels_to_remove_ids = []
        for idx in branch_df.index[branches_to_remove_mask]:
            pixels_to_remove_ids.extend(skel.path(idx))
        
        # Remove the pixels from the skeleton image
        if pixels_to_remove_ids:
            current_skeleton_image.flat[pixels_to_remove_ids] = False
    
    # Keep only the largest connected components
    if np.any(current_skeleton_image):
        labeled_array, n_features = ndimage.label(current_skeleton_image, 
                                                   structure=ndimage.generate_binary_structure(2, 2))
        if n_features > max_features:
            pixel_counts = np.bincount(labeled_array.ravel())
            # Ignore background (index 0)
            pixel_counts[0] = 0
            # Get indices of largest components
            largest_labels = np.argsort(pixel_counts)[-max_features:]
            # Create mask keeping only largest components
            mask = np.isin(labeled_array, largest_labels)
            current_skeleton_image = current_skeleton_image & mask
        
        # If max_total_pixels is set, further reduce to meet target
        if max_total_pixels is not None and np.sum(current_skeleton_image) > max_total_pixels:
            labeled_array, n_features = ndimage.label(current_skeleton_image, 
                                                       structure=ndimage.generate_binary_structure(2, 2))
            pixel_counts = np.bincount(labeled_array.ravel())
            pixel_counts[0] = 0
            # Sort by size (largest first)
            sorted_labels = np.argsort(pixel_counts)[::-1]
            cumsum = 0
            keep_labels = []
            for label in sorted_labels:
                if pixel_counts[label] == 0:
                    continue
                if cumsum + pixel_counts[label] <= max_total_pixels:
                    cumsum += pixel_counts[label]
                    keep_labels.append(label)
                if len(keep_labels) >= max_features:
                    break
            if keep_labels:
                mask = np.isin(labeled_array, keep_labels)
                current_skeleton_image = current_skeleton_image & mask
             
    return current_skeleton_image


def skeleton(m, blur_sigma=2):
    m = ndimage.gaussian_filter(m*15, sigma=blur_sigma)+m
    m = skeletonize(m)
    return m

def process_flow(flow, high_percentile=99.5, low_percentile=95, blur_sigma=1.5):
    """Process flow accumulation with stricter thresholds for fewer features."""
    high_thresh = np.percentile(flow, high_percentile)
    low_thresh = np.percentile(flow, low_percentile)
    hysteresis_binary = filters.apply_hysteresis_threshold(flow, low=low_thresh, high=high_thresh)
    return hysteresis_binary

def get_maps(dem, dem_size=30, canny_sigma=1.5, blur_sigma=1.5, slope_threshold_steep=20.0, 
             slope_threshold_flat=2.0, high_percentile=99.7, low_percentile=99, 
             min_branch_length=100, max_features=2, max_total_pixels=5000,line_sigma=1.5):
    cell_size=dem_size/dem.shape[1]
    geotr=(0.0, cell_size, 0.0, 0.0, 0.0, -cell_size)
    dem=dem.reshape(dem.shape[1],dem.shape[1])
    dem = ndimage.gaussian_filter(dem, sigma=blur_sigma)
    dem_inv=dem.max()-dem
    dem = rd.rdarray(dem, no_data=0)
    dem.geotransform = geotr
    dem_inv = rd.rdarray(dem_inv, no_data=0)
    dem_inv.geotransform = geotr

    flow = rd.FlowAccumulation(dem, method='D8')
    flow_inv = rd.FlowAccumulation(dem_inv, method='D8')
    
    valleys = process_flow(flow, high_percentile, low_percentile)
    valleys = skeleton(valleys)
    valleys = clean(valleys, min_length=min_branch_length, max_features=max_features, 
                    max_total_pixels=max_total_pixels)
    valleys = ndimage.gaussian_filter(valleys*5, sigma=line_sigma)+valleys
    
    ridges = process_flow(flow_inv, high_percentile, low_percentile)
    ridges = skeleton(ridges)
    ridges = clean(ridges, min_length=min_branch_length, max_features=max_features,
                   max_total_pixels=max_total_pixels)
    ridges = ndimage.gaussian_filter(ridges*5, sigma=line_sigma)+ridges
    
    slope = rd.TerrainAttribute(dem, attrib='slope_degrees')
    flat_regions = (slope < slope_threshold_flat)
    steep_terrain = (slope > slope_threshold_steep)
    dem_norm=dem/dem.max()
    cliff_edges = feature.canny(dem_norm, sigma=canny_sigma)
    cliff_edges = (cliff_edges & steep_terrain)
    # cliff_edges = skeleton(cliff_edges)
    # Also clean and limit cliff edges
    cliff_edges = clean(cliff_edges, min_length=min_branch_length, max_features=max_features,
                        max_total_pixels=max_total_pixels)
    cliff_edges = ndimage.gaussian_filter(cliff_edges*5, sigma=line_sigma)+cliff_edges
    
    return valleys, ridges, cliff_edges, flat_regions


def get_map_combined(dem, dem_size=30, canny_sigma=1.5, blur_sigma=1.5, slope_threshold_steep=20.0, 
                     slope_threshold_flat=2.0, high_percentile=99.7, low_percentile=99.0,
                     min_branch_length=100, max_features=1, max_total_pixels=3000,line_sigma=1.5):
    valleys, ridges, cliff, flat_regions = get_maps(
        dem, dem_size=dem_size, canny_sigma=canny_sigma, blur_sigma=blur_sigma,
        slope_threshold_steep=slope_threshold_steep, slope_threshold_flat=slope_threshold_flat,
        high_percentile=high_percentile, low_percentile=int(low_percentile),
        min_branch_length=min_branch_length, max_features=max_features,
        max_total_pixels=max_total_pixels
    )
    
    map_combined=np.zeros((dem.shape[0], dem.shape[1], 3))
    valleys=valleys>0
    ridges=ridges>0
    cliff=cliff>0
    map_combined[:,:,2]=valleys
    map_combined[:,:,0]=ridges
    map_combined[:,:,1]=cliff
    return map_combined, flat_regions