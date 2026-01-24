from turtle import width
from datasets import load_dataset
from io import BytesIO
import numpy as np
import PIL.Image as PILImage

import dataprep


OUTPUT = r"D:\DATA\dataset\atlas\data_with_features.parquet\*.parquet"


def to_uint8_rgb(arr):
    """Convert normalized float (-1..1) RGB array to uint8 RGB image."""
    img = np.clip(((arr + 1.0) / 2.0) * 255.0, 0, 255).astype(np.uint8)
    return img


def to_uint8_gray(arr):
    """Convert normalized float (-1..1) single-channel array to uint8 grayscale image."""
    img = np.clip(((arr + 1.0) / 2.0) * 255.0, 0, 255).astype(np.uint8)
    return img

def test_sample(sample,show=False,do_print=True):
    width=768
    height=768
    if do_print:
        print("Decoding RGB via dataprep.decode_img...")
    img_arr = dataprep.decode_img(sample, width=width, height=height)
    if do_print:
        print("Size:", img_arr.shape, "Dtype:", img_arr.dtype)
    if show:
        img_vis = to_uint8_rgb(img_arr)
        PILImage.fromarray(img_vis, mode="RGB").show()

    if do_print:
        print("Decoding DEM via dataprep.decode_dem...")
    dem_arr = dataprep.decode_dem(sample, width=width, height=height)
    if do_print:
        print("Size:", dem_arr.shape, "Dtype:", dem_arr.dtype)
    if show:
        dem_vis = to_uint8_gray(dem_arr)
        PILImage.fromarray(dem_vis, mode="L").show()

    if do_print:
        print("Decoding feature map via dataprep.decode_feature...")
    feat = dataprep.decode_feature(sample, width=width, height=height)
    if do_print:
        print("Size:", feat.shape, "Dtype:", feat.dtype)
    if show:
        feat_vis = to_uint8_rgb(feat)
        PILImage.fromarray(feat_vis, mode="RGB").show()

    if do_print:
        print("Decoding cloud mask via dataprep.decode_cloud_cover...")
    cloud = dataprep.decode_cloud_cover(sample, width=width, height=height)
    if do_print:
        print("Size:", cloud.shape, "Dtype:", cloud.dtype)
    if show:
        c = cloud.squeeze()
        cloud_vis = (c>0).astype(np.uint8)*255
        PILImage.fromarray(cloud_vis, mode="L").show()
    
    # asserts
    assert img_arr.shape == (width, height, 3)
    assert dem_arr.shape == (width, height)
    assert feat.shape == (width, height, 3)
    assert cloud.shape == (width, height)

def main():
    print("Loading dataset...")
    ds = load_dataset("parquet", data_files=OUTPUT, streaming=True)
    # ds = ds.shuffle(seed=99)
    print("Columns:", ds.column_names)
    # for i in range(50):
    sample = next(iter(ds["train"]))
    # print(f"\nSample {i}:")
    test_sample(sample, show=True,do_print=False)
        

    print("Done.")


if __name__ == "__main__":
    main()

