import random
from io import BytesIO

import cv2
import numpy as np
import PIL.Image as PILImage
import torch
from rasterio.io import MemoryFile

from dataset.feature_map import get_map_combined


def norm(data, center=True):
    lo, hi = data.min(), data.max()
    denom = hi - lo
    if denom < 1e-8:
        return np.zeros_like(data, dtype=np.float32)
    if center:
        return ((data - lo) / denom) * 2.0 - 1.0
    return (data - lo) / denom


def decode_feature(batch, width=768, height=768):
    arr = np.frombuffer(batch["feature_map"], dtype=np.float64).astype(np.float32)
    size = 356 * 356 * 3
    if arr.size > size:
        arr = arr[:size]
    pad_width = size - arr.size
    if pad_width > 0:
        arr = np.pad(arr, (0, pad_width), mode="constant")
    arr = arr.reshape((356, 356, 3))
    arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)
    result = norm(arr, center=False)

    del arr
    return result


def decode_img(batch, width=768, height=768):
    bands = []
    for key in ("B04", "B03", "B02"):
        bbytes = batch[key]
        img_band = PILImage.open(BytesIO(bbytes))
        arr = np.array(img_band, dtype=np.float32)
        del img_band

        arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)
        bands.append(arr)
        del arr

    arr_rgb = np.stack(bands, axis=-1)
    arr_rgb = cv2.resize(arr_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    arr_rgb = norm(arr_rgb)
    del bands
    return arr_rgb


def decode_dem(batch, width=768, height=768, lower_bound=None, upper_bound=None):
    with MemoryFile(batch["DEM"]) as mem_f:
        with mem_f.open(driver="GTiff") as f:
            dem = f.read(1)

    dem = dem.astype(np.float32)
    dem = cv2.resize(dem, (width, height), interpolation=cv2.INTER_LINEAR)

    if lower_bound is not None and upper_bound is not None:
        result = np.clip(
            ((dem - lower_bound) / (upper_bound - lower_bound)) * 2.0 - 1.0, -1.5, 1.5
        )
    else:
        result = norm(dem)

    del dem
    return result


def decode_cloud_cover(batch, width=768, height=768):
    raw = batch["cloud_mask"]

    img = PILImage.open(BytesIO(bytes(raw))).convert("L")
    img_arr = np.array(img, dtype=np.float32)
    img_arr = (img_arr > 0).astype(np.float32)
    img_arr = cv2.resize(img_arr, (width, height), interpolation=cv2.INTER_NEAREST)
    del img
    return img_arr


def preprocess(
    batch,
    tokenizer,
    width,
    height,
    tokenizer_path=None,
    proportion_empty_prompts=0,
    lower_bound=None,
    upper_bound=None,
    generate_features=False,
    **kwargs,
):
    if tokenizer is None and tokenizer_path is not None:
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, subfolder="tokenizer")
    batch_size = len(batch)

    imgs = torch.zeros((batch_size, 3, height, width), dtype=torch.float32)
    dems = torch.zeros((batch_size, 3, height, width), dtype=torch.float32)
    fmaps = torch.zeros((batch_size, 3, height, width), dtype=torch.float32)
    cloud_masks = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)
    txts = []

    for idx, data in enumerate(batch):
        # Process image
        img_arr = decode_img(data, width, height)
        imgs[idx] = torch.from_numpy(img_arr).permute(2, 0, 1)
        del img_arr

        # Process DEM
        dem_arr = decode_dem(data, width, height, lower_bound, upper_bound)
        dem_tensor = torch.from_numpy(dem_arr).unsqueeze(0)
        dems[idx] = dem_tensor.expand(3, -1, -1)
        del dem_tensor

        # Process feature map
        if generate_features:
            fmaps[idx] = torch.zeros((3, height, width), dtype=torch.float32)
            feature_map, _ = get_map_combined(dem_arr * 1000, dem_size=width, **kwargs)
            feature_map = cv2.resize(
                feature_map, (width, height), interpolation=cv2.INTER_LINEAR
            )
            feature_map = np.clip(feature_map, 0, 1)
            fmaps[idx] = torch.from_numpy(feature_map).permute(2, 0, 1)
            del feature_map
        else:
            fmap_arr = decode_feature(data, width, height)
            fmaps[idx] = torch.from_numpy(fmap_arr).permute(2, 0, 1)
            del fmap_arr
        del dem_arr
        # Process cloud cover mask
        cloud_arr = decode_cloud_cover(data, width, height)
        cloud_masks[idx] = (
            torch.from_numpy(cloud_arr).unsqueeze(0)
            if cloud_arr.ndim == 2
            else torch.from_numpy(cloud_arr).permute(2, 0, 1)
        )
        del cloud_arr

        # Process text
        if random.random() < proportion_empty_prompts:
            txts.append(
                tokenizer(
                    "",
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
            )
        else:
            txts.append(
                tokenizer(
                    data["prompt"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
            )

    result = {
        "img": imgs,
        "dem": dems,
        "feature_map": fmaps,
        "cloud_mask": cloud_masks,
        "txt": torch.cat(txts, dim=0),
    }

    del imgs, dems, fmaps, cloud_masks, txts

    return result
