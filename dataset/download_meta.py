import os

import daft
import numpy as np
import rasterio
from daft import col
from rasterio.warp import transform

meta_temp = r"D:\DATA\dataset\temp\meta_merged.parquet"
max_no_data = 0.2
# HFI scores: 0 = pristine, 50 = severe human pressure See: https://datadryad.org/dataset/doi:10.5061/dryad.ttdz08m1f
max_hfi = 10.0
hpf_path = r"D:\DATA\dataset\hfi\HFP-100m-2020\Users\jmazzariello\Desktop\hfp\2020\hfp_2020_global.vrt"
meta_s2l2_path = r"D:\DATA\dataset\S2L2A\metadata.parquet"
meta_dem_path = r"D:\DATA\dataset\dem\metadata.parquet"
output = r"D:\DATA\dataset\temp\meta_filtered.parquet"
fraction = 0.04


@daft.udf(return_dtype=daft.DataType.float64())
def get_hfi_score(
    centre_lat: daft.Series, centre_lon: daft.Series, crs: daft.Series
) -> daft.Series:
    lats = np.array(centre_lat.to_pylist())
    lons = np.array(centre_lon.to_pylist())

    with rasterio.open(hpf_path) as src:
        xs, ys = transform(
            "EPSG:4326",
            src.crs,
            lons.tolist(),
            lats.tolist(),
        )

        coords = list(zip(xs, ys))
        samples = np.array(list(src.sample(coords)))
        values = samples[:, 0]

        hfi_scores = np.where(
            (values == 65535) | (values == 64536),
            np.nan,
            np.where(
                values > 50000,
                np.nan,
                values / 1000.0,
            ),
        )

    return daft.Series.from_pylist(hfi_scores.tolist())


def load_meta() -> daft.DataFrame:
    img = daft.read_parquet(meta_s2l2_path)
    dem = daft.read_parquet(meta_dem_path)

    img = img.with_column_renamed("parquet_url", "parquet_url_img")
    img = img.with_column_renamed("parquet_row", "parquet_row_img")

    dem = dem.with_column_renamed("parquet_url", "parquet_url_dem")
    dem = dem.with_column_renamed("parquet_row", "parquet_row_dem")

    merge = img.join(dem, on="grid_cell")

    merge = merge.with_column(
        "hfi_score", get_hfi_score(col("centre_lat"), col("centre_lon"), col("crs"))
    )
    return merge


def filter_meta(meta: daft.DataFrame) -> daft.DataFrame:
    print(f"  Max HFI score: {max_hfi}")
    print(f"  Max nodata: {max_no_data}")
    meta = meta.where(col("hfi_score").between(0, max_hfi))
    meta = meta.where(col("hfi_score").not_null())
    meta = meta.where(col("nodata").between(0.0, max_no_data))

    meta = meta.select(
        "grid_cell",
        "cloud_cover",
        "hfi_score",
        "centre_lat",
        "centre_lon",
        "parquet_url_img",
        "parquet_url_dem",
        "parquet_row_dem",
        "parquet_row_img",
    )
    return meta


# if os.path.exists(meta_temp):
#     meta = daft.read_parquet(meta_temp)
#     meta.write_parquet(meta_temp)
# else:
meta = load_meta()

print(f"\n{'=' * 60}")
print(f"Dataset: {meta.count_rows()} rows")
print(f"{'=' * 60}")
filter = filter_meta(meta)
print(f"\n{'=' * 60}")
print(f"Filtered dataset: {filter.count_rows()} rows")
print(f"{'=' * 60}")
filter = filter.sample(fraction)
print(f"\n{'=' * 60}")
print(f"Sampled dataset: {filter.count_rows()} rows")
print(f"{'=' * 60}")
filter.write_parquet(output, write_mode="overwrite")
# ============================================================
# Sampled dataset: 44462 rows
# ============================================================
