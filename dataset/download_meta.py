import daft

img = daft.read_parquet("hf://datasets/Major-TOM/Core-S2L2A/metadata.parquet")
dem = daft.read_parquet("hf://datasets/Major-TOM/Core-DEM/metadata.parquet")

img = img.with_column_renamed("parquet_url", "parquet_url_img")
img = img.with_column_renamed("parquet_row", "parquet_row_img")

dem = dem.with_column_renamed("parquet_url", "parquet_url_dem")
dem = dem.with_column_renamed("parquet_row", "parquet_row_dem")
merge = img.join(dem, on="grid_cell")
merge = merge.where("nodata < 0.1")

merge = merge.sample(fraction=0.005)

merge = merge.select(
    "grid_cell",
    "cloud_cover",
    "parquet_url_img",
    "parquet_url_dem",
    "parquet_row_dem",
    "parquet_row_img",
)
print(f"Got {merge.count_rows()} rows")
merge.write_parquet("meta.parquet")
