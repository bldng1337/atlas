import daft

meta = daft.read_parquet("meta.parquet")
img = daft.read_parquet("temp/img/*/**")
dem = daft.read_parquet("temp/dem/*/**")
img = img.with_column_renamed("thumbnail", "img_thumb")
dem = dem.with_column_renamed("thumbnail", "dem_thumb")

meta = meta.join(img, "grid_cell").join(dem, "grid_cell")

output = "D:\\DATA\\dataset\\atlas\\data.parquet"

meta.write_parquet(output)
