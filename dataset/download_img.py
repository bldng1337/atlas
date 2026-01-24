import shutil
import daft
from tqdm import tqdm
from time import sleep
import os
import fsspec
import pyarrow.parquet as pq
from fsspec.parquet import open_parquet_file

# We only do very little work in daft so the progress bars only flash and dont really show anything
os.environ["DAFT_PROGRESS_BAR"] = "0"


def download(
    grid_cells: list[str], url: str, to_extract: list[str], row_group: list[int]
):
    tries = 0
    while True:
        try:
            file = open_parquet_file(
                url,
                columns=to_extract,
                row_groups=row_group,
                engine="pyarrow",
                footer_sample_size=3_000_000,
            )
            # with fsspec.open(url, mode="rb") as file:
            with pq.ParquetFile(file) as pf:
                table = pf.read_row_groups(row_group, columns=to_extract)
            df = daft.from_arrow(table)
            df.where(df["grid_cell"].is_in(grid_cells))
            return df.collect()
        except Exception as e:
            print(f"Failed request with {e}")
            tries += 1
            if tries > 5:
                raise e
            seconds = 5.0 ** (tries / 2.0)
            tqdm.write(f"Waiting {seconds}s for ratelimit...")
            sleep(seconds)


meta = daft.read_parquet("meta.parquet")
print(f"Downloading Images for {meta.count_rows()} rows")
imgdata = (
    meta.select("parquet_url_img", "parquet_row_img", "grid_cell")
    .groupby("parquet_url_img")
    .agg_list()
    .to_pylist()
)

demdata = (
    meta.select("parquet_url_dem", "parquet_row_dem", "grid_cell")
    .groupby("parquet_url_dem")
    .agg_list()
    .to_pylist()
)

demdata = [
    dem
    for dem in demdata
    if not os.path.exists(
        f"temp/dem/{dem['parquet_url_dem'].split('/')[-1].split('.')[0]}.parquet"
    )
]

for i, row in tqdm(enumerate(demdata), "Downloading DEM Data", total=len(demdata)):
    url = row["parquet_url_dem"]
    name = f"temp/dem/{url.split('/')[-1].split('.')[0]}.parquet"
    if os.path.exists(name):
        tqdm.write(f"Skipping {name}")
        continue
    table = download(
        row["grid_cell"],
        url,
        ["grid_cell", "thumbnail", "DEM", "compressed"],
        row["parquet_row_dem"],
    )
    tqdm.write(f"Writing {table.count_rows()}")
    table.write_parquet(name)


imgdata = [
    img
    for img in imgdata
    if not os.path.exists(
        f"temp/img/{img['parquet_url_img'].split('/')[-1].split('.')[0]}.parquet"
    )
]

for i, row in tqdm(enumerate(imgdata), "Downloading Image Data", total=len(imgdata)):
    url = row["parquet_url_img"]
    name = f"temp/img/{url.split('/')[-1].split('.')[0]}.parquet"
    if os.path.exists(name):
        tqdm.write(f"Skipping {name}")
        continue
    table = download(
        row["grid_cell"],
        url,
        ["grid_cell", "thumbnail", "B04", "B03", "B02", "cloud_mask"],
        row["parquet_row_img"],
    )
    table.write_parquet(name)

