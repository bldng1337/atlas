import daft
from tqdm import tqdm
from daft.io import IOConfig, HuggingFaceConfig
from time import sleep
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
from fsspec.parquet import open_parquet_file
import urllib.request
import os

meta = daft.read_parquet("meta.parquet")

print(meta.count_rows())
