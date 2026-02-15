from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from feature_map import get_map_combined
from rasterio.io import MemoryFile
from tqdm import tqdm

input_path = "D:/DATA/dataset/temp/final_data.parquet"
output_path = "D:\\DATA\\dataset\\atlas\\final.parquet"
prompt_path = "C:\\Users\\bldng\\DEV\\atlas\\weights\\prompts.csv"


def get_dem(dem_bytes):
    with MemoryFile(dem_bytes) as mem_f:
        with mem_f.open(driver="GTiff") as f:
            dem = f.read()
    dem = dem.reshape(356, 356)
    return dem


def get_combined(dem):
    combined, flat_regions = get_map_combined(
        dem,
    )
    return combined


def is_valid_dem(dem):
    """Check if DEM has valid data (not empty and has variation)."""
    dem_max = np.max(dem)
    dem_min = np.min(dem)

    # Filter out if max is 0 (empty DEM)
    if dem_max == 0:
        return False

    # Filter out if only contains 0 and max values (no variation)
    unique_values = np.unique(dem)
    if len(unique_values) <= 2 and (0 in unique_values or dem_min == dem_max):
        # Check if all values are either 0 or the max value
        if set(unique_values).issubset({0, dem_max}):
            return False
    if len(unique_values) <= 10:
        return False
    return True


def process_batch(batch, prompts_dict):
    """Process a batch of rows, filter invalid DEMs, and add feature_map and prompt columns."""
    feature_maps = []
    prompts = []
    valid_indices = []
    dem_column = batch.column("DEM")
    grid_cell_column = batch.column("grid_cell")

    for i in range(len(batch)):
        dem_bytes = dem_column[i].as_py()
        if isinstance(dem_bytes, list):
            dem_bytes = dem_bytes[0]

        dem = get_dem(dem_bytes)

        # Skip invalid DEMs
        if not is_valid_dem(dem):
            continue

        # Get grid_cell and look up prompt
        grid_cell = grid_cell_column[i].as_py()
        if isinstance(grid_cell, list):
            grid_cell = grid_cell[0]

        # Skip if no prompt found for this grid_cell
        if grid_cell not in prompts_dict:
            continue

        # NDArray[float64]
        combined = get_combined(dem)
        # Convert to bytes for storage
        feature_maps.append(combined.tobytes())
        prompts.append(prompts_dict[grid_cell])
        valid_indices.append(i)

    # If no valid rows, return None
    if not valid_indices:
        return None

    # Filter batch to only valid rows
    filtered_batch = batch.take(valid_indices)

    # Add feature_map and prompt columns to filtered batch
    feature_map_column = pa.array(feature_maps, type=pa.binary())
    prompt_column = pa.array(prompts, type=pa.string())
    filtered_batch = filtered_batch.append_column("feature_map", feature_map_column)
    return filtered_batch.append_column("prompt", prompt_column)


def process_parquet_streaming():
    """Process parquet files in streaming fashion to avoid OOM."""
    # Load prompts CSV into dictionary
    print("Loading prompts...")
    prompts_df = pd.read_csv(prompt_path)
    prompts_dict = dict(zip(prompts_df["grid_cell"], prompts_df["prompt"]))
    print(f"Loaded {len(prompts_dict)} prompts")

    input_dir = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all parquet files in the input directory
    parquet_files = sorted(input_dir.glob("*.parquet"))

    for file_path in tqdm(parquet_files, desc="Processing parquet files"):
        parquet_file = pq.ParquetFile(file_path)
        output_file = output_dir / file_path.name
        if output_file.exists():
            print(f"Skipping {file_path.name} - already exists")
            continue
        writer = None

        try:
            # Process in batches using row groups
            for batch_idx in tqdm(
                range(parquet_file.metadata.num_row_groups),
                desc=f"Processing {file_path.name}",
                leave=False,
            ):
                for itry in range(3):
                    # Read one row group at a time
                    table = parquet_file.read_row_group(batch_idx)

                    # Process the batch
                    try:
                        processed_table = process_batch(table, prompts_dict)
                    except Exception as e:
                        print(
                            f"Error processing batch {batch_idx} in {file_path.name}: {e}"
                        )
                        continue
                    # Skip if all rows were filtered out
                    if processed_table is None:
                        break

                    # Initialize writer with schema from first batch
                    if writer is None:
                        writer = pq.ParquetWriter(output_file, processed_table.schema)

                    writer.write_table(processed_table)
        finally:
            if writer is not None:
                writer.close()

    print(f"Processing complete. Output saved to: {output_path}")


if __name__ == "__main__":
    process_parquet_streaming()
