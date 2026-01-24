import daft

data_path = "D:\\DATA\\dataset\\atlas\\data_with_features.parquet"
prompt_path = "C:\\Users\\bldng\\DEV\\atlas\\weights\\prompts.csv"
output = "D:\\DATA\\dataset\\atlas\\final_data.parquet"

data = daft.read_parquet(data_path)
prompt = daft.read_csv(prompt_path)
print(data.column_names)
print(prompt.column_names)

merge = prompt.join(data, on="grid_cell")


merge.write_parquet(output)