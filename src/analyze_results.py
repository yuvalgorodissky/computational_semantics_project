import os
import json
import pandas as pd

base_path = "/Users/noamazulay/Desktop/studies/fourth year/semester b/computational_semantics_project"
model_path = f"{base_path}/models/flan-t5-base"
# Get all the files in the model path
train_datasets = os.listdir(model_path)
# Sort the datasets by length
train_datasets.sort(key=len)
val_datasets = os.listdir(f"{model_path}/{train_datasets[0]}")
metrics = ["average_exact_match", "average_f1", "exact_match_has_ans", "f1_has_ans", "exact_match_no_ans", "f1_no_ans"]

# Dictionary to store the data
data = {train_dataset: {} for train_dataset in train_datasets}
for metric in metrics:
    # Loop through the files and extract the metrics
    for train_dataset in train_datasets:
        for val_dataset in val_datasets:
            with open(f"{model_path}/{train_dataset}/{val_dataset}/metrics.json", 'r') as f:
                content = json.load(f)[metric]
                data[train_dataset][val_dataset] = content

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')

    # Save the DataFrame to an Excel file with conditional formatting
    excel_filename = f"{base_path}/eval/metrics_{metric}.xlsx"
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        # Define the conditional formatting rule
        for col_num in range(1, df.shape[1] + 1):  # Starting from 1 to skip index column
            color_scale_rule = {
                'type': '2_color_scale',
                'min_color': "#FF9999",  # Red
                'max_color': "#99CCFF"  # Blue
            }
            # Apply the conditional formatting to each column individually
            worksheet.conditional_format(1, col_num, df.shape[0], col_num, color_scale_rule)

    print(f"Excel file '{excel_filename}' with conditional formatting has been created successfully.")
