import os
import json
import pandas as pd

model_path = "/Users/noamazulay/Desktop/studies/fourth year/semester b/computational_semantics_project/models/flan-t5-base"

# Get all the files in the model path
train_datasets = os.listdir(model_path)
val_datasets = os.listdir(f"{model_path}/{train_datasets[0]}")

metric = 'average_f1'
# Dictionary to store the data
data = {train_dataset: {} for train_dataset in train_datasets}

for train_dataset in train_datasets:
    for val_dataset in val_datasets:
        with open(f"{model_path}/{train_dataset}/{val_dataset}/metrics.json", 'r') as f:
            content = json.load(f)[metric]
            data[train_dataset][val_dataset] = content

# Convert the dictionary to a DataFrame
df = pd.DataFrame.from_dict(data, orient='index')

# Save the DataFrame to a CSV file
csv_filename = f"metrics_{metric}.csv"
df.to_csv(csv_filename, index_label="train_dataset")

print(f"CSV file '{csv_filename}' has been created successfully.")
