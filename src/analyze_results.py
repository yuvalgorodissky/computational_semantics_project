import os
import json
import pandas as pd


model_path = "/Users/noamazulay/Desktop/studies/fourth year/semester b/computational_semantics_project/models/flan-t5-base"
##get all the files in the model path
datasets = os.listdir(model_path)

# Dictionary to store the data
data = {}

for dataset in datasets:
    with open(f"{model_path}/{dataset}/valid.json/metrics.json", 'r') as f:
        content = json.load(f)
        data[dataset] = content

# Convert the dictionary to a DataFrame
df = pd.DataFrame.from_dict(data, orient='index')

# Save the DataFrame to a CSV file
csv_filename = "metrics.csv"
df.to_csv(csv_filename, index_label="filename")

print(f"CSV file '{csv_filename}' has been created successfully.")
