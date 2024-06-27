import os
import json
import pandas as pd

base_path = "/Users/noamazulay/Desktop/studies/fourth year/semester b/computational_semantics_project"
model_name = "flan-t5-base"
model_path = f"{base_path}/models/{model_name}"
# Get all the files in the model path
train_datasets = os.listdir(model_path)
# Sort the datasets by length
train_datasets.sort(key=len)
val_datasets = os.listdir(f"{model_path}/{train_datasets[0]}")

val_datasets=[x[:-5] for x in val_datasets]
## rename the datasets
val_datasets = [x.replace('valid','squad2-devset') for x in val_datasets]
val_datasets = [x.replace('english_dev_tydiqa_filterd', ' tydiqa_devset ') for x in val_datasets]
val_datasets.sort(key=len)

metrics = ["average_exact_match", "average_f1", "exact_match_has_ans", "f1_has_ans", "exact_match_no_ans", "f1_no_ans"]
# Dictionary to store the data
data = {train_dataset: {} for train_dataset in train_datasets}
for metric in metrics:
    # Loop through the files and extract the metrics
    for train_dataset in train_datasets:
        for val_dataset in val_datasets:
            val_dataset_name= val_dataset
            if val_dataset == "squad2-devset":
                val_dataset_name = 'valid'
            if val_dataset == ' tydiqa_devset ':
                val_dataset_name = 'english_dev_tydiqa_filterd'
            with open(f"{model_path}/{train_dataset}/{val_dataset_name}.json/metrics.json", 'r') as f:
                content = json.load(f)[metric]

                data[train_dataset][val_dataset] = content

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')

    ## Create a dictionary to store the data
    os.makedirs(f"{base_path}/eval/{model_name}/tables", exist_ok=True)
    # Save the DataFrame to an Excel file with conditional formatting
    csv_filename = f"{base_path}/eval/{model_name}/tables/metrics_{metric}.csv"
    df.to_csv(csv_filename)
    print(f"Saved {csv_filename}")
