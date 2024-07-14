import os
import json
import pandas as pd

base_path = "/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project"
model_name = "bert-large-uncased"
model_path = f"{base_path}/models/{model_name}"

train_datasets = ['squad2', 'squad_crqda', 'squad_unansq', 'squad_antonyms_ppl', 'squad_entities_one']
val_datasets = ['valid', 'english_dev_tydiqa_filterd', 'ACE-whQA-IDK-competitive', 'ACE-whQA-IDK-non-competitive', 'ACE-whQA-has-answer']
metrics = ["exact_match", "f1", "exact_match_has_ans", "f1_has_ans", "exact_match_no_ans", "f1_no_ans"]
# Dictionary to store the data
data = {train_dataset: {} for train_dataset in train_datasets}
for metric in metrics:
    # Loop through the files and extract the metrics
    for train_dataset in train_datasets:
        for val_dataset in val_datasets:
            with open(f"{model_path}/{train_dataset}/{val_dataset}/metrics.json", 'r') as f:
                content = json.load(f).get(metric,0)

                data[train_dataset][val_dataset] = content

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')

    ## Create a dictionary to store the data
    os.makedirs(f"{base_path}/eval/{model_name}/tables", exist_ok=True)
    # Save the DataFrame to an Excel file with conditional formatting
    csv_filename = f"{base_path}/eval/{model_name}/tables/metrics_{metric}.csv"
    df.to_csv(csv_filename)
    print(f"Saved {csv_filename}")
