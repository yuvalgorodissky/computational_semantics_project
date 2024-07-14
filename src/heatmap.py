import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to read CSV and set the first column as index
def read_and_process_csv(file_path):
    data = pd.read_csv(file_path)
    data.set_index(data.columns[0], inplace=True)
    return data

# Function to generate heatmap
def generate_heatmap(data, title):
   ## Create a figure with figsize
    plt.figure(figsize=(16, 12))

    sns.heatmap(data, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Score'})
    plt.title(title, fontsize=14)
    plt.xticks(rotation=22)
    plt.yticks(rotation=0)
    plt.ylabel('Train Dataset')
    plt.xlabel('Validation Dataset')
    ## Save the heatmap as an image
    os.makedirs(f"{directory_path}/{model_name}/heatmaps", exist_ok=True)
    plt.savefig(f"{directory_path}/{model_name}/heatmaps/{title}.png")

# List of CSV files and their titles
csv_files = [
    ('metrics_average_exact_match.csv', "Average Exact Match"),
    ('metrics_average_f1.csv', "Average F1"),
    ('metrics_exact_match_has_ans.csv', "Exact Match (Has Answer)"),
    ('metrics_exact_match_no_ans.csv', "Exact Match (No Answer)"),
    ('metrics_f1_has_ans.csv', "F1 Score (Has Answer)"),
    ('metrics_f1_no_ans.csv', "F1 Score (No Answer)")
]
model_name = "Meta-Llama-3-8B-Instruct"


# Path to the directory containing CSV files
directory_path = '/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project/eval/'

# Read each CSV file, process it, and generate the heatmap
for file_name, title in csv_files:
    file_path = f"{directory_path}/{model_name}/tables/{file_name}"
    data = read_and_process_csv(file_path)
    generate_heatmap(data, f"{title}\n{model_name}")
