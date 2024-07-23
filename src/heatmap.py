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
    plt.figure(figsize=(24, 16))

    ax = sns.heatmap(data, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100, annot_kws={"size": 24})
    plt.title(title, fontsize=30)
    plt.xticks(rotation=15 , fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.ylabel('Train Dataset', fontsize=24)
    plt.xlabel('Validation Dataset', fontsize=24)
    # Customize color bar tick labels
    cbar = ax.collections[0].colorbar
    cbar.set_label('Score', size=20)  # Set font size for the color bar label
    cbar.ax.tick_params(labelsize=18)  # Set font size
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
model_name = "bert-large-uncased"

# Path to the directory containing CSV files
directory_path = '/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project/eval/'


def transform_df_columns_row(df):
    # Dictionary to map old column names to new names
    cols_names = {
        "valid": "SQuAD2.0 (dev)",
        'english_dev_tydiqa_filterd': "TyDi QA (English)",
        "ACE-whQA-IDK-competitive": "ACE WhQA IDK Competitive",
        "ACE-whQA-IDK-non-competitive": "ACE WhQA IDK Non Competitive",
        "ACE-whQA-has-answer": "ACE WhQA Has Answer",
    }
    # Dictionary to map old row indices to new indices
    rows_names = {
        "squad2": "SQuAD2.0",
        "squad_crqda": "SQuAD CRQDA",
        "squad_unansq": "SQuAD UNANSQ",
        "squad_antonyms_ppl": "SQuAD Antonym",
        "squad_entities_one": "SQuAD Entity"
    }

    # Rename columns and rows
    df = df.rename(columns=cols_names, index=rows_names)

    # Reorder columns and rows to match the order specified in the dictionaries
    df = df[cols_names.values()]  # Reorder columns
    df = df.loc[rows_names.values()]  # Reorder rows

    return df


# Read each CSV file, process it, and generate the heatmap
for file_name, title in csv_files:
    file_path = f"{directory_path}/{model_name}/tables/{file_name}"
    data = read_and_process_csv(file_path)
    data = transform_df_columns_row(data)
    generate_heatmap(data, f"{title}\n{model_name}")
