#!/bin/bash



MODEL_NAME="Meta-Llama-3-8B-Instruct"
BASE_PATH="/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project/models/${MODEL_NAME}"
SH_SCRIPT="/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project/src/scripts/run_eval_decoder_sbatch.sh"
# Define an array of datasets
datasets=("squad2" "squad_antonyms_ppl" "squad_crqda" "squad_entities_one" "squad_unansq")

PREDICTION_FILES=(
    "ACE-whQA/ACE-whQA-has-answer.json"
    "ACE-whQA/ACE-whQA-IDK-competitive.json"
    "ACE-whQA/ACE-whQA-IDK-non-competitive.json"
    "squad_data/valid.json"
    "tydiqa/english_dev_tydiqa_filterd.json"
)
PREDICTION_BASE="/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project/datasets"

for dataset in "${datasets[@]}"; do
    model_path="${BASE_PATH}/${dataset}"
    for prediction_file in "${PREDICTION_FILES[@]}"; do
        dev_set="${PREDICTION_BASE}/${prediction_file}"
        file_name=$(basename "$prediction_file" .json)
        output_dir="${model_path}/${file_name}"
        echo "Running evaluation for ${model_path} on ${dev_set} with output to ${output_dir}"

        # Pass the model name, model architecture, dataset, and model path to the script
        sbatch $SH_SCRIPT $model_path $dev_set $output_dir

    done

done
