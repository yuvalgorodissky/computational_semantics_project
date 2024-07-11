#!/bin/bash

# Define the Python script location
PYTHON_SCRIPT="/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project/src/run_squad.py"

PREDICTION_FILES=(
    "ACE-whQA/ACE-whQA-has-answer.json"
    "ACE-whQA/ACE-whQA-IDK-competitive.json"
    "ACE-whQA/ACE-whQA-IDK-non-competitive.json"
    "squad_data/valid.json"
    "tydiqa/english_dev_tydiqa_filterd.json"
)
# Directory where output will be saved

OUTPUT_DIR_BASE="/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project/models"
PREDICTION_BASE="/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project/datasets"
# Different configurations for model names or paths
#MODEL_PATHS=(
#    "bert-large-uncased/squad2"
#    "bert-large-uncased/squad_antonyms_ppl"
#    "bert-large-uncased/squad_crqda"
#    "bert-large-uncased/squad_entities_one"
#    "bert-large-uncased/squad_unansq"
#)
MODEL_PATHS=(
    "bert-large-uncased/squad_crqda"
)

# Loop over different models and prediction files
for MODEL_PATH in "${MODEL_PATHS[@]}"; do

    for PREDICTION_FILE in "${PREDICTION_FILES[@]}"; do

        BASE_NAME="$(basename "${PREDICTION_FILE}" .json)"
        OUTPUT_DIR="${OUTPUT_DIR_BASE}/${MODEL_PATH}/${BASE_NAME}"
        MODEL_PATH_1="${OUTPUT_DIR_BASE}/${MODEL_PATH}"
        # Check if output directory exists, if not, create it
        if [ ! -d "$OUTPUT_DIR" ]; then
            mkdir -p "$OUTPUT_DIR"
        fi

        # Running the Python script with different model paths and output directories
        echo "Running model ${MODEL_PATH} with predictions file ${PREDICTION_FILE}"
        python "${PYTHON_SCRIPT}" \
        --model_type "bert" \
        --model_name_or_path "${MODEL_PATH_1}" \
        --do_eval \
        --do_lower_case  \
        --predict_file "$PREDICTION_BASE/$PREDICTION_FILE"  \
        --per_gpu_train_batch_size 12 \
        --learning_rate 5e-5 \
        --num_train_epochs 2.0 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --version_2_with_negative \
        --output_dir "${OUTPUT_DIR}" \
        --overwrite_cache \
        --overwrite_output_dir
        echo "Run completed for model: $MODEL_PATH with predictions file $(basename ${PREDICTION_FILE})"
    done
done
echo "All model runs completed."