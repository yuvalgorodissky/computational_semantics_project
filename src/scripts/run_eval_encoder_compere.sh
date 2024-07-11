#!/bin/bash

# Define the Python script location
PYTHON_SCRIPT="/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project/src/evaluation_script.py"

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
MODEL_PATHS=(
    "bert-large-uncased/squad2"
    "bert-large-uncased/squad_antonyms_ppl"
    "bert-large-uncased/squad_crqda"
    "bert-large-uncased/squad_entities_one"
    "bert-large-uncased/squad_unansq"
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
        "$PREDICTION_BASE/$PREDICTION_FILE" \
        "${OUTPUT_DIR}/predictions_.json" \
        "${OUTPUT_DIR}/nbest_predictions_.json" \
        --out-file "${OUTPUT_DIR}/metrics.json" \
        --na-prob-file "${OUTPUT_DIR}/null_odds_.json" \
        --out-image-dir "${OUTPUT_DIR}/images" \
        --out-f1 "${OUTPUT_DIR}/f1_score.json"
        echo "Run evaluation_script for model: $MODEL_PATH with predictions file $(basename ${PREDICTION_FILE})"
    done
done
echo "All model runs completed."