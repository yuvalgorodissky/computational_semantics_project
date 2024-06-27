#!/bin/bash

# Define the Python script location
PYTHON_SCRIPT="/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project/src/encoder_decoder.py"

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
    "flan-t5-base/squad2"
    "flan-t5-base/squad_antonyms_ppl"
    "flan-t5-base/squad_crqda"
    "flan-t5-base/squad_entities_one"
    "flan-t5-base/squad_unansq"
)

# Loop over different models and prediction files
for MODEL_PATH in "${MODEL_PATHS[@]}"; do

    for PREDICTION_FILE in "${PREDICTION_FILES[@]}"; do

        OUTPUT_DIR="${OUTPUT_DIR_BASE}/${MODEL_PATH}/$(basename ${PREDICTION_FILE})"
        MODEL_PATH_1="${OUTPUT_DIR_BASE}/${MODEL_PATH}"
        # Check if output directory exists, if not, create it
        if [ ! -d "$OUTPUT_DIR" ]; then
            mkdir -p "$OUTPUT_DIR"
        fi

        # Running the Python script with different model paths and output directories
        echo "Running model ${MODEL_PATH} with predictions file ${PREDICTION_FILE}"
        python $PYTHON_SCRIPT --model_name_or_path $MODEL_PATH_1 \
                              --output_dir $OUTPUT_DIR \
                              --do_eval \
                              --path_dev_set "$PREDICTION_BASE/$PREDICTION_FILE" \
                              --batch_size_dev 8 \
                              --seed 42

        echo "Run completed for model: $MODEL_PATH with predictions file $(basename ${PREDICTION_FILE})"
    done
done
echo "All model runs completed."