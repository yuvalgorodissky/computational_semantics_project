# Declare associative arrays for model architectures and paths
declare -A models
models["bert-large-uncased"]="encoder"
models["flan-t5-base"]="encoder_decoder"

declare -A model_paths
model_paths["bert-large-uncased"]="google-bert/bert-large-uncased"
model_paths["flan-t5-base"]="google/flan-t5-base"

# Define an array of datasets
datasets=("squad2" "squad_antonyms_ppl" "squad_crqda" "squad_entities_one" "squad_unansq")

# Iterate over models and datasets
for model in "${!models[@]}"; do
    model_path="${model_paths[$model]}"
    for dataset in "${datasets[@]}"; do
        echo "Training $model on $dataset with model path $model_path and architecture ${models[$model]}"
        # Pass the model name, model architecture, dataset, and model path to the script
        sbatch training_models.sh "$model" "${models[$model]}" "$dataset" "$model_path"
    done
done