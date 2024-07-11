# Declare associative arrays for model architectures and paths
declare -A models
models["bert-large-uncased"]="encoder"
#models["flan-t5-base"]="encoder_decoder"
#models["Meta-Llama-3-8B-Instruct"]="decoder"

declare -A model_paths
model_paths["bert-large-uncased"]="google-bert/bert-large-uncased"
#model_paths["flan-t5-base"]="google/flan-t5-base"
#model_paths["Meta-Llama-3-8B-Instruct"]="meta-llama/meta-llama-3-8B-Instruct"

declare -A model_scripts
model_scripts["bert-large-uncased"]="/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project/src/scripts/run_encoder_train.sh"
#model_scripts["flan-t5-base"]="training_models.sh"
#model_scripts["Meta-Llama-3-8B-Instruct"]="/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project/src/scripts/training_models.sh"

# Define an array of datasets
datasets=("squad2" "squad_antonyms_ppl" "squad_crqda" "squad_entities_one" "squad_unansq")
#datasets=("squad2")
# Iterate over models and datasets
for model in "${!models[@]}"; do
    model_path="${model_paths[$model]}"
    for dataset in "${datasets[@]}"; do
        echo "Training $model on $dataset with model path $model_path and architecture ${models[$model]}"
        # Pass the model name, model architecture, dataset, and model path to the script
        sbatch "${model_scripts[$model]}" "$model" "${models[$model]}" "$dataset" "$model_path"
    done
done