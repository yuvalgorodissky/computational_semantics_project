#!/bin/bash


################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 7-00:00:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name models_training			### name of the job
#SBATCH --output job-%J-$x.out			### output log for running job - %J for job number
#SBATCH --gpus=rtx_6000:1			### number of GPUs, allocating more than 1 requires IT team's permission. Example to request 3090 gpu: #SBATCH --gpus=rtx_3090:1

# Note: the following 4 lines are commented out
##SBATCH --mail-user=user@post.bgu.ac.il	### user's email for sending job status messages
##SBATCH --mail-type=ALL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
##SBATCH --mem=60G				### ammount of RAM memory, allocating more than 60G requires IT team's permission

################  Following lines will be executed by a compute node    #######################

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate 	/sise/eliorsu-group/yuvalgor/.conda/yuvalgor_env/			### activate a conda environment, replace my_env with your conda environment





# Assign positional parameters to variables
model_name="$1"
model_arch="$2"
dataset="$3"
model_path="$4"

base_path="/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project"
# Define the output path for model training
output_path="$base_path/models/$model_name/$dataset"

# Create the output directory and any necessary parent directories
mkdir -p "$output_path"

# Execute the Python script with the correct continuation of lines
python "$base_path/src/$python_file_path/$model_arch.py" \
  --epochs 10 \
  --lr 3e-5 \
  --batch_size_train 16 \
  --batch_size_dev 16 \
  --path_dev_set "/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project/datasets/squad_data/valid.json" \
  --path_train_set "/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project/datasets/squad_data/$dataset.json" \
  --do_train \
  --do_eval \
  --output_dir "$output_path" \
  --model_name_or_path "$model_path"
