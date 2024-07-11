#!/bin/bash


################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 7-00:00:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name models_training			### name of the job
#SBATCH --output /sise/eliorsu-group/yuvalgor/courses/computational_semantics_project/jobs_out/decoder_eval_job-%J.out			### output log for running job - %J for job number
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
source activate /sise/eliorsu-group/yuvalgor/.conda/yuvalgor_env/		### activate a conda environment, replace my_env with your conda environment

model_path="$1"
dev_set="$2"
output_dir="$3"

# Create the output directory and any necessary parent directories
mkdir -p "$output_dir"

# Execute the Python script with the correct continuation of lines
python "/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project/src/decoder.py" \
        --model_name_or_path $model_path \
        --output_dir $output_dir \
        --do_eval \
        --path_dev_set "$dev_set" \
        --batch_size_dev 4 \
        --seed 42
