#!/bin/bash
#SBATCH --job-name=model-inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gpus=1
#SBATCH --time=23:59:59
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

module purge
module load shared
module load gpu
module load cuda 
module load slurm
eval "$(conda shell.bash hook)"
conda activate $HOME/anaconda3/envs/benchmark_env
echo 'modules loaded'

export OMP_NUM_THREADS=1

DATASETS=(
    neutral_aimnet2_supported
    neutral_others
    charged_aimnet2_supported
    charged_uma_supported
)

for dataset in "${DATASETS[@]}"
do
    echo "Running dataset: ${dataset}"
    python batched_inference.py --dataset_type ${dataset}
done
