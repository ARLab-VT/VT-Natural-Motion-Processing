#!/bin/bash

#SBATCH --nodes=1 #(implies --ntasks=1 unless otherwise specified)
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH -t 72:00:00
#SBATCH -p normal_q

#SBATCH -A MotionPred

module purge
module load gcc cuda Anaconda3 jdk

source activate powerai16_ibm

./train_position_inference.sh

exit;


