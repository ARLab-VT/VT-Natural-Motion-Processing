#!/bin/bash
 
#### Resource Request: ####
# Cascades has the following hardware:
#   a. 190 32-core, 128 GB Intel Broadwell nodes
#   b.   4 32-core, 512 GB Intel Broadwell nodes with 2 Nvidia K80 GPU
#   c.   2 72-core,   3 TB Intel Broadwell nodes
#   d.  39 24-core, 376 GB Intel Skylake nodes with 2 Nvidia V100 GPU
#
# Resources can be requested by specifying the number of nodes, cores, memory, GPUs, etc
#SBATCH --nodes=1 #(implies --ntasks=1 unless otherwise specified)
#SBATCH --gres=gpu:1
#SBATCH --mem=256G

#### Walltime #### 
#SBATCH -t 72:00:00

#### Queue ####
# Queue name. Cascades has five queues:
#   normal_q        for production jobs on all Broadwell nodes
#   largemem_q      for jobs on the two 3TB, 60-core Ivy Bridge servers
#   dev_q           for development/debugging jobs. These jobs must be short but can be large.
#   v100_normal_q   for production jobs on Skylake/V100 nodes
#   v100_dev_q      for development/debugging jobs on Skylake/V100 nodes
#SBATCH -p v100_normal_q

#### Account ####
#SBATCH -A arc-train1

#### Module Loading ####
module purge
module load intel mvapich2
module load Anaconda

source activate motion-prediction

# Run the training shell script

./build_position_inference_dataset.sh

exit;


