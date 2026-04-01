#!/bin/bash

#SBATCH --job-name=openvsp-scripts       # Job name
#SBATCH --time=01:00:00          # Time limit: dd-hh:mm
#SBATCH --nodes=1                # Number of nodes (1 max)
#SBATCH --ntasks-per-node=1      # Number of tasks per node (1 max)
#SBATCH --cpus-per-task=20       # Specify cores per node (64 max)
#SBATCH --mem-per-cpu=4G         # Specify memory per node (0 max)
#SBATCH --output=openvsp-scripts.log     # Output log file

python3 vsp_optimization_hpc.py
