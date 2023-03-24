#!/bin/bash
#SBATCH --job-name="RelaNet++"
#SBATCH --partition=v100-32g
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --time=3-0:0
#SBATCH --chdir=./
#SBATCH --output=./log/eval_out.txt
#SBATCH --error=./log/eval_err.txt
echo
echo "============================ Messages from Goddess============================"
echo " * Job starting from: "`date`
echo " * Job ID : "$SLURM_JOBID
echo " * Job name : "$SLURM_JOB_NAME
echo " * Job partition : "$SLURM_JOB_PARTITION
echo " * Nodes : "$SLURM_JOB_NUM_NODES
echo " * Cores : "$SLURM_NTASKS
echo " * Working directory: "${SLURM_SUBMIT_DIR/$HOME/"~"}
echo "==============================================================================="
echo

source ~/courses/DLCV/vitEnv/bin/activate
python3 detectron_test.py

echo
echo "============================ Messages from Goddess============================"
echo " * Jab ended at : "`date`
echo "==============================================================================="              
