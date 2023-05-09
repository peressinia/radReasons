#!/bin/bash

#SBATCH --job-name="batch-radBERT-epoch6-batch32-ml64-dV7_v2-1"
#SBATCH --partition=batch
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=radBERT-epoch6-batch32-ml64-dV7_v2-1_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anthony.peressini@marquette.edu

export SLURM_SUBMIT_DIR=/mmfs1/home/peressinia/mimic
cd $SLURM_SUBMIT_DIR
source ./env/bin/activate
. lmpy							# load pytorch module: "module load pytorch-py37-cuda10.2-gcc/"
python ./model-BERT-v2-2.py