#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH -C gpunode
#SBATCH --gres=gpu:1                       # Number of GPUs needed

#SBATCH --output=out_lr.log                # Output file
#SBATCH --error=err_lr.log                 # Error file

. /etc/bashrc
. /etc/profile.d/modules.sh

# Load pyenv
export PYENV_ROOT="/var/scratch/yzg244/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# Set the desired Python version
pyenv global 3.9.13

cd /home/yzg244/pred_baselines

# Run the Python script
python pred_lr.py
