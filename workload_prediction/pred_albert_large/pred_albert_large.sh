#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH -C gpunode
#SBATCH --gres=gpu:1                           # Number of GPUs needed

#SBATCH --output=pred_out_albert_large.log     # Output file
#SBATCH --error=pred_err_albert_large.log      # Error file

. /etc/bashrc
. /etc/profile.d/modules.sh

# Load pyenv
export PYENV_ROOT="/var/scratch/yzg244/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# Set the desired Python version
pyenv global 3.9.13

cd /home/yzg244/pred_albert_large

python pred_albert_large.py