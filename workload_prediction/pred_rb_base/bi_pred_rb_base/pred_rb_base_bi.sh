#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH -C gpunode
#SBATCH --gres=gpu:1                       # Number of GPUs needed

#SBATCH --output=pred_out_rb_base_bi.log   # Output file
#SBATCH --error=pred_err_rb_base_bi.log    # Error file

. /etc/bashrc
. /etc/profile.d/modules.sh

export PYENV_ROOT="/var/scratch/yzg244/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

pyenv global 3.9.13

cd /home/yzg244/pred_rb_base

python pred_rb_base_bi.py
