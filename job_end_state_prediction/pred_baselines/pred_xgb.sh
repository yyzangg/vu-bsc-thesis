#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH -N 1

#SBATCH --output=xgb_out.log               # Output file
#SBATCH --error=xgb_err.log                # Error file

. /etc/bashrc
. /etc/profile.d/modules.sh

export PYENV_ROOT="/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

pyenv global 3.9.13

cd /pred_baselines

python pred_xgb.py
