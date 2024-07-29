#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH -C gpunode
#SBATCH --gres=gpu:1

#SBATCH --output=pred_out_bert_large.log       # Output file
#SBATCH --error=pred_err_bert_large.log        # Error file

. /etc/bashrc
. /etc/profile.d/modules.sh

export PYENV_ROOT="/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

pyenv global 3.9.13

cd /pred_bert_large

python pred_bert_large.py
