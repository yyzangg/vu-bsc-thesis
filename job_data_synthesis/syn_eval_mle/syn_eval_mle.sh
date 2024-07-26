#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH -N 1

#SBATCH --output=syn_eval_mle_out.log           # Output file
#SBATCH --error=syn_eval_mle_err.log            # Error file

. /etc/bashrc
. /etc/profile.d/modules.sh

export PYENV_ROOT="/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

pyenv global 3.9.13

cd /syn_eval_mle

python syn_eval_mle.py
