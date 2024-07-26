#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH -C gpunode
#SBATCH --gres=gpu:1

#SBATCH --output=syn_rltf_out.log          # Output file
#SBATCH --error=syn_rltf_err.log           # Error file

. /etc/bashrc
. /etc/profile.d/modules.sh

export PYENV_ROOT="/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

pyenv global 3.9.13

module load cuda11.1/toolkit
module load cuDNN/cuda11.1

cd /syn_rltf

python syn_rltf.py
