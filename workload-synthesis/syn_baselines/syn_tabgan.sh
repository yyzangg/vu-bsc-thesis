#!/bin/bash
#SBATCH --time=00:15:00                    # FixMe
#SBATCH -N 1
#SBATCH -C gpunode                         # FixMe
#SBATCH --gres=gpu:1                       # Number of GPUs needed

#SBATCH --output=tabgan_out.log             # Output file
#SBATCH --error=tabgan_err.log              # Error file

. /etc/bashrc
. /etc/profile.d/modules.sh

# Load pyenv
export PYENV_ROOT="/var/scratch/yzg244/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# Set the desired Python version
pyenv global 3.9.13

# Load required modules
module load cuda11.1/toolkit
module load cuDNN/cuda11.1

cd /home/yzg244/syn_baselines

# Run the Python script
echo "Running syn_tabgan.py..."
python syn_tabgan.py

echo "Script execution completed."
