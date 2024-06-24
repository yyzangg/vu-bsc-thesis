#!/bin/bash
#SBATCH --time=00:15:00                    # FixMe
#SBATCH -N 1
#SBATCH -C TitanX                          # FixMe
#SBATCH --gres=gpu:1                       # Number of GPUs needed

#SBATCH --output=out_pred_gpt.log          # Output file
#SBATCH --error=err_pred_gpt.log           # Error file

# echo "Loading environment modules..."
. /etc/bashrc
. /etc/profile.d/modules.sh

# Load pyenv
export PYENV_ROOT="/var/scratch/yzg244/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# Set the desired Python version
pyenv global 3.9.13

cd /home/yzg244/pred_gpt

# Run the Python script
echo "Running pred_gpt.py..."
python pred_gpt.py

echo "Script execution completed."
