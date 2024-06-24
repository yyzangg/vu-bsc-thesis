#!/bin/bash
#SBATCH --time=00:15:00                    # FixMe
#SBATCH -N 1
#SBATCH -C gpunode                         # FixMe
#SBATCH --gres=gpu:1                       # Number of GPUs needed

#SBATCH --output=pred_out_bert_large.log          # Output file
#SBATCH --error=pred_err_bert_large.log           # Error file

. /etc/bashrc
. /etc/profile.d/modules.sh

# Load pyenv
export PYENV_ROOT="/var/scratch/yzg244/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# Set the desired Python version
pyenv global 3.9.13

cd /home/yzg244/pred_bert_large

# Run the Python script
echo "Running pred_bert_large.py..."
python pred_bert_large.py

echo "Script execution completed."
