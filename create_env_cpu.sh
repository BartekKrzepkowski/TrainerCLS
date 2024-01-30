#!/bin/bash
eval "$(conda shell.bash hook)"
export CONDA_ALWAYS_YES="true"
if [ -f environment.yml ]; then
  conda env create -f environment.yml
else
  conda create -n trainer_cls_env python=3.10
  conda activate trainer_cls_env
  # mkdir pip-build

  conda install pytorch torchvision torchaudio cpuonly -c pytorch --yes
  conda install -c conda-forge scikit-learn seaborn --yes
  conda install -c conda-forge clearml wandb tensorboard --yes
  conda install -c conda-forge tqdm omegaconf --yes

  # rm -rf pip-build
  # conda env export | grep -v "^prefix: " > environment.yml
fi
