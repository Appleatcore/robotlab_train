#!/bin/bash
#SBATCH --job-name=go2w_hmap_lstm_2gpu
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx4090d:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

echo "===== job start ====="
date
hostname

# ===== 初始化环境 =====
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
nvidia-smi

# ===== 进入目录 =====
cd ~/workspace/robotlab_train

# ===== 执行你的命令 =====
torchrun --standalone --nproc_per_node=2 \
  scripts/reinforcement_learning/cusrl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent cusrl_heightmap_cfg_entry_point \
  --num_envs 8192 \
  --max_iterations 60000 \
  --run_name hmap_lstm_new_cfg_fstumble_m04_no_fhbody_flat_ori_m2 \
  --headless
