#!/bin/zshrc
#SBATCH --job-name=debug_4090
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx4090d:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# ===== 初始化环境 =====
source ~/.zshrc
conda activate env_isaaclab

# ===== 进入目录 =====
cd ~/workspace/robotlab_train

# ===== 执行你的命令 =====
python scripts/reinforcement_learning/cusrl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent cusrl_heightmap_cfg_entry_point \
  --num_envs 4096 \
  --max_iterations 20000 \
  --headless
