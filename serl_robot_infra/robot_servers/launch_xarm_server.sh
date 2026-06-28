#!/usr/bin/env bash
set -euo pipefail

# Run from:
#   cd <hil-serl>/serl_robot_infra/robot_servers
#   bash launch_xarm_server.sh
#
# If import robot_servers fails, run from repo root and set PYTHONPATH:
#   export PYTHONPATH=$PYTHONPATH:<hil-serl>/serl_robot_infra

python xarm_server.py \
  --robot_ip=192.168.1.XXX \
  --flask_url=127.0.0.1 \
  --flask_port=5000 \
  --dof_robot=6 \
  --dof_env=7 \
  --speed=30 \
  --mvacc=300 \
  --max_pos_delta=0.02 \
  --max_rot_delta=0.25 \
  --gripper_open_value=850 \
  --gripper_closed_value=0 \
  --gripper_speed=3000
