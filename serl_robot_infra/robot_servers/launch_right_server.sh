#!/usr/bin/env bash
set -euo pipefail

echo "[HIL-SERL] Launching xArm server on 127.0.0.2:5000 ..."

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERL_INFRA_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$SERL_INFRA_DIR:${PYTHONPATH:-}"

cd "$SERL_INFRA_DIR"

python -m robot_servers.xarm_server \
    --robot_ip=192.168.1.219 \
    --flask_url=127.0.0.2 \
    --flask_port=5000 \
    --dof_robot=6 \
    --dof_env=7 \
    --speed=10 \
    --mvacc=100 \
    --max_pos_delta=0.005 \
    --max_rot_delta=0.05 \
    --gripper_open_value=850 \
    --gripper_closed_value=0 \
    --gripper_speed=1000
