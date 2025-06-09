#!/bin/bash

# Launch a new terminal with 2x the default height and run this same script inside it
if [[ -z "$MSR_TMUX_ALREADY_RESIZED" ]]; then
    # Estimate default size as 80x24 → double height = 80x48
    gnome-terminal --geometry=80x48 -- bash -c "MSR_TMUX_ALREADY_RESIZED=1 $0"
    exit 0
fi

# Setup ROS 2 workspace
source install/setup.bash

# Kill existing session named msr if it exists
if tmux has-session -t msr 2>/dev/null; then
    tmux kill-session -t msr
fi

# Start new tmux session and split it
tmux new-session -d -s msr \; \
  send-keys 'source install/setup.bash && ros2 run motor_control teleop_twist_keyboard' Enter \; \
  split-window -h \; \
  send-keys 'source install/setup.bash && ros2 run msr_simulation trajectory_commander' Enter \; \
  select-pane -L \; \
  split-window -v \; \
  send-keys 'source install/setup.bash && ros2 run map_context_tests map_tagger' Enter \; \
  select-layout tiled \; \
  attach
