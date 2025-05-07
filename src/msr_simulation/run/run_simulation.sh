#!/bin/bash

source install/setup.bash

# Kill existing session named msr if it exists
if tmux has-session -t msr 2>/dev/null; then
    tmux kill-session -t msr
fi

# Start new tmux session and immediately split it
tmux new-session -d -s msr 'ros2 run motor_control teleop_twist_keyboard'
tmux split-window -h -t msr:0 'ros2 run map_context_tests map_tagger'
tmux split-window -v -t msr:0 'ros2 run msr_simulation trajectory_commander'

# Arrange panes nicely (optional: even out)
tmux select-layout -t msr:0 tiled

# Attach to session
tmux attach-session -t msr
