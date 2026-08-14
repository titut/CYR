#!/bin/bash

# 1. Catch Ctrl+C (SIGINT) and exit signals (SIGTERM)
# This kills all background processes running in this script's process group
trap 'echo "Stopping all programs..."; kill 0' SIGINT SIGTERM

# 2. Start your infinite programs in the background
python3 zenoh/pose_estimation/pose_estimator.py home.json &
python3 zenoh/navigation/navigator.py home.json &
python3 zenoh/apriltag_detection/detector.py &

# 3. Wait indefinitely for the user to stop the script
wait