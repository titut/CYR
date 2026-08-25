#!/bin/bash

# 1. Kill any leftover nodes from a previous run.  A stale process (from an
# earlier session that wasn't cleaned up) keeps publishing stale data on the
# same topics and silently overrides the fresh nodes below — e.g. an old
# navigator with a remembered goal, or an old drive node holding the sim's
# teleop at (0,0).  None of the launches below should already exist.
# (The 3D sim itself is started separately and is NOT killed here.)
pkill -f "zenoh/control/drive.py"        2>/dev/null
pkill -f "zenoh/control/controller.py"   2>/dev/null
pkill -f "zenoh/pose_estimation/pose_estimator.py" 2>/dev/null
pkill -f "zenoh/navigation/navigator.py" 2>/dev/null
pkill -f "zenoh/apriltag_detection/detector.py"    2>/dev/null
pkill -f "zenoh/logger.py"               2>/dev/null
pkill -f "zenoh/sim_viewer.py"           2>/dev/null
pkill -f "simulation3d/simulator.py"           2>/dev/null
sleep 0.2

# 2. Catch Ctrl+C (SIGINT) and exit signals (SIGTERM)
# This kills all background processes running in this script's process group.
# The trap is reset first so that the kill signal we send does not re-trigger it.
trap 'trap - SIGINT SIGTERM; echo "Stopping all programs..."; kill 0' SIGINT SIGTERM

# 3. Start your infinite programs in the background
python3 simulation3d/simulator.py --map home.json &
python3 zenoh/control/drive.py &
python3 zenoh/control/controller.py &
python3 zenoh/pose_estimation/pose_estimator.py home.json &
python3 zenoh/navigation/navigator.py home.json &
python3 zenoh/apriltag_detection/detector.py &
python3 zenoh/logger.py &

# 4. Wait indefinitely for the user to stop the script
wait