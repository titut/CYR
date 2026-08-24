"""Zenoh-based AprilTag detector node.

Subscribes to raw camera-visible tag data from the simulator and publishes
detected tags with their relative transforms to the robot frame.

In a real system this node would process camera images, detect AprilTags,
and estimate their 6-DOF pose.  Here we receive pre-computed range/bearing
measurements from the simulator and republish them as structured detections.

Topics:
    Subscribes:  sensor/camera/apriltag  — {"t": float, "detections": [{id, range_m, ...}]}
    Publishes:   detection/apriltag      — {"t": float, "detections": [{id, x_rel, y_rel, yaw_rel, size_m}]}

Usage:
    python -m zenoh.apriltag_detection.detector [path/to/map.json]

Or:
    python zenoh/apriltag_detection/detector.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import List, Optional

import zenoh
import zenoh.handlers

# Allow running this file directly or as a module.
_SCRIPT_DIR = Path(__file__).resolve().parent
_ZENOH_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _ZENOH_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_ZENOH_DIR) not in sys.path:
    sys.path.insert(0, str(_ZENOH_DIR))

from core.messages import SchemaError, decode, encode


# Simulated image-processing time per detection (seconds).
DETECTION_PROCESSING_S = 0.08


class ApriltagDetector:
    """Receives raw camera-visible tag data and publishes relative transforms.

    The simulator publishes {id, range_m, bearing_rad, tag_yaw_rad, tag_size_m}
    for each tag within the camera FOV.  This node converts those polar
    measurements into Cartesian relative transforms (x_rel, y_rel, yaw_rel)
    in the robot's body frame.
    """

    def __init__(self):
        self._session = zenoh.open(zenoh.Config())

        # Publisher: structured detection data.
        self._pub_detection = self._session.declare_publisher(
            "detection/apriltag",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
        )

        # Subscriber: raw camera-visible tag data (RingChannel(1)).
        self._sub_camera = self._session.declare_subscriber(
            "sensor/camera/apriltag",
            zenoh.handlers.RingChannel(1),
        )

    # -----------------------------------------------------------------------
    # Processing
    # -----------------------------------------------------------------------

    def _process_camera_data(self, sample):
        """Convert polar measurements to Cartesian relative transforms."""
        try:
            raw = decode("sensor/camera/apriltag", sample)
            raw_tags = raw.get("detections", [])
            source_t = raw.get("t")
        except SchemaError as exc:
            print(f"[apriltag_detector] sensor/camera/apriltag dropped: {exc}")
            return

        detections = []
        for tag in raw_tags:
            tag_id = tag["id"]
            range_m = tag["range_m"]
            bearing_rad = tag["bearing_rad"]
            tag_yaw_rad = tag["tag_yaw_rad"]
            tag_size_m = tag["tag_size_m"]

            # Convert polar (range, bearing) to Cartesian in robot frame.
            # Robot frame: +x forward, +y left, +θ CCW.
            x_rel = range_m * math.cos(bearing_rad)
            y_rel = range_m * math.sin(bearing_rad)

            detections.append(
                {
                    "id": tag_id,
                    "x_rel": round(x_rel, 4),
                    "y_rel": round(y_rel, 4),
                    "yaw_rel": round(tag_yaw_rad, 4),  # tag yaw in robot frame
                    "size_m": tag_size_m,
                }
            )

        if detections:
            # Simulate image-processing latency.  The output keeps the camera's
            # capture time ("t") so the downstream anchor is not shifted by the
            # detection compute time.
            time.sleep(DETECTION_PROCESSING_S)

            fields = {"detections": detections}
            if source_t is not None:
                fields["t"] = source_t
            self._pub_detection.put(encode("detection/apriltag", fields))

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self):
        print("[apriltag_detector] Running. Press Ctrl+C to stop.")
        try:
            while True:
                sample = self._sub_camera.recv()
                self._process_camera_data(sample)
        except KeyboardInterrupt:
            print("[apriltag_detector] Stopping…")
        finally:
            self._session.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    ApriltagDetector().run()


if __name__ == "__main__":
    main()
