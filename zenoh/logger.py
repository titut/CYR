"""Zenoh topic recorder node.

Subscribes to *every* Zenoh topic (wildcard ``**``) and appends each message
as one JSON line to a log file, so an offline analysis can replay / inspect the
full message flow (pose estimates, LIDAR, commands, paths, detections, ...).

Each line is:
    {"t_wall": <ISO wall-clock time>, "t_mono": <seconds since boot>,
     "topic": <key expression>, "payload": <raw string>}

The payload is stored verbatim (most topics are already JSON strings).  Flushed
after every line so nothing is lost if the node is killed mid-run.

Usage:
    python zenoh/logger.py [path/to/log.jsonl]

Defaults to zenoh/logs/recording_<timestamp>.jsonl if no path is given.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import zenoh

# Allow running this file directly or as a module.
_SCRIPT_DIR = Path(__file__).resolve().parent
_ZENOH_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _ZENOH_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_ZENOH_DIR) not in sys.path:
    sys.path.insert(0, str(_ZENOH_DIR))

from clock import sleep_until


class Logger:
    def __init__(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(out_path, "a", encoding="utf-8")
        self._write_lock = threading.Lock()
        self._count = 0

        self._session = zenoh.open(zenoh.Config())
        self._sub = self._session.declare_subscriber("**", self._on_sample)

        print(f"[logger] Recording every topic to {out_path}")
        print("[logger] Press Ctrl+C to stop.")

    def _on_sample(self, sample):
        record = {
            "t_wall": datetime.now().isoformat(timespec="milliseconds"),
            "t_mono": time.monotonic(),
            "topic": str(sample.key_expr),
            "payload": sample.payload.to_string(),
        }
        with self._write_lock:
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()
            self._count += 1

    def run(self):
        # Keep-alive loop (the real work happens in the Zenoh callback).
        next_tick = time.monotonic()
        try:
            while True:
                next_tick += 1.0
                sleep_until(next_tick)
        except KeyboardInterrupt:
            print(f"[logger] Stopping. Logged {self._count} messages.")
        finally:
            self._session.close()
            self._file.close()


def main():
    out_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("zenoh/logs") / f"recording_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    )
    Logger(out_path).run()


if __name__ == "__main__":
    main()
