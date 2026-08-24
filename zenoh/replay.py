"""Zenoh session replayer (T-020).

Reads a recorded session (the JSONL written by ``zenoh/logger.py``) and
republishes every message back onto its original topic, paced by the recorded
timestamps — a rosbag-play equivalent so the live stack can be driven by
recorded data.

Only the original ``payload`` of each record is republished: the logger's
``t_wall``/``t_mono``/``topic`` envelope is metadata, not part of the messages.

Usage:
    python zenoh/replay.py <log.jsonl> [--speed 2.0] [--topics sensor/lidar,estimate/pose]

    --speed   pacing multiplier (default 1.0 = real time; 0 = as fast as
              possible)
    --topics  optional comma-separated topic filter
    --dry-run print a summary without publishing

For a *deterministic* replay, seed the receiving nodes (they use the global
RNG) and replay with ``--speed 0``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import zenoh

# Allow running this file directly or as a module.
_SCRIPT_DIR = Path(__file__).resolve().parent
_ZENOH_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _ZENOH_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_ZENOH_DIR) not in sys.path:
    sys.path.insert(0, str(_ZENOH_DIR))

from core.clock import sleep_until


class Replayer:
    def __init__(self, log_path: Path, speed: float, topics: Optional[set]):
        self._log_path = log_path
        self._speed = speed
        self._topics = topics
        self._session = zenoh.open(zenoh.Config())
        self._pubs: Dict[str, object] = {}

    def _publisher(self, topic: str):
        pub = self._pubs.get(topic)
        if pub is None:
            pub = self._session.declare_publisher(
                topic,
                congestion_control=zenoh.CongestionControl.DROP,
                reliability=zenoh.Reliability.BEST_EFFORT,
            )
            self._pubs[topic] = pub
        return pub

    def _summarize(self) -> None:
        counts: Dict[str, int] = {}
        duration = 0.0
        first: Optional[float] = None
        last: Optional[float] = None
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                topic = rec.get("topic")
                if self._topics and topic not in self._topics:
                    continue
                t = rec.get("t_mono")
                if first is None:
                    first = t
                last = t
                counts[topic] = counts.get(topic, 0) + 1
        if first is not None and last is not None:
            duration = last - first
        print(f"[replay] {self._log_path}")
        for topic, n in sorted(counts.items()):
            print(f"[replay]   {topic}: {n} messages")
        print(f"[replay]   recorded duration: {duration:.1f} s")

    def run(self) -> None:
        print(f"[replay] Replaying {self._log_path} "
              f"(speed={self._speed or 'as-fast-as-possible'})")
        start_wall = time.monotonic()
        start_log: Optional[float] = None
        n = 0

        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                topic = rec.get("topic")
                payload = rec.get("payload", "")
                if not topic:
                    continue
                if self._topics and topic not in self._topics:
                    continue

                t_mono = rec.get("t_mono")
                if start_log is None:
                    start_log = t_mono
                if self._speed and t_mono is not None and start_log is not None:
                    target = start_wall + (t_mono - start_log) / self._speed
                    sleep_until(target)

                self._publisher(topic).put(payload)
                n += 1

        print(f"[replay] Done: {n} messages replayed in "
              f"{time.monotonic() - start_wall:.1f} s.")
        self._session.close()


def main():
    parser = argparse.ArgumentParser(description="Replay a recorded Zenoh session.")
    parser.add_argument("log", type=Path, help="path to the recorded .jsonl log")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="pacing multiplier (0 = as fast as possible)")
    parser.add_argument("--topics", type=str, default=None,
                        help="comma-separated topic filter")
    parser.add_argument("--dry-run", action="store_true",
                        help="summarize the log without publishing")
    args = parser.parse_args()

    if not args.log.exists():
        print(f"[replay] No such log: {args.log}")
        sys.exit(1)

    topics = set(t.strip() for t in args.topics.split(",")) if args.topics else None
    replayer = Replayer(args.log, args.speed, topics)
    if args.dry_run:
        replayer._summarize()
        replayer._session.close()
        return
    replayer.run()


if __name__ == "__main__":
    main()
