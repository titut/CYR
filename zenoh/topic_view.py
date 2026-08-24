"""Subscribe to one or more zenoh topics and print a readable summary.

Lets you watch the simulator's output live (sensors, truth pose, registry, ...).
Messages are decoded/validated via the schema registry in ``core.messages``, so
malformed or unexpected payloads are reported rather than silently dropped.

Usage (run the simulator in another terminal first):
    python3 zenoh/topic_view.py sensor/lidar
    python3 zenoh/topic_view.py sensor/imu sensor/camera/apriltag sim/truth/pose
    python3 zenoh/topic_view.py sensor/lidar --rate 5   # rate-limit prints

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import zenoh

from core.messages import SchemaError, decode, decode_path, decode_text


def _pretty(topic: str, payload: str) -> str:
    """Try to render the payload readably; fall back to raw text."""
    try:
        data = json.loads(payload)
        return json.dumps(data, indent=None, default=str)
    except (json.JSONDecodeError, ValueError):
        return payload


def main():
    parser = argparse.ArgumentParser(description="View live zenoh topics")
    parser.add_argument("topics", nargs="+", help="topic key expressions to subscribe to")
    parser.add_argument("--rate", type=float, default=0.0,
                        help="print at most N messages/sec per topic (0 = all)")
    args = parser.parse_args()

    session = zenoh.open(zenoh.Config())
    print(f"Subscribed to: {', '.join(args.topics)}  (Ctrl+C to stop)\n")

    last_print = {}

    def on_sample(sample):
        topic = str(sample.key_expr)
        now = time.monotonic()
        if args.rate > 0 and now - last_print.get(topic, 0.0) < 1.0 / args.rate:
            return
        last_print[topic] = now

        payload = sample.payload.to_string()
        try:
            # dict messages validate + decode; text/array topics fall back.
            decoded = decode(topic, payload)
            rendered = json.dumps(decoded, default=str)
        except SchemaError as exc:
            rendered = f"<schema: {exc}> {payload}"
        except Exception:
            rendered = _pretty(topic, payload)

        print(f"[{topic}] {rendered[:400]}")

    subs = [
        session.declare_subscriber(t, on_sample)
        for t in args.topics
    ]

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        session.close()


if __name__ == "__main__":
    main()
