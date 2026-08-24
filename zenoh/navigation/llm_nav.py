"""LLM integration for natural-language navigation commands.

Sends the user's text command alongside the map's room layout to DeepSeek
(via the OpenAI-compatible API) and returns a target location in meters.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Optional, Tuple

from core.map_format import MapData

_log = logging.getLogger("llm_nav")


def _build_prompt(map_data: MapData, user_command: str) -> str:
    """Build the system + user prompt for the LLM.

    ``map_data`` geometry is in meters.
    """
    room_info = []
    for room in map_data.rooms:
        room_info.append(
            {
                "id": room.id,
                "name": room.name,
                "center_m": [round(room.center[0], 3), round(room.center[1], 3)],
            }
        )

    system = (
        "You are a navigation assistant for a 2D indoor map.  "
        "The user gives a natural-language command like 'go to the kitchen'.  "
        "You must respond with ONLY a JSON object containing the target "
        'coordinates in meters: {"x": float, "y": float}.  '
        "Pick the location of the room that best matches the user's command.  "
        "The user will tell you to go to a specific location based on description like"
        "Go to kitchen near the corner, reason about where that is in the room"
        "Do not include any other text."
    )

    user = (
        f"Map rooms (centres in meters):\n{json.dumps(room_info, indent=2)}\n\n"
        f'User command: "{user_command}"'
    )

    return f"{system}\n\n{user}"


def query_location(
    map_data: MapData,
    user_command: str,
    api_key: str,
    model: str = "deepseek-v4-pro",
    base_url: str = "https://api.deepseek.com",
    timeout: float = 10.0,
) -> Optional[Tuple[float, float]]:
    """Send a navigation query to DeepSeek and return the target in meters.

    Returns ``None`` if the API call fails or the response cannot be parsed.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("[llm_nav] openai package not installed.  pip install openai")
        return None

    prompt = _build_prompt(map_data, user_command)

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=256,
            temperature=0.0,
        )
    except Exception as e:
        print(f"[llm_nav] API error: {e}")
        return None

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

    try:
        data = json.loads(raw)
        x_m = float(data["x"])
        y_m = float(data["y"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        _log.error("Failed to parse response: %r  error: %s", raw, e)
        print(f"[llm_nav] Failed to parse response: {raw!r}  error: {e}")
        return None

    result = (x_m, y_m)
    _log.info("Resolved to map meters: (%.3f, %.3f)", result[0], result[1])
    return result


def query_location_async(
    map_data: MapData,
    user_command: str,
    api_key: str,
    callback,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    timeout: float = 10.0,
):
    """Run query_location in a background thread and call ``callback(result)``.

    The callback receives ``Optional[Tuple[float, float]]`` — the target in
    meters, or ``None`` on failure.
    """

    def _worker():
        _log.info("Async query started for: %r", user_command)
        result = query_location(
            map_data, user_command, api_key, model, base_url, timeout
        )
        _log.info("Async query complete, result=%s", result)
        callback(result)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
