"""A simple desktop map editor for semantic navigation maps.

Run from the project root with:
    python -m map_editor.editor [path/to/map.json]

Or:
    python map_editor/editor.py [path/to/map.json]

Tools:
    Wall   - Click-drag or click-click to draw wall segments.
    Room   - Click to add polygon vertices; close the polygon to name the room.
    Scale  - Click two points, then enter the real-world distance.
    Origin - Click a point to set the map origin; all geometry is rebased relative to it.
"""

from __future__ import annotations

import json
import math
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog
from typing import List, Optional, Tuple

from map_format import Apriltag, MapData, Metadata, Obstacle, Room, Wall, new_empty_map

CANVAS_MARGIN = 20
GRID_SPACING = 20
SNAP_RADIUS = 10
WALL_COLOR = "#222222"
WALL_WIDTH = 3
ROOM_FILL = "#e3f2fd"
ROOM_OUTLINE = "#2196f3"
ROOM_OUTLINE_WIDTH = 2
SCALE_COLOR = "#ff9800"
ORIGIN_COLOR = "#d32f2f"
APRILTAG_COLOR = "#2e7d32"
APRILTAG_DELETE_COLOR = "#c62828"
OBSTACLE_COLOR = "#ff7043"
OBSTACLE_OUTLINE = "#bf360c"
PREVIEW_COLOR = "#9e9e9e"


def _point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    """Even-odd rule point-in-polygon test."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        ):
            inside = not inside
    return inside


@dataclass
class EditorState:
    """Transient drawing state for the currently active tool."""

    tool: str = "wall"  # wall | room | scale | origin | apriltag | obstacle
    wall_start: Optional[Tuple[float, float]] = None
    room_points: List[Tuple[float, float]] = None  # type: ignore[assignment]
    scale_points: List[Tuple[float, float]] = None  # type: ignore[assignment]
    preview_id: Optional[int] = None

    def __post_init__(self):
        if self.room_points is None:
            self.room_points = []
        if self.scale_points is None:
            self.scale_points = []

    def reset(self):
        self.wall_start = None
        self.room_points = []
        self.scale_points = []
        self.preview_id = None


class MapEditor:
    def __init__(self, root: tk.Tk, initial_map: Optional[Path] = None):
        self.root = root
        self.root.title("Semantic Navigation Map Editor")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)

        self.map_data: MapData = new_empty_map()
        self.file_path: Optional[Path] = None
        self.state = EditorState()

        if initial_map is not None:
            self._load_file(initial_map)

        self._build_ui()
        self._bind_events()
        self._render_map()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Menu bar
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New", command=self._new_map, accelerator="Ctrl+N")
        file_menu.add_command(
            label="Open...", command=self._open_file, accelerator="Ctrl+O"
        )
        file_menu.add_command(
            label="Save", command=self._save_file, accelerator="Ctrl+S"
        )
        file_menu.add_command(label="Save As...", command=self._save_as_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        # Main layout frames
        self.toolbar = tk.Frame(
            self.root, width=140, bg="#f5f5f5", relief=tk.RIDGE, bd=2
        )
        self.toolbar.pack(side=tk.LEFT, fill=tk.Y)

        self.canvas_frame = tk.Frame(self.root, bg="white")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.sidebar = tk.Frame(
            self.root, width=240, bg="#fafafa", relief=tk.RIDGE, bd=2
        )
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)

        # Toolbar buttons
        tk.Label(
            self.toolbar, text="Tools", bg="#f5f5f5", font=("Helvetica", 12, "bold")
        ).pack(pady=10)

        self.tool_buttons: dict[str, tk.Button] = {}
        for tool, icon_text in [
            ("wall", "Wall"),
            ("room", "Room"),
            ("scale", "Scale"),
            ("origin", "Origin"),
            ("apriltag", "AprilTag"),
            ("obstacle", "Obstacle"),
        ]:
            btn = tk.Button(
                self.toolbar,
                text=icon_text,
                width=12,
                relief=tk.RAISED,
                command=lambda t=tool: self._set_tool(t),
            )
            btn.pack(pady=5, padx=10)
            self.tool_buttons[tool] = btn

        tk.Button(
            self.toolbar,
            text="Clear All",
            width=12,
            fg="red",
            command=self._clear_map,
        ).pack(side=tk.BOTTOM, pady=20, padx=10)

        # Canvas
        width, height = self.map_data.metadata.size_px
        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=width + 2 * CANVAS_MARGIN,
            height=height + 2 * CANVAS_MARGIN,
            bg="white",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Sidebar
        tk.Label(
            self.sidebar,
            text="Properties",
            bg="#fafafa",
            font=("Helvetica", 12, "bold"),
        ).pack(pady=10)

        self.sidebar_info = tk.Label(
            self.sidebar,
            text="No item selected",
            bg="#fafafa",
            justify=tk.LEFT,
            wraplength=220,
        )
        self.sidebar_info.pack(pady=5, padx=10, anchor=tk.NW)

        self.close_room_btn = tk.Button(
            self.sidebar,
            text="Close Polygon",
            state=tk.DISABLED,
            command=self._close_room_polygon,
        )
        self.close_room_btn.pack(pady=10, padx=10, fill=tk.X)

        # Status bar
        self.status = tk.Label(
            self.root,
            text="Ready",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
        )
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        self._set_tool("wall")

    def _bind_events(self):
        self.root.bind("<Control-n>", lambda e: self._new_map())
        self.root.bind("<Control-o>", lambda e: self._open_file())
        self.root.bind("<Control-s>", lambda e: self._save_file())

        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Double-Button-1>", self._on_canvas_double_click)
        self.canvas.bind("<BackSpace>", self._on_backspace)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.root.bind("<Escape>", lambda e: self._cancel_drawing())

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    @property
    def _origin_x(self) -> float:
        return float(self.map_data.metadata.origin_px[0])

    @property
    def _origin_y(self) -> float:
        return float(self.map_data.metadata.origin_px[1])

    def _to_map(self, cx: float, cy: float) -> Tuple[float, float]:
        """Convert canvas coordinates to map coordinates (relative to origin)."""
        return (
            cx - CANVAS_MARGIN - self._origin_x,
            cy - CANVAS_MARGIN - self._origin_y,
        )

    def _to_canvas(self, mx: float, my: float) -> Tuple[float, float]:
        """Convert map coordinates (relative to origin) to canvas coordinates."""
        return (
            mx + self._origin_x + CANVAS_MARGIN,
            my + self._origin_y + CANVAS_MARGIN,
        )

    def _snap(self, mx: float, my: float) -> Tuple[float, float]:
        """Snap a map coordinate to the nearest grid intersection."""
        return (
            round(mx / GRID_SPACING) * GRID_SPACING,
            round(my / GRID_SPACING) * GRID_SPACING,
        )

    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _point_to_segment_distance(px: float, py: float, wall: Wall) -> float:
        """Distance from point (px, py) to the wall segment."""
        wx = wall.x2 - wall.x1
        wy = wall.y2 - wall.y1
        length_sq = wx * wx + wy * wy
        if length_sq == 0:
            return math.hypot(px - wall.x1, py - wall.y1)
        t = max(
            0.0,
            min(1.0, ((px - wall.x1) * wx + (py - wall.y1) * wy) / length_sq),
        )
        proj_x = wall.x1 + t * wx
        proj_y = wall.y1 + t * wy
        return math.hypot(px - proj_x, py - proj_y)

    # ------------------------------------------------------------------
    # Tool switching
    # ------------------------------------------------------------------

    def _set_tool(self, tool: str):
        self.state.reset()
        self.state.tool = tool
        for t, btn in self.tool_buttons.items():
            btn.config(relief=tk.SUNKEN if t == tool else tk.RAISED)
        self._update_status()
        self._update_sidebar()

    def _update_status(self):
        tool = self.state.tool
        messages = {
            "wall": "Wall tool: click-drag or click two points to draw a wall.",
            "room": "Room tool: click to add vertices, double-click or Close Polygon to finish.",
            "scale": "Scale tool: click two points, then enter the real distance.",
            "origin": "Origin tool: click a point to set the map origin. Right-click origin to reset.",
            "apriltag": "AprilTag tool: click to place a tag. Right-click a tag to delete it.",
            "obstacle": "Obstacle tool: click to place an obstacle. Right-click one to delete it.",
        }
        self.status.config(text=messages.get(tool, "Ready"))

    def _update_sidebar(self):
        lines = [f"Tool: {self.state.tool.capitalize()}", ""]

        if self.state.tool == "room":
            lines.append(f"Room vertices: {len(self.state.room_points)}")
            self.close_room_btn.config(
                state=tk.NORMAL if len(self.state.room_points) >= 3 else tk.DISABLED
            )
        else:
            lines.append("No active polygon.")
            self.close_room_btn.config(state=tk.DISABLED)

        lines.append("")
        lines.append(f"Walls: {len(self.map_data.walls)}")
        lines.append(f"Rooms: {len(self.map_data.rooms)}")
        lines.append(f"AprilTags: {len(self.map_data.apriltags)}")
        lines.append(f"Obstacles: {len(self.map_data.obstacles)}")
        lines.append("")
        scale = self.map_data.metadata.scale_m_per_px
        lines.append(f"Scale: {scale:.4f} m/px")
        lines.append("")
        origin = self.map_data.metadata.origin_px
        lines.append(f"Origin: ({int(origin[0])}, {int(origin[1])})")

        self.sidebar_info.config(text="\n".join(lines))

    # ------------------------------------------------------------------
    # Canvas events
    # ------------------------------------------------------------------

    def _on_canvas_click(self, event: tk.Event):
        mx, my = self._snap(*self._to_map(event.x, event.y))

        if self.state.tool == "wall":
            if self.state.wall_start is None:
                self.state.wall_start = (mx, my)
            else:
                self._add_wall(
                    self.state.wall_start[0], self.state.wall_start[1], mx, my
                )
                self.state.wall_start = None
                self._clear_preview()

        elif self.state.tool == "room":
            # Close polygon if clicking near the first point
            if len(self.state.room_points) >= 3:
                first = self.state.room_points[0]
                if self._distance((mx, my), first) < SNAP_RADIUS:
                    self._close_room_polygon()
                    return
            self.state.room_points.append((mx, my))
            self._render_drawing_state()

        elif self.state.tool == "scale":
            if len(self.state.scale_points) < 2:
                self.state.scale_points.append((mx, my))
                if len(self.state.scale_points) == 2:
                    self._finish_scale()
                else:
                    self._render_drawing_state()

        elif self.state.tool == "origin":
            self._set_origin(mx, my)

        elif self.state.tool == "apriltag":
            self._add_apriltag(mx, my)

        elif self.state.tool == "obstacle":
            self._add_obstacle(mx, my)

        self._update_sidebar()

    def _on_canvas_drag(self, event: tk.Event):
        if self.state.tool != "wall" or self.state.wall_start is None:
            return
        mx, my = self._snap(*self._to_map(event.x, event.y))
        self._draw_preview_line(self.state.wall_start, (mx, my))

    def _on_canvas_release(self, event: tk.Event):
        if self.state.tool == "wall" and self.state.wall_start is not None:
            mx, my = self._snap(*self._to_map(event.x, event.y))
            if self._distance(self.state.wall_start, (mx, my)) > 0:
                self._add_wall(
                    self.state.wall_start[0], self.state.wall_start[1], mx, my
                )
            self.state.wall_start = None
            self._clear_preview()

    def _on_canvas_motion(self, event: tk.Event):
        mx, my = self._snap(*self._to_map(event.x, event.y))
        self.status.config(
            text=f"{self.state.tool.capitalize()} tool | Map: ({int(mx)}, {int(my)})"
        )

    def _on_canvas_double_click(self, event: tk.Event):
        if self.state.tool == "room" and len(self.state.room_points) >= 3:
            self._close_room_polygon()

    def _on_backspace(self, event: tk.Event):
        if self.state.tool == "room" and self.state.room_points:
            self.state.room_points.pop()
            self._render_drawing_state()
            self._update_sidebar()

    def _on_right_click(self, event: tk.Event):
        """Right-click: delete room/wall, or reset origin when Origin tool is active."""
        if self.state.tool == "origin":
            self._reset_origin()
            return

        if self.state.tool == "apriltag":
            mx, my = self._to_map(event.x, event.y)
            idx = self._find_nearest_apriltag(mx, my)
            if idx is not None:
                tag = self.map_data.apriltags[idx]
                del self.map_data.apriltags[idx]
                self._render_map()
                self._update_sidebar()
                self.status.config(text=f"Deleted AprilTag {tag.id}")
            else:
                self.status.config(text="No tag near cursor to delete.")
            return

        if self.state.tool == "obstacle":
            mx, my = self._to_map(event.x, event.y)
            idx = self._find_nearest_obstacle(mx, my)
            if idx is not None:
                obstacle = self.map_data.obstacles[idx]
                del self.map_data.obstacles[idx]
                self._render_map()
                self._update_sidebar()
                self.status.config(text=f"Deleted Obstacle {obstacle.id}")
            else:
                self.status.config(text="No obstacle near cursor to delete.")
            return

        mx, my = self._to_map(event.x, event.y)

        # Check rooms first — if the click is inside a room polygon, delete it.
        room_name = self._find_room_at_point(mx, my)
        if room_name is not None:
            self.map_data.rooms = [
                r for r in self.map_data.rooms if r.name != room_name
            ]
            self._render_map()
            self._update_sidebar()
            self.status.config(text=f"Deleted room '{room_name}'")
            return

        # Otherwise try to delete a wall.
        idx = self._find_nearest_wall(mx, my)
        if idx is not None:
            wall = self.map_data.walls[idx]
            del self.map_data.walls[idx]
            self._render_map()
            self._update_sidebar()
            self.status.config(
                text=f"Deleted wall ({wall.x1:.0f}, {wall.y1:.0f}) → "
                f"({wall.x2:.0f}, {wall.y2:.0f})"
            )
        else:
            self.status.config(text="No room or wall near cursor to delete.")

    def _find_room_at_point(self, mx: float, my: float) -> Optional[str]:
        """Return the name of the room containing (mx, my), or None."""
        for room in self.map_data.rooms:
            if _point_in_polygon(mx, my, room.polygon):
                return room.name
        return None

    def _find_nearest_wall(self, mx: float, my: float) -> Optional[int]:
        """Return the index of the wall closest to (mx, my), or None if
        no wall is within SNAP_RADIUS."""
        best_idx = None
        best_dist = SNAP_RADIUS
        for i, wall in enumerate(self.map_data.walls):
            d = self._point_to_segment_distance(mx, my, wall)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _find_nearest_apriltag(self, mx: float, my: float) -> Optional[int]:
        """Return the index of the tag closest to (mx, my), or None."""
        best_idx = None
        best_dist = SNAP_RADIUS * 1.5
        for i, tag in enumerate(self.map_data.apriltags):
            d = math.hypot(mx - tag.x, my - tag.y)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _find_nearest_obstacle(self, mx: float, my: float) -> Optional[int]:
        """Return the index of the obstacle containing (mx, my), or None."""
        best_idx = None
        best_dist = float("inf")
        for i, obstacle in enumerate(self.map_data.obstacles):
            d = math.hypot(mx - obstacle.x, my - obstacle.y)
            if d <= obstacle.radius_px + SNAP_RADIUS and d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _cancel_drawing(self):
        self.state.reset()
        self._clear_preview()
        self._render_map()
        self._update_sidebar()

    # ------------------------------------------------------------------
    # Origin handling
    # ------------------------------------------------------------------

    def _set_origin(self, new_ox: float, new_oy: float):
        """Set the map origin to (new_ox, new_oy) and rebase all geometry.

        (new_ox, new_oy) are in the current map coordinate frame.
        After this call, all coordinates are shifted so the origin is at (0, 0)
        in the stored frame; origin_px records where the origin was placed.
        """
        old_ox = self._origin_x
        old_oy = self._origin_y

        # The point the user clicked is (new_ox, new_oy) in the current relative
        # frame.  Its absolute position is:
        #   abs_x = new_ox + old_ox
        #   abs_y = new_oy + old_oy
        #
        # We want origin_px = (abs_x, abs_y) and all geometry to be relative
        # to that point.  So we shift every coordinate by subtracting (abs_x, abs_y).
        abs_x = new_ox + old_ox
        abs_y = new_oy + old_oy

        # Shift all walls.
        for wall in self.map_data.walls:
            wall.x1 -= abs_x
            wall.y1 -= abs_y
            wall.x2 -= abs_x
            wall.y2 -= abs_y

        # Shift all AprilTags.
        for tag in self.map_data.apriltags:
            tag.x -= abs_x
            tag.y -= abs_y

        # Shift all obstacles.
        for obstacle in self.map_data.obstacles:
            obstacle.x -= abs_x
            obstacle.y -= abs_y

        # Shift all rooms.
        for room in self.map_data.rooms:
            room.polygon = [(px - abs_x, py - abs_y) for px, py in room.polygon]
            cx, cy = room.center
            room.center = (cx - abs_x, cy - abs_y)

        # Update metadata.
        self.map_data.metadata.origin_px = (int(abs_x), int(abs_y))

        self._render_map()
        self._update_sidebar()
        self.status.config(
            text=f"Origin set to ({int(abs_x)}, {int(abs_y)}) — geometry rebased."
        )

    def _reset_origin(self):
        """Reset the origin back to (0, 0), shifting all geometry back.

        This is the inverse of _set_origin: we add origin_px back to every
        coordinate and set origin_px to (0, 0).
        """
        old_ox = self._origin_x
        old_oy = self._origin_y
        if old_ox == 0 and old_oy == 0:
            self.status.config(text="Origin is already at (0, 0).")
            return

        # Shift all walls back.
        for wall in self.map_data.walls:
            wall.x1 += old_ox
            wall.y1 += old_oy
            wall.x2 += old_ox
            wall.y2 += old_oy

        # Shift all AprilTags back.
        for tag in self.map_data.apriltags:
            tag.x += old_ox
            tag.y += old_oy

        # Shift all obstacles back.
        for obstacle in self.map_data.obstacles:
            obstacle.x += old_ox
            obstacle.y += old_oy

        # Shift all rooms back.
        for room in self.map_data.rooms:
            room.polygon = [(px + old_ox, py + old_oy) for px, py in room.polygon]
            cx, cy = room.center
            room.center = (cx + old_ox, cy + old_oy)

        self.map_data.metadata.origin_px = (0, 0)

        self._render_map()
        self._update_sidebar()
        self.status.config(text="Origin reset to (0, 0) — geometry restored.")

    # ------------------------------------------------------------------
    # Map mutations
    # ------------------------------------------------------------------

    def _add_wall(self, x1: float, y1: float, x2: float, y2: float):
        if self._distance((x1, y1), (x2, y2)) < 1:
            return
        self.map_data.walls.append(Wall(x1=x1, y1=y1, x2=x2, y2=y2))
        self._render_map()

    def _close_room_polygon(self):
        points = self.state.room_points
        if len(points) < 3:
            return

        name = simpledialog.askstring(
            "Room Name", "Enter room name:", initialvalue="Room"
        )
        if not name:
            name = f"room_{len(self.map_data.rooms) + 1}"

        room_id = name.lower().replace(" ", "_")
        # Make id unique
        existing_ids = {r.id for r in self.map_data.rooms}
        base_id = room_id
        suffix = 1
        while room_id in existing_ids:
            room_id = f"{base_id}_{suffix}"
            suffix += 1

        center = self._polygon_center(points)
        self.map_data.rooms.append(
            Room(id=room_id, name=name, polygon=points.copy(), center=center)
        )
        self.state.room_points = []
        self._render_map()
        self._update_sidebar()

    def _finish_scale(self):
        p1, p2 = self.state.scale_points
        pixel_distance = self._distance(p1, p2)
        if pixel_distance <= 0:
            messagebox.showerror("Scale Error", "Scale reference line has zero length.")
            self.state.scale_points = []
            return

        real_distance = simpledialog.askfloat(
            "Scale",
            "Real-world length of the reference line (meters):",
            minvalue=0.001,
        )
        if real_distance is None:
            self.state.scale_points = []
            self._render_map()
            return

        self.map_data.metadata.scale_m_per_px = real_distance / pixel_distance
        self.state.scale_points = []
        self._render_map()
        self._update_sidebar()

    # ------------------------------------------------------------------
    # AprilTag mutations
    # ------------------------------------------------------------------

    def _add_apriltag(self, mx: float, my: float):
        """Place a new AprilTag at the snapped map position."""
        # Auto-increment tag ID based on existing tags.
        existing_ids = {t.id for t in self.map_data.apriltags}
        tag_id = 0
        while tag_id in existing_ids:
            tag_id += 1

        self.map_data.apriltags.append(
            Apriltag(id=tag_id, x=mx, y=my, yaw_rad=0.0, size_m=0.16)
        )
        self._render_map()
        self._update_sidebar()
        self.status.config(text=f"Placed AprilTag {tag_id} at ({int(mx)}, {int(my)})")

    # ------------------------------------------------------------------
    # Obstacle mutations
    # ------------------------------------------------------------------

    def _add_obstacle(self, mx: float, my: float):
        """Place a new circular obstacle at the snapped map position."""
        existing_ids = {o.id for o in self.map_data.obstacles}
        obstacle_id = 0
        while obstacle_id in existing_ids:
            obstacle_id += 1

        self.map_data.obstacles.append(
            Obstacle(id=obstacle_id, x=mx, y=my, radius_px=GRID_SPACING)
        )
        self._render_map()
        self._update_sidebar()
        self.status.config(
            text=f"Placed Obstacle {obstacle_id} at ({int(mx)}, {int(my)})"
        )

    @staticmethod
    def _polygon_center(points: List[Tuple[float, float]]) -> Tuple[float, float]:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_map(self):
        self.canvas.delete("all")
        self._draw_grid()

        # Rooms
        for room in self.map_data.rooms:
            canvas_poly = [c for p in room.polygon for c in self._to_canvas(*p)]
            self.canvas.create_polygon(
                canvas_poly,
                fill=ROOM_FILL,
                outline=ROOM_OUTLINE,
                width=ROOM_OUTLINE_WIDTH,
                tags=("room", room.id),
            )
            cx, cy = self._to_canvas(*room.center)
            self.canvas.create_text(
                cx,
                cy,
                text=room.name,
                fill=ROOM_OUTLINE,
                font=("Helvetica", 10, "bold"),
                tags=("room_label", room.id),
            )

        # AprilTags
        for tag in self.map_data.apriltags:
            cx, cy = self._to_canvas(tag.x, tag.y)
            half = max(4.0, (tag.size_m / self.map_data.metadata.scale_m_per_px) / 2.0)
            rect_id = self.canvas.create_rectangle(
                cx - half,
                cy - half,
                cx + half,
                cy + half,
                fill=APRILTAG_COLOR,
                outline="#000000",
                width=1,
                tags=("apriltag", str(tag.id)),
            )
            # Direction line
            tip_x = cx + half * 1.6 * math.cos(tag.yaw_rad)
            tip_y = cy + half * 1.6 * math.sin(tag.yaw_rad)
            self.canvas.create_line(
                cx, cy, tip_x, tip_y, fill="#000000", width=1, tags=("apriltag",)
            )
            # Label
            self.canvas.create_text(
                cx,
                cy - half - 10,
                text=f"T{tag.id}",
                fill=APRILTAG_COLOR,
                font=("Helvetica", 8, "bold"),
                tags=("apriltag_label", str(tag.id)),
            )

        # Obstacles
        for obstacle in self.map_data.obstacles:
            cx, cy = self._to_canvas(obstacle.x, obstacle.y)
            self.canvas.create_oval(
                cx - obstacle.radius_px,
                cy - obstacle.radius_px,
                cx + obstacle.radius_px,
                cy + obstacle.radius_px,
                fill=OBSTACLE_COLOR,
                outline=OBSTACLE_OUTLINE,
                width=2,
                tags=("obstacle", str(obstacle.id)),
            )
            self.canvas.create_text(
                cx,
                cy,
                text=f"O{obstacle.id}",
                fill="#ffffff",
                font=("Helvetica", 8, "bold"),
                tags=("obstacle_label", str(obstacle.id)),
            )

        # Walls
        for wall in self.map_data.walls:
            x1, y1 = self._to_canvas(wall.x1, wall.y1)
            x2, y2 = self._to_canvas(wall.x2, wall.y2)
            self.canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill=WALL_COLOR,
                width=WALL_WIDTH,
                tags=("wall",),
            )

        # Scale reference line if active
        if len(self.state.scale_points) == 2:
            p1, p2 = self.state.scale_points
            self.canvas.create_line(
                *self._to_canvas(*p1),
                *self._to_canvas(*p2),
                fill=SCALE_COLOR,
                width=2,
                dash=(4, 4),
            )

        # Origin marker — draw at (0, 0) relative to origin, which is at
        # the origin_px absolute position.  Since _to_canvas adds origin_px,
        # canvas coords of the origin are simply (origin_px + MARGIN).
        self._draw_origin_marker()

        # Room preview
        self._render_drawing_state()

    def _draw_origin_marker(self):
        """Draw a crosshair marker at the origin position (0, 0 relative)."""
        # The origin in relative coords is always (0, 0).
        cx, cy = self._to_canvas(0, 0)
        size = 12

        # Crosshair
        self.canvas.create_line(
            cx - size,
            cy,
            cx + size,
            cy,
            fill=ORIGIN_COLOR,
            width=2,
            tags=("origin_marker",),
        )
        self.canvas.create_line(
            cx,
            cy - size,
            cx,
            cy + size,
            fill=ORIGIN_COLOR,
            width=2,
            tags=("origin_marker",),
        )
        # Circle
        self.canvas.create_oval(
            cx - size,
            cy - size,
            cx + size,
            cy + size,
            outline=ORIGIN_COLOR,
            width=2,
            tags=("origin_marker",),
        )
        # Label
        self.canvas.create_text(
            cx + size + 6,
            cy + size + 6,
            text="Origin",
            fill=ORIGIN_COLOR,
            font=("Helvetica", 8, "bold"),
            anchor=tk.NW,
            tags=("origin_marker",),
        )

    def _render_drawing_state(self):
        # Remove only preview items
        if self.state.preview_id is not None:
            self.canvas.delete(self.state.preview_id)
            self.state.preview_id = None

        if self.state.tool == "room" and self.state.room_points:
            points = self.state.room_points
            canvas_points = [c for p in points for c in self._to_canvas(*p)]
            if len(points) >= 2:
                self.canvas.create_line(
                    canvas_points,
                    fill=PREVIEW_COLOR,
                    width=2,
                    dash=(4, 4),
                    tags=("preview",),
                )
            for p in points:
                x, y = self._to_canvas(*p)
                self.canvas.create_oval(
                    x - 3,
                    y - 3,
                    x + 3,
                    y + 3,
                    fill=PREVIEW_COLOR,
                    tags=("preview",),
                )

    def _draw_grid(self):
        width, height = self.map_data.metadata.size_px
        x0, y0 = self._to_canvas(0, 0)
        x1, y1 = self._to_canvas(width, height)

        # Border
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#cccccc", width=1)

        # Grid lines — draw in the relative frame so they stay aligned with
        # snapped coordinates.
        for x in range(0, width + 1, GRID_SPACING):
            cx1, _ = self._to_canvas(x, 0)
            cx2, _ = self._to_canvas(x, height)
            self.canvas.create_line(cx1, y0, cx2, y1, fill="#eeeeee")
        for y in range(0, height + 1, GRID_SPACING):
            _, cy1 = self._to_canvas(0, y)
            _, cy2 = self._to_canvas(width, y)
            self.canvas.create_line(x0, cy1, x1, cy2, fill="#eeeeee")

    def _draw_preview_line(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ):
        self._clear_preview()
        x1, y1 = self._to_canvas(*start)
        x2, y2 = self._to_canvas(*end)
        self.state.preview_id = self.canvas.create_line(
            x1,
            y1,
            x2,
            y2,
            fill=PREVIEW_COLOR,
            width=WALL_WIDTH,
            dash=(4, 4),
        )

    def _clear_preview(self):
        if self.state.preview_id is not None:
            self.canvas.delete(self.state.preview_id)
            self.state.preview_id = None

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _new_map(self):
        if not self._confirm_discard():
            return
        self.map_data = new_empty_map()
        self.file_path = None
        self.state.reset()
        self._render_map()
        self._update_sidebar()
        self.root.title("Semantic Navigation Map Editor")

    def _open_file(self):
        if not self._confirm_discard():
            return
        path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON map", "*.json"), ("All files", "*.*")],
        )
        if path:
            self._load_file(Path(path))

    def _load_file(self, path: Path):
        try:
            self.map_data = MapData.from_json(path)
            self.file_path = path
            self.state.reset()
            self._render_map()
            self._update_sidebar()
            self.root.title(f"Semantic Navigation Map Editor - {path.name}")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            messagebox.showerror("Open Error", f"Failed to load map:\n{e}")

    def _save_file(self):
        if self.file_path is None:
            self._save_as_file()
        else:
            self.map_data.to_json(self.file_path)
            self.status.config(text=f"Saved {self.file_path.name}")

    def _save_as_file(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON map", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.file_path = Path(path)
            self._save_file()
            self.root.title(f"Semantic Navigation Map Editor - {self.file_path.name}")

    def _clear_map(self):
        if messagebox.askyesno("Clear Map", "Delete all walls and rooms?"):
            self.map_data.walls = []
            self.map_data.rooms = []
            self.map_data.apriltags = []
            self.map_data.obstacles = []
            self.state.reset()
            self._render_map()
            self._update_sidebar()

    def _confirm_discard(self) -> bool:
        # In a production editor you'd track dirty state. For now, always confirm.
        return messagebox.askyesno(
            "Discard Changes?",
            "Any unsaved changes will be lost. Continue?",
        )


def main():
    root = tk.Tk()
    initial_map = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    MapEditor(root, initial_map)
    root.mainloop()


if __name__ == "__main__":
    main()
