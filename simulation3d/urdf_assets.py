"""Generate URDF assets for the 3D sim from the robot config (T3D-01, T3D-02).

The mobile base's URDF is *generated* from ``robot.yaml`` (via
``robot_config``) rather than hand-written, so the 3D body always matches the
dimensions the brains already use — no duplicated constants.  A different base
is a ``robot.yaml`` edit + regenerate.

Layout (2 driven wheels + 2 free wheels, per the agreed design):
    - base_link: a box centred at the chassis centre.
    - wheel_drive_L / wheel_drive_R: powered mid-side wheels (axis along body Y),
      separated by ``wheel_track_m``.
    - wheel_free_F / wheel_free_B: free-spinning wheels on the front/back
      centre-line (axis along body Y), separated by ``wheelbase_m``, keeping
      the chassis level.
    - arm_mount: a pad on top of the chassis where the arm's ``g_base`` will be
      constrained (T3D-02).

The arm URDF (T3D-02) is rebuilt as pure primitives (no meshes) so PyBullet can
load it reliably: every link gets a representative box (visual + collision)
plus an inertial, and a simple two-finger gripper is appended.  The per-link
box geometry was baked once from the original COLLADA meshes (see
``_ARM_PRIMITIVES``), so regeneration does not depend on the mesh files.

Friction and other contact dynamics are applied at load time in PyBullet
(T3D-04), not in the URDF.
"""

from __future__ import annotations

import math
from typing import Tuple

from core.robot_config import RobotConfig


def _box_inertia(mass: float, l: float, w: float, h: float) -> Tuple[float, float, float]:
    """(ixx, iyy, izz) principal moments for a box of mass m, dims (l, w, h)."""
    return (
        mass / 12.0 * (w * w + h * h),
        mass / 12.0 * (l * l + h * h),
        mass / 12.0 * (l * l + w * w),
    )


def _cylinder_inertia(mass: float, radius: float, length: float) -> Tuple[float, float, float]:
    """(ixx, iyy, izz) for a cylinder, axis along Z (izz = spin axis)."""
    i_axis = 0.5 * mass * radius * radius
    i_trans = (1.0 / 12.0) * mass * (3.0 * radius * radius + length * length)
    return (i_trans, i_trans, i_axis)


def generate_base_urdf(cfg: RobotConfig) -> str:
    """Return the base URDF as XML text for the given robot config."""
    ch = cfg.chassis

    # Absolute geometry (metres).  Ground is z=0; wheels rest on it.
    wheel_z = ch.wheel_radius_m  # wheel centre height so the tyre bottom is at 0
    base_z = wheel_z + ch.height_m / 2.0  # chassis centre height
    wheel_z_rel = wheel_z - base_z  # wheel centre relative to base_link frame

    ixx, iyy, izz = _box_inertia(ch.mass_kg, ch.footprint_m, ch.footprint_m, ch.height_m)
    parts = [
        f'<?xml version="1.0"?>',
        f'<robot name="{cfg.name}_base">',
        # ---------------- base_link ----------------
        f'  <link name="base_link">',
        f'    <inertial>',
        f'      <origin xyz="0 0 {ch.cg_offset_m}" rpy="0 0 0"/>',
        f'      <mass value="{ch.mass_kg}"/>',
        f'      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>',
        f'    </inertial>',
        f'    <visual>',
        f'      <origin xyz="0 0 0" rpy="0 0 0"/>',
        f'      <geometry><box size="{ch.footprint_m} {ch.footprint_m} {ch.height_m}"/></geometry>',
        f'    </visual>',
        f'    <collision>',
        f'      <origin xyz="0 0 0" rpy="0 0 0"/>',
        f'      <geometry><box size="{ch.footprint_m} {ch.footprint_m} {ch.height_m}"/></geometry>',
        f'    </collision>',
        f'  </link>',
    ]

    # ---------------- wheels ----------------
    w_ixx, w_iyy, w_izz = _cylinder_inertia(0.4, ch.wheel_radius_m, ch.wheel_width_m)

    def _wheel_link(name: str, cx: float, cy: float):
        # The cylinder's axis is along the link Z.  The joint's rpy rotates the
        # child frame about X so the cylinder axis lies horizontal (along body
        # Y); the joint axis (0 1 0) then spins the wheel about its own axis.
        parts.extend(
            [
                f'  <link name="{name}">',
                f'    <inertial>',
                f'      <origin xyz="0 0 0" rpy="0 0 0"/>',
                f'      <mass value="0.4"/>',
                f'      <inertia ixx="{w_ixx}" ixy="0" ixz="0" iyy="{w_iyy}" iyz="0" izz="{w_izz}"/>',
                f'    </inertial>',
                f'    <visual>',
                f'      <origin xyz="0 0 0" rpy="0 0 0"/>',
                f'      <geometry><cylinder radius="{ch.wheel_radius_m}" length="{ch.wheel_width_m}"/></geometry>',
                f'    </visual>',
                f'    <collision>',
                f'      <origin xyz="0 0 0" rpy="0 0 0"/>',
                f'      <geometry><cylinder radius="{ch.wheel_radius_m}" length="{ch.wheel_width_m}"/></geometry>',
                f'    </collision>',
                f'  </link>',
                f'  <joint name="{name}_joint" type="continuous">',
                f'    <parent link="base_link"/>',
                f'    <child link="{name}"/>',
                f'    <origin xyz="{cx} {cy} {wheel_z_rel}" rpy="1.5708 0 0"/>',
                # PyBullet interprets the axis in the CHILD frame.  The joint
                # origin rotates the child so its Z (the cylinder axle) points
                # along body Y; axis -Z makes a positive joint velocity roll the
                # wheel forward (+x), matching the unicycle->wheel kinematics.
                f'    <axis xyz="0 0 -1"/>',
                f'  </joint>',
            ]
        )

    def _caster_wheel_link(name: str, cx: float, cy: float):
        # A swivelling (caster) free wheel: a fork that rotates about the
        # vertical axis at the corner, carrying a wheel that rolls about its
        # horizontal axle.  Lets the base turn cleanly instead of scrubbing.
        parts.extend(
            [
                # ---- fork (swivels about vertical Z through the corner) ----
                f'  <link name="{name}_fork">',
                f'    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.05"/>'
                f'<inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/></inertial>',
                f'    <visual><origin xyz="0 0 0" rpy="0 0 0"/>'
                f'<geometry><box size="0.04 0.04 0.10"/></geometry></visual>',
                f'    <collision><origin xyz="0 0 0" rpy="0 0 0"/>'
                f'<geometry><box size="0.04 0.04 0.10"/></geometry></collision>',
                f'  </link>',
                f'  <joint name="{name}_fork_joint" type="continuous">',
                f'    <parent link="base_link"/>',
                f'    <child link="{name}_fork"/>',
                f'    <origin xyz="{cx} {cy} {wheel_z_rel}" rpy="0 0 0"/>',
                f'    <axis xyz="0 0 1"/>',
                f'  </joint>',
                # ---- wheel (rolls about its horizontal axle, held by the fork) ----
                f'  <link name="{name}">',
                f'    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.4"/>'
                f'<inertia ixx="{w_ixx}" ixy="0" ixz="0" iyy="{w_iyy}" iyz="0" izz="{w_izz}"/></inertial>',
                f'    <visual><origin xyz="0 0 0" rpy="0 0 0"/>'
                f'<geometry><cylinder radius="{ch.wheel_radius_m}" length="{ch.wheel_width_m}"/></geometry></visual>',
                f'    <collision><origin xyz="0 0 0" rpy="0 0 0"/>'
                f'<geometry><cylinder radius="{ch.wheel_radius_m}" length="{ch.wheel_width_m}"/></geometry></collision>',
                f'  </link>',
                f'  <joint name="{name}_joint" type="continuous">',
                f'    <parent link="{name}_fork"/>',
                f'    <child link="{name}"/>',
                # Wheel sits behind the fork swivel axis (trail) so the caster
                # self-aligns and rolls tangentially while turning; a centre-pivot
                # caster (wheel under the axis) just skids sideways instead.
                f'    <origin xyz="-0.06 0 0" rpy="1.5708 0 0"/>',
                f'    <axis xyz="0 0 -1"/>',
                f'  </joint>',
            ]
        )

    # Corner wheel layout (T3D-01): all four wheels at the base corners,
    # rolling forward (axis along body Y).  The rear pair (x = -corner_x) is
    # driven (fixed wheels); the front pair (x = +corner_x) are swivelling
    # casters so the base can turn without the wheels scrubbing.  Facing +x,
    # +y is the LEFT side (positive yaw = CCW), so the ``_L`` wheels sit at
    # +corner_y and ``_R`` at -corner_y.  (This was the wrong way around: the
    # ``_L``/``_R`` wheels were physically swapped, which inverted every
    # angular command coming from the drive stack.)
    corner_x = ch.wheelbase_m / 2.0  # front/back wheel x-offset
    corner_y = ch.wheel_track_m / 2.0  # left/right wheel y-offset
    _wheel_link("wheel_drive_L", -corner_x, corner_y)  # rear-left  (driven)
    _wheel_link("wheel_drive_R", -corner_x, -corner_y)  # rear-right (driven)
    _caster_wheel_link("wheel_free_L", corner_x, corner_y)  # front-left  (caster)
    _caster_wheel_link("wheel_free_R", corner_x, -corner_y)  # front-right (caster)

    # ---------------- arm mount ----------------
    mount_h = 0.01
    parts.extend(
        [
            f'  <link name="arm_mount">',
            f'    <inertial>',
            f'      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <mass value="0.1"/>',
            f'      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>',
            f'    </inertial>',
            f'    <visual>',
            f'      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry><box size="0.10 0.10 {mount_h}"/></geometry>',
            f'    </visual>',
            f'    <collision>',
            f'      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry><box size="0.10 0.10 {mount_h}"/></geometry>',
            f'    </collision>',
            f'  </link>',
            # The mount sits on the top face of the chassis (base_link centre is
            # the chassis centre, so the top face is at +height/2).
            f'  <joint name="arm_mount_joint" type="fixed">',
            f'    <parent link="base_link"/>',
            f'    <child link="arm_mount"/>',
            f'    <origin xyz="0 0 {ch.height_m / 2.0}" rpy="0 0 0"/>',
            f'  </joint>',
            f'</robot>',
        ]
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Arm URDF (T3D-02) — primitive-only, so PyBullet can load it without meshes.
# ---------------------------------------------------------------------------

# Mass (kg) per arm link, from the T3D-02 spec.
_ARM_LINK_MASS = {
    "g_base": 0.8,
    "joint1": 0.5,
    "joint2": 0.6,
    "joint3": 0.5,
    "joint4": 0.4,
    "joint5": 0.3,
    "joint6": 0.2,
    "joint6_flange": 0.1,
}

# Representative box per link: (size, centre, rpy) in the link frame.  Baked
# once from the original COLLADA mesh AABBs (mm->m), so the generator does not
# depend on the mesh files.  "Representative", not exact — per the agreed design.
_ARM_PRIMITIVES = {
    "g_base": ((0.15, 0.0174, 0.11), (0.0073, 0.0, -0.03), (0.0, 0.0, 1.5708)),
    "joint1": ((0.0926, 0.0985, 0.0272), (0.122, 0.0013, -0.1283), (0.0, 0.0, 0.0)),
    "joint2": ((0.0605, 0.048, 0.0871), (0.0, -0.0063, -0.0207), (0.0, 0.0, -1.5708)),
    "joint3": ((0.0554, 0.048, 0.156), (-0.054, 0.0, 0.0576), (0.0, -1.5708, 0.0)),
    "joint4": ((0.0578, 0.0432, 0.1368), (-0.0468, 0.0, 0.0065), (0.0, -1.5708, 0.0)),
    "joint5": ((0.0474, 0.0386, 0.0363), (0.0, -0.1984, -0.0935), (0.0, -1.5708, 1.5708)),
    "joint6": ((0.0384, 0.0567, 0.0571), (0.0, 0.0067, -0.0094), (0.0, 0.0, 0.0)),
    "joint6_flange": ((0.0384, 0.0384, 0.0137), (0.0, 0.0, -0.0051), (0.0, 0.0, 0.0)),
}

# Revolute joints (name, parent, child, origin xyz, origin rpy, lower, upper).
_ARM_JOINTS = [
    ("g_base_to_joint1", "g_base", "joint1", (0, 0, 0), (0, 0, 0), "fixed"),
    ("joint2_to_joint1", "joint1", "joint2", (0, 0, 0.13956), (0, 0, 0), "revolute"),
    ("joint3_to_joint2", "joint2", "joint3", (0, 0, -0.001), (0, 1.5708, -1.5708), "revolute"),
    ("joint4_to_joint3", "joint3", "joint4", (-0.1104, 0, 0), (0, 0, 0), "revolute"),
    ("joint5_to_joint4", "joint4", "joint5", (-0.096, 0, 0.06462), (0, 0, -1.5708), "revolute"),
    ("joint6_to_joint5", "joint5", "joint6", (0, -0.07318, -0.001), (1.5708, -1.5708, 0), "revolute"),
    ("joint6output_to_joint6", "joint6", "joint6_flange", (0, 0.0456, 0), (-1.5708, 0, 0), "revolute"),
]

_ARM_JOINT_LIMITS = {
    "joint2_to_joint1": (-2.9321, 2.9321),
    "joint3_to_joint2": (-2.4434, 2.4434),
    "joint4_to_joint3": (-2.6179, 2.6179),
    "joint5_to_joint4": (-2.6179, 2.6179),
    "joint6_to_joint5": (-2.7052, 2.7925),
    "joint6output_to_joint6": (-3.14, 3.14159),
}


def _box_inertial(size: Tuple[float, float, float], mass: float) -> str:
    """An <inertial> for a box of `size` (l, w, h) and `mass`, at its centre."""
    sx, sy, sz = size
    ixx = mass / 12.0 * (sy * sy + sz * sz)
    iyy = mass / 12.0 * (sx * sx + sz * sz)
    izz = mass / 12.0 * (sx * sx + sy * sy)
    return (
        f'    <inertial>\n'
        f'      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        f'      <mass value="{mass}"/>\n'
        f'      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>\n'
        f'    </inertial>'
    )


def generate_arm_urdf(cfg: RobotConfig) -> str:
    """Return the arm URDF as XML text (primitive-only, plus a two-finger
    gripper).  Mounted by constraining ``g_base`` to the base's ``arm_mount``."""
    parts = [
        f'<?xml version="1.0"?>',
        f'<robot name="{cfg.name}_arm">',
    ]

    # --- arm links: primitive box visual + collision + inertial ---
    for link, (size, center, rpy) in _ARM_PRIMITIVES.items():
        mass = _ARM_LINK_MASS[link]
        cx, cy, cz = center
        rx, ry, rz = rpy
        parts.extend(
            [
                f'  <link name="{link}">',
                _box_inertial(size, mass),
                f'    <visual>',
                f'      <origin xyz="{cx} {cy} {cz}" rpy="{rx} {ry} {rz}"/>',
                f'      <geometry><box size="{size[0]} {size[1]} {size[2]}"/></geometry>',
                f'    </visual>',
                f'    <collision>',
                f'      <origin xyz="{cx} {cy} {cz}" rpy="{rx} {ry} {rz}"/>',
                f'      <geometry><box size="{size[0]} {size[1]} {size[2]}"/></geometry>',
                f'    </collision>',
                f'  </link>',
            ]
        )

    # --- joints ---
    for name, parent, child, xyz, rpy, jtype in _ARM_JOINTS:
        origin = f'<origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>'
        if jtype == "fixed":
            # No <axis>/<limit> on a fixed joint (strict parsers reject them).
            parts.extend(
                [
                    f'  <joint name="{name}" type="fixed">',
                    f'    <parent link="{parent}"/>',
                    f'    <child link="{child}"/>',
                    f'    {origin}',
                    f'  </joint>',
                ]
            )
        else:
            lower, upper = _ARM_JOINT_LIMITS[name]
            parts.extend(
                [
                    f'  <joint name="{name}" type="revolute">',
                    f'    <axis xyz="0 0 1"/>',
                    f'    <limit effort="1000.0" lower="{lower}" upper="{upper}" velocity="0"/>',
                    f'    <parent link="{parent}"/>',
                    f'    <child link="{child}"/>',
                    f'    {origin}',
                    f'  </joint>',
                ]
            )

    # --- two-finger gripper attached to joint6_flange ---
    # gripper_link: small box fixed to the flange, extending +z.
    parts.extend(
        [
            f'  <link name="gripper_link">',
            _box_inertial((0.06, 0.06, 0.02), 0.08),
            f'    <visual><origin xyz="0 0 0" rpy="0 0 0"/>'
            f'<geometry><box size="0.06 0.06 0.02"/></geometry></visual>',
            f'    <collision><origin xyz="0 0 0" rpy="0 0 0"/>'
            f'<geometry><box size="0.06 0.06 0.02"/></geometry></collision>',
            f'  </link>',
            f'  <joint name="gripper_to_flange" type="fixed">',
            f'    <parent link="joint6_flange"/>',
            f'    <child link="gripper_link"/>',
            f'    <origin xyz="0 0 0.02" rpy="0 0 0"/>',
            f'  </joint>',
        ]
    )

    # Fingers: thin vertical plates at y=±0.012, prismatic along gripper X.
    # Left moves +X and right moves -X as the joint position rises 0 -> 0.04,
    # so the fingers close toward the centre.  (Exact grasp is T3D-09.)
    for side, y, axis in (("left", -0.012, 1), ("right", 0.012, -1)):
        parts.extend(
            [
                f'  <link name="{side}_finger">',
                _box_inertial((0.05, 0.012, 0.12), 0.03),
                f'    <visual><origin xyz="0 0 0" rpy="0 0 0"/>'
                f'<geometry><box size="0.05 0.012 0.12"/></geometry></visual>',
                f'    <collision><origin xyz="0 0 0" rpy="0 0 0"/>'
                f'<geometry><box size="0.05 0.012 0.12"/></geometry></collision>',
                f'  </link>',
                f'  <joint name="gripper_joint_{side}" type="prismatic">',
                f'    <axis xyz="{axis} 0 0"/>',
                f'    <limit effort="10.0" lower="0.0" upper="0.04" velocity="0.05"/>',
                f'    <parent link="gripper_link"/>',
                f'    <child link="{side}_finger"/>',
                f'    <origin xyz="0 {y} 0" rpy="0 0 0"/>',
                f'  </joint>',
                # fingertip pad (contact patch) at the bottom of the finger.
                f'  <link name="{side}_finger_tip">',
                _box_inertial((0.012, 0.012, 0.01), 0.01),
                f'    <visual><origin xyz="0 0 0" rpy="0 0 0"/>'
                f'<geometry><box size="0.012 0.012 0.01"/></geometry></visual>',
                f'    <collision><origin xyz="0 0 0" rpy="0 0 0"/>'
                f'<geometry><box size="0.012 0.012 0.01"/></geometry></collision>',
                f'  </link>',
                f'  <joint name="gripper_{side}_tip_joint" type="fixed">',
                f'    <parent link="{side}_finger"/>',
                f'    <child link="{side}_finger_tip"/>',
                f'    <origin xyz="0 0 -0.065" rpy="0 0 0"/>',
                f'  </joint>',
            ]
        )

    parts.append("</robot>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Graspable objects (T3D-03) — single-link primitive URDFs, parametrised by
# the world registry so size/mass/colour come from world.json, not hand files.
# ---------------------------------------------------------------------------


def _material(name: str, color) -> str:
    """A <material> with an RGBA colour (PyBullet renders primitive visuals)."""
    r, g, b = color[:3]
    return (
        f'<material name="{name}"><color rgba="{r} {g} {b} 1.0"/></material>'
    )


def generate_object_urdf(
    shape: str,
    size_m: float,
    mass_kg: float,
    color=(0.7, 0.7, 0.7, 1.0),
) -> str:
    """Return a single-link primitive URDF for a graspable object.

    ``shape`` is ``cube`` (box ``size_m`` per side), ``cylinder`` (radius
    ``size_m/2``, height ``size_m``, axis Z) or ``ball`` (sphere radius
    ``size_m/2``).  All geometry/inertia/colour are primitive-only.
    """
    if shape == "cube":
        geo = f'<box size="{size_m} {size_m} {size_m}"/>'
        s = size_m
        ixx = iyy = izz = mass_kg / 12.0 * (s * s + s * s)
    elif shape == "cylinder":
        r = size_m / 2.0
        h = size_m
        geo = f'<cylinder radius="{r}" length="{h}"/>'
        ixx = iyy = (1.0 / 12.0) * mass_kg * (3.0 * r * r + h * h)
        izz = 0.5 * mass_kg * r * r
    elif shape == "ball":
        r = size_m / 2.0
        geo = f'<sphere radius="{r}"/>'
        i = 0.4 * mass_kg * r * r
        ixx = iyy = izz = i
    else:
        raise ValueError(f"unknown object shape: {shape!r}")

    return "\n".join(
        [
            '<?xml version="1.0"?>',
            f'<robot name="{shape}_{size_m}">',
            f'  <link name="{shape}">',
            f'    <inertial>',
            f'      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <mass value="{mass_kg}"/>',
            f'      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>',
            f'    </inertial>',
            f'    <visual>',
            f'      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry>{geo}</geometry>',
            f'      {_material("mat", color)}',
            f'    </visual>',
            f'    <collision>',
            f'      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry>{geo}</geometry>',
            f'    </collision>',
            f'  </link>',
            f'</robot>',
        ]
    )
