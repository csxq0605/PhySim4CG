"""
Taichi Lab 1 - complex rigid body scene.

Features:
- four rigid boxes with random initial state
- gravity + floor + four fixed walls
- LMB drag applies force at an off-center local point on the selected body
- RMB drag rotates camera, WASDQE moves camera
- TAB cycles the currently selected body
- R resets the random scene
- SPACE pauses / resumes
"""

import numpy as np
import taichi as ti


ti.init(arch=ti.gpu, default_fp=ti.f32)


N_BODIES = 4
FRAME_DT = 1.0 / 60.0
SUBSTEPS = 3
SOLVER_ITERS = 1
DT = FRAME_DT / SUBSTEPS
GRAVITY = np.array([0.0, -9.8, 0.0], dtype=np.float32)
LINEAR_DAMPING = 0.03
ANGULAR_DAMPING = 0.05
BODY_RESTITUTION = 0.45
BODY_FRICTION = 0.18
PLANE_RESTITUTION = 0.25
PLANE_FRICTION = 0.30
EPSILON = 1e-6
POSITION_CORRECTION_PERCENT = 0.65
POSITION_CORRECTION_SLOP = 0.002
DRAG_FORCE_SCALE = 1200.0
FORCE_LINE_SCALE = 0.0035
CAMERA_MOVE_SPEED = 2.8
CAMERA_ROTATE_SPEED = 2.4

ARENA_HALF_X = 2.25
ARENA_HALF_Z = 2.25
ARENA_HEIGHT = 3.2
FLOOR_Y = 0.0


CUBE_LOCAL_VERTICES = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float32,
)

CUBE_INDICES = np.array(
    [
        0,
        1,
        2,
        0,
        2,
        3,
        4,
        5,
        6,
        4,
        6,
        7,
        0,
        3,
        7,
        0,
        7,
        4,
        1,
        2,
        6,
        1,
        6,
        5,
        0,
        4,
        5,
        0,
        5,
        1,
        3,
        2,
        6,
        3,
        6,
        7,
    ],
    dtype=np.int32,
)

WORLD_AXIS_LINES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.4, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.4, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.4],
    ],
    dtype=np.float32,
)

ARENA_LINE_VERTICES = np.array(
    [
        [-ARENA_HALF_X, FLOOR_Y, -ARENA_HALF_Z],
        [ARENA_HALF_X, FLOOR_Y, -ARENA_HALF_Z],
        [ARENA_HALF_X, FLOOR_Y, -ARENA_HALF_Z],
        [ARENA_HALF_X, FLOOR_Y, ARENA_HALF_Z],
        [ARENA_HALF_X, FLOOR_Y, ARENA_HALF_Z],
        [-ARENA_HALF_X, FLOOR_Y, ARENA_HALF_Z],
        [-ARENA_HALF_X, FLOOR_Y, ARENA_HALF_Z],
        [-ARENA_HALF_X, FLOOR_Y, -ARENA_HALF_Z],
        [-ARENA_HALF_X, FLOOR_Y, -ARENA_HALF_Z],
        [-ARENA_HALF_X, ARENA_HEIGHT, -ARENA_HALF_Z],
        [ARENA_HALF_X, FLOOR_Y, -ARENA_HALF_Z],
        [ARENA_HALF_X, ARENA_HEIGHT, -ARENA_HALF_Z],
        [ARENA_HALF_X, FLOOR_Y, ARENA_HALF_Z],
        [ARENA_HALF_X, ARENA_HEIGHT, ARENA_HALF_Z],
        [-ARENA_HALF_X, FLOOR_Y, ARENA_HALF_Z],
        [-ARENA_HALF_X, ARENA_HEIGHT, ARENA_HALF_Z],
        [-ARENA_HALF_X, ARENA_HEIGHT, -ARENA_HALF_Z],
        [ARENA_HALF_X, ARENA_HEIGHT, -ARENA_HALF_Z],
        [ARENA_HALF_X, ARENA_HEIGHT, -ARENA_HALF_Z],
        [ARENA_HALF_X, ARENA_HEIGHT, ARENA_HALF_Z],
        [ARENA_HALF_X, ARENA_HEIGHT, ARENA_HALF_Z],
        [-ARENA_HALF_X, ARENA_HEIGHT, ARENA_HALF_Z],
        [-ARENA_HALF_X, ARENA_HEIGHT, ARENA_HALF_Z],
        [-ARENA_HALF_X, ARENA_HEIGHT, -ARENA_HALF_Z],
    ],
    dtype=np.float32,
)

FLOOR_VERTICES = np.array(
    [
        [-ARENA_HALF_X, FLOOR_Y, -ARENA_HALF_Z],
        [ARENA_HALF_X, FLOOR_Y, -ARENA_HALF_Z],
        [ARENA_HALF_X, FLOOR_Y, ARENA_HALF_Z],
        [-ARENA_HALF_X, FLOOR_Y, ARENA_HALF_Z],
    ],
    dtype=np.float32,
)
FLOOR_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)

PLANE_POINTS = [
    np.array([0.0, FLOOR_Y, 0.0], dtype=np.float32),
    np.array([-ARENA_HALF_X, 0.0, 0.0], dtype=np.float32),
    np.array([ARENA_HALF_X, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, -ARENA_HALF_Z], dtype=np.float32),
    np.array([0.0, 0.0, ARENA_HALF_Z], dtype=np.float32),
]
PLANE_NORMALS = [
    np.array([0.0, 1.0, 0.0], dtype=np.float32),
    np.array([1.0, 0.0, 0.0], dtype=np.float32),
    np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, 1.0], dtype=np.float32),
    np.array([0.0, 0.0, -1.0], dtype=np.float32),
]
PLANE_CONTACTS = list(zip(PLANE_POINTS, PLANE_NORMALS))


cube_local_verts = ti.Vector.field(3, dtype=ti.f32, shape=8)
mesh_vertices = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES * 8)
mesh_indices = ti.field(dtype=ti.i32, shape=N_BODIES * 36)
axis_lines = ti.Vector.field(3, dtype=ti.f32, shape=6)
arena_line_verts = ti.Vector.field(3, dtype=ti.f32, shape=24)
floor_vertices_ti = ti.Vector.field(3, dtype=ti.f32, shape=4)
floor_indices_ti = ti.field(dtype=ti.i32, shape=6)
selected_com_vis = ti.Vector.field(3, dtype=ti.f32, shape=1)
force_anchor_vis = ti.Vector.field(3, dtype=ti.f32, shape=1)
force_line = ti.Vector.field(3, dtype=ti.f32, shape=2)

position = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
orientation = ti.Vector.field(4, dtype=ti.f32, shape=N_BODIES)
half_extent = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
applied_force = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
force_local_point = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)

cube_local_verts.from_numpy(CUBE_LOCAL_VERTICES)
axis_lines.from_numpy(WORLD_AXIS_LINES)
arena_line_verts.from_numpy(ARENA_LINE_VERTICES)
floor_vertices_ti.from_numpy(FLOOR_VERTICES)
floor_indices_ti.from_numpy(FLOOR_INDICES)


state_position = np.zeros((N_BODIES, 3), dtype=np.float32)
state_velocity = np.zeros((N_BODIES, 3), dtype=np.float32)
state_angular_velocity = np.zeros((N_BODIES, 3), dtype=np.float32)
state_half_extent = np.zeros((N_BODIES, 3), dtype=np.float32)
state_mass = np.ones(N_BODIES, dtype=np.float32)
state_inv_mass = np.ones(N_BODIES, dtype=np.float32)
state_orientation = np.zeros((N_BODIES, 4), dtype=np.float32)
state_orientation[:, 0] = 1.0
state_body_inertia_ref = np.zeros((N_BODIES, 3, 3), dtype=np.float32)
state_body_inertia_ref_inv = np.zeros((N_BODIES, 3, 3), dtype=np.float32)
state_applied_force = np.zeros((N_BODIES, 3), dtype=np.float32)
state_force_local_point = np.zeros((N_BODIES, 3), dtype=np.float32)


@ti.func
def quat_normalize(q):
    norm = ti.sqrt(q.dot(q))
    q_safe = q
    if norm < 1e-8:
        q_safe = ti.Vector([1.0, 0.0, 0.0, 0.0])
        norm = 1.0
    return q_safe / norm


@ti.func
def quat_to_matrix(q):
    q = quat_normalize(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return ti.Matrix(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    )


@ti.kernel
def update_mesh_vertices():
    for i in range(N_BODIES):
        pos = position[i]
        rot = quat_to_matrix(orientation[i])
        ext = half_extent[i]
        for k in range(8):
            lv = cube_local_verts[k]
            local = ti.Vector([lv[0] * ext[0], lv[1] * ext[1], lv[2] * ext[2]])
            mesh_vertices[i * 8 + k] = rot @ local + pos


@ti.kernel
def update_selected_visuals(selected_idx: ti.i32):
    pos = position[selected_idx]
    rot = quat_to_matrix(orientation[selected_idx])
    selected_com_vis[0] = pos
    force_anchor_vis[0] = pos + rot @ force_local_point[selected_idx]
    force_line[0] = force_anchor_vis[0]
    force_line[1] = force_anchor_vis[0] + applied_force[selected_idx] * FORCE_LINE_SCALE


def build_mesh_indices():
    indices = np.zeros(N_BODIES * 36, dtype=np.int32)
    for body_id in range(N_BODIES):
        start = body_id * 36
        indices[start : start + 36] = CUBE_INDICES + body_id * 8
    mesh_indices.from_numpy(indices)


def upload_render_state():
    position.from_numpy(state_position)
    orientation.from_numpy(state_orientation)
    half_extent.from_numpy(state_half_extent)
    applied_force.from_numpy(state_applied_force)
    force_local_point.from_numpy(state_force_local_point)


def safe_normalize(v: np.ndarray):
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        fallback = np.zeros_like(v, dtype=np.float32)
        fallback[0] = 1.0
        return fallback
    return (v / norm).astype(np.float32)


def quat_normalize_np(q: np.ndarray):
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def quat_mul_np(q1: np.ndarray, q2: np.ndarray):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def axis_angle_to_quat(axis: np.ndarray, angle: float):
    axis = safe_normalize(axis)
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)


def quat_to_matrix_np(q: np.ndarray):
    q = quat_normalize_np(q)
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def box_inertia_tensor(mass_value: float, ext: np.ndarray):
    size = 2.0 * ext
    ixx = mass_value * (size[1] ** 2 + size[2] ** 2) / 12.0
    iyy = mass_value * (size[0] ** 2 + size[2] ** 2) / 12.0
    izz = mass_value * (size[0] ** 2 + size[1] ** 2) / 12.0
    inertia = np.diag([ixx, iyy, izz]).astype(np.float32)
    inertia_inv = np.diag([1.0 / ixx, 1.0 / iyy, 1.0 / izz]).astype(np.float32)
    return inertia, inertia_inv


def get_rotation_matrix(body_id: int):
    return quat_to_matrix_np(state_orientation[body_id])


def get_box_vertices(body_id: int):
    pos = state_position[body_id]
    rot = get_rotation_matrix(body_id)
    ext = state_half_extent[body_id]
    verts = np.zeros((8, 3), dtype=np.float32)
    for k in range(8):
        verts[k] = rot @ (CUBE_LOCAL_VERTICES[k] * ext) + pos
    return verts


def collision_manifold(i: int, j: int):
    verts_a = get_box_vertices(i)
    verts_b = get_box_vertices(j)
    rot_a = get_rotation_matrix(i)
    rot_b = get_rotation_matrix(j)

    axes_a = [rot_a[:, k] for k in range(3)]
    axes_b = [rot_b[:, k] for k in range(3)]
    axes = axes_a + axes_b

    for axis_a in axes_a:
        for axis_b in axes_b:
            cross_axis = np.cross(axis_a, axis_b)
            if np.dot(cross_axis, cross_axis) > EPSILON:
                axes.append(safe_normalize(cross_axis))

    center_a = state_position[i]
    center_b = state_position[j]
    min_overlap = np.inf
    best_axis = None

    for axis in axes:
        axis = safe_normalize(axis)
        proj_a = verts_a @ axis
        proj_b = verts_b @ axis
        min_a, max_a = proj_a.min(), proj_a.max()
        min_b, max_b = proj_b.min(), proj_b.max()

        if max_a < min_b - EPSILON or max_b < min_a - EPSILON:
            return False, None, 0.0, None

        overlap = min(max_a - min_b, max_b - min_a)
        if overlap < min_overlap:
            min_overlap = overlap
            d = center_b - center_a
            if np.dot(axis, d) < 0.0:
                axis = -axis
            best_axis = axis

    if best_axis is None:
        return False, None, 0.0, None

    normal = safe_normalize(best_axis)
    idx_a = np.argmax(verts_a @ normal)
    idx_b = np.argmin(verts_b @ normal)
    contact = 0.5 * (verts_a[idx_a] + verts_b[idx_b])
    return True, normal, float(min_overlap), contact.astype(np.float32)


def impulse_denominator(inv_mass_value: float, inv_inertia: np.ndarray, r: np.ndarray, direction: np.ndarray):
    angular_term = np.cross(inv_inertia @ np.cross(r, direction), r)
    return inv_mass_value + np.dot(angular_term, direction)


def integrate_free_motion_cpu():
    linear_decay = np.exp(-LINEAR_DAMPING * DT).astype(np.float32)
    angular_decay = np.exp(-ANGULAR_DAMPING * DT).astype(np.float32)

    for body_id in range(N_BODIES):
        linear_acc = GRAVITY + state_applied_force[body_id] * state_inv_mass[body_id]
        state_velocity[body_id] += linear_acc * DT
        state_velocity[body_id] *= linear_decay
        state_position[body_id] += state_velocity[body_id] * DT

        q = quat_normalize_np(state_orientation[body_id])
        rot = quat_to_matrix_np(q)
        omega = state_angular_velocity[body_id]
        torque = np.cross(rot @ state_force_local_point[body_id], state_applied_force[body_id])
        inertia_world = rot @ state_body_inertia_ref[body_id] @ rot.T
        inertia_world_inv = rot @ state_body_inertia_ref_inv[body_id] @ rot.T

        angular_acc = inertia_world_inv @ (torque - np.cross(omega, inertia_world @ omega))
        omega = omega + angular_acc * DT
        omega = omega * angular_decay
        state_angular_velocity[body_id] = omega.astype(np.float32)

        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=np.float32)
        q = q + 0.5 * DT * quat_mul_np(omega_quat, q)
        state_orientation[body_id] = quat_normalize_np(q)


def resolve_body_body_impulse(i: int, j: int, normal: np.ndarray, penetration: float, contact: np.ndarray):
    pos_a = state_position[i]
    pos_b = state_position[j]
    vel_a = state_velocity[i]
    vel_b = state_velocity[j]
    omega_a = state_angular_velocity[i]
    omega_b = state_angular_velocity[j]

    rot_a = get_rotation_matrix(i)
    rot_b = get_rotation_matrix(j)
    inv_inertia_a = rot_a @ state_body_inertia_ref_inv[i] @ rot_a.T
    inv_inertia_b = rot_b @ state_body_inertia_ref_inv[j] @ rot_b.T

    inv_mass_a = float(state_inv_mass[i])
    inv_mass_b = float(state_inv_mass[j])
    if inv_mass_a + inv_mass_b < EPSILON:
        return

    r_a = contact - pos_a
    r_b = contact - pos_b
    v_contact_a = vel_a + np.cross(omega_a, r_a)
    v_contact_b = vel_b + np.cross(omega_b, r_b)
    relative_velocity = v_contact_b - v_contact_a

    vel_along_normal = float(np.dot(relative_velocity, normal))
    if vel_along_normal > 0.0:
        return

    denom_n = (
        impulse_denominator(inv_mass_a, inv_inertia_a, r_a, normal)
        + impulse_denominator(inv_mass_b, inv_inertia_b, r_b, normal)
    )
    if denom_n < EPSILON:
        return

    j_n = -(1.0 + BODY_RESTITUTION) * vel_along_normal / denom_n
    impulse = j_n * normal

    tangent = relative_velocity - vel_along_normal * normal
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm > EPSILON:
        tangent = tangent / tangent_norm
        denom_t = (
            impulse_denominator(inv_mass_a, inv_inertia_a, r_a, tangent)
            + impulse_denominator(inv_mass_b, inv_inertia_b, r_b, tangent)
        )
        if denom_t > EPSILON:
            j_t = -np.dot(relative_velocity, tangent) / denom_t
            friction_limit = BODY_FRICTION * j_n
            j_t = np.clip(j_t, -friction_limit, friction_limit)
            impulse = impulse + j_t * tangent

    state_velocity[i] = (vel_a - inv_mass_a * impulse).astype(np.float32)
    state_velocity[j] = (vel_b + inv_mass_b * impulse).astype(np.float32)
    state_angular_velocity[i] = (omega_a - inv_inertia_a @ np.cross(r_a, impulse)).astype(np.float32)
    state_angular_velocity[j] = (omega_b + inv_inertia_b @ np.cross(r_b, impulse)).astype(np.float32)

    correction_mag = max(penetration - POSITION_CORRECTION_SLOP, 0.0)
    if correction_mag > 0.0:
        correction_mag *= POSITION_CORRECTION_PERCENT / (inv_mass_a + inv_mass_b)
        correction = correction_mag * normal
        state_position[i] = (pos_a - inv_mass_a * correction).astype(np.float32)
        state_position[j] = (pos_b + inv_mass_b * correction).astype(np.float32)


def resolve_body_plane_impulse(body_id: int, plane_point: np.ndarray, plane_normal: np.ndarray):
    pos = state_position[body_id]
    vel = state_velocity[body_id]
    omega = state_angular_velocity[body_id]
    rot = get_rotation_matrix(body_id)
    inv_inertia = rot @ state_body_inertia_ref_inv[body_id] @ rot.T
    inv_mass_value = float(state_inv_mass[body_id])

    verts = get_box_vertices(body_id)
    signed_dist = (verts - plane_point) @ plane_normal
    contact_idx = int(np.argmin(signed_dist))
    if signed_dist[contact_idx] >= 0.0:
        return

    contact = verts[contact_idx]
    penetration = -float(signed_dist[contact_idx])
    r = contact - pos
    v_contact = vel + np.cross(omega, r)
    vel_along_normal = float(np.dot(v_contact, plane_normal))

    if vel_along_normal < 0.0:
        denom_n = impulse_denominator(inv_mass_value, inv_inertia, r, plane_normal)
        if denom_n > EPSILON:
            j_n = -(1.0 + PLANE_RESTITUTION) * vel_along_normal / denom_n
            impulse = j_n * plane_normal

            tangent = v_contact - vel_along_normal * plane_normal
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm > EPSILON:
                tangent = tangent / tangent_norm
                denom_t = impulse_denominator(inv_mass_value, inv_inertia, r, tangent)
                if denom_t > EPSILON:
                    j_t = -np.dot(v_contact, tangent) / denom_t
                    friction_limit = PLANE_FRICTION * j_n
                    j_t = np.clip(j_t, -friction_limit, friction_limit)
                    impulse = impulse + j_t * tangent

            state_velocity[body_id] = (vel + inv_mass_value * impulse).astype(np.float32)
            state_angular_velocity[body_id] = (omega + inv_inertia @ np.cross(r, impulse)).astype(np.float32)

    correction_mag = max(penetration - POSITION_CORRECTION_SLOP, 0.0)
    if correction_mag > 0.0:
        correction = POSITION_CORRECTION_PERCENT * correction_mag * plane_normal
        state_position[body_id] = (pos + correction).astype(np.float32)


def clear_applied_forces_cpu():
    state_applied_force.fill(0.0)
    state_force_local_point.fill(0.0)


def set_applied_force_cpu(body_id: int, force: np.ndarray, local_point: np.ndarray):
    state_applied_force[body_id] = force.astype(np.float32)
    state_force_local_point[body_id] = local_point.astype(np.float32)


def step_simulation():
    for _ in range(SUBSTEPS):
        integrate_free_motion_cpu()

        for _ in range(SOLVER_ITERS):
            for i in range(N_BODIES):
                for j in range(i + 1, N_BODIES):
                    collided, normal, penetration, contact = collision_manifold(i, j)
                    if collided:
                        resolve_body_body_impulse(i, j, normal, penetration, contact)

            for body_id in range(N_BODIES):
                for plane_point, plane_normal in PLANE_CONTACTS:
                    resolve_body_plane_impulse(body_id, plane_point, plane_normal)


rng = np.random.default_rng()


def random_quaternion():
    axis = safe_normalize(rng.normal(size=3).astype(np.float32))
    angle = float(rng.uniform(0.0, np.pi))
    return axis_angle_to_quat(axis, angle)


def sample_nonoverlapping_positions(extents: np.ndarray):
    positions = []
    radii = 1.12 * np.linalg.norm(extents, axis=1)
    for i in range(N_BODIES):
        success = False
        for _ in range(200):
            candidate = np.array(
                [
                    rng.uniform(-0.85, 0.85),
                    rng.uniform(0.8, 2.4),
                    rng.uniform(-0.85, 0.85),
                ],
                dtype=np.float32,
            )
            ok = True
            for j, existing in enumerate(positions):
                if np.linalg.norm(candidate - existing) < radii[i] + radii[j]:
                    ok = False
                    break
            if ok:
                positions.append(candidate)
                success = True
                break
        if not success:
            row = i // 3
            col = i % 3
            positions.append(
                np.array(
                    [
                        -0.7 + 0.7 * col + rng.uniform(-0.08, 0.08),
                        1.0 + 0.45 * row + rng.uniform(-0.05, 0.05),
                        rng.uniform(-0.25, 0.25),
                    ],
                    dtype=np.float32,
                )
            )
    return np.stack(positions, axis=0).astype(np.float32)


def randomize_complex_scene():
    extents = rng.uniform(0.16, 0.28, size=(N_BODIES, 3)).astype(np.float32)
    masses = rng.uniform(0.8, 1.5, size=N_BODIES).astype(np.float32)
    quats = np.stack([random_quaternion() for _ in range(N_BODIES)], axis=0).astype(np.float32)
    ang_vel = rng.uniform(-0.9, 0.9, size=(N_BODIES, 3)).astype(np.float32)
    velocities = rng.uniform(-0.25, 0.25, size=(N_BODIES, 3)).astype(np.float32)
    velocities[:, 1] = rng.uniform(-0.15, 0.10, size=N_BODIES).astype(np.float32)
    positions = sample_nonoverlapping_positions(extents)

    inertias = np.zeros((N_BODIES, 3, 3), dtype=np.float32)
    inertias_inv = np.zeros((N_BODIES, 3, 3), dtype=np.float32)
    for i in range(N_BODIES):
        inertias[i], inertias_inv[i] = box_inertia_tensor(float(masses[i]), extents[i])

    state_position[:] = positions
    state_velocity[:] = velocities
    state_angular_velocity[:] = ang_vel
    state_half_extent[:] = extents
    state_mass[:] = masses
    state_inv_mass[:] = (1.0 / masses).astype(np.float32)
    state_orientation[:] = quats
    state_body_inertia_ref[:] = inertias
    state_body_inertia_ref_inv[:] = inertias_inv

    clear_applied_forces_cpu()
    upload_render_state()
    update_mesh_vertices()

    print("Randomized complex scene (CPU solver)")
    for i in range(N_BODIES):
        print(f"  body{i}: pos={positions[i]}, vel={velocities[i]}, mass={masses[i]:.3f}")


camera_pos = np.array([4.2, 2.7, 4.2], dtype=np.float32)
camera_target = np.array([0.0, 1.1, 0.0], dtype=np.float32)


def apply_mouse_drag_force(dx: float, dy: float, selected_body: int):
    ext = state_half_extent[selected_body]
    local_anchor = np.array([0.65 * ext[0], 0.15 * ext[1], 0.45 * ext[2]], dtype=np.float32)

    forward = safe_normalize(camera_target - camera_pos)
    right = safe_normalize(np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32)))
    up = safe_normalize(np.cross(right, forward))

    force = DRAG_FORCE_SCALE * (dx * right + dy * up)
    set_applied_force_cpu(selected_body, force, local_anchor)


def update_camera_from_keyboard(window):
    global camera_pos, camera_target

    forward = safe_normalize(camera_target - camera_pos)
    right = safe_normalize(np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32)))
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    move = np.zeros(3, dtype=np.float32)
    if window.is_pressed("w"):
        move += forward
    if window.is_pressed("s"):
        move -= forward
    if window.is_pressed("a"):
        move -= right
    if window.is_pressed("d"):
        move += right
    if window.is_pressed("q"):
        move += world_up
    if window.is_pressed("e"):
        move -= world_up

    move_norm = float(np.linalg.norm(move))
    if move_norm > 1e-6:
        move = move / move_norm * (CAMERA_MOVE_SPEED * FRAME_DT)
        camera_pos = camera_pos + move
        camera_target = camera_target + move


def rotate_camera_from_mouse(dx: float, dy: float):
    global camera_pos, camera_target

    offset = camera_pos - camera_target
    radius = float(np.linalg.norm(offset))
    if radius < 1e-6:
        radius = 1.0
        offset = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    azimuth = np.arctan2(offset[2], offset[0])
    horizontal = np.sqrt(offset[0] * offset[0] + offset[2] * offset[2])
    elevation = np.arctan2(offset[1], horizontal)

    azimuth -= dx * CAMERA_ROTATE_SPEED
    elevation -= dy * CAMERA_ROTATE_SPEED
    elevation = np.clip(elevation, -0.48 * np.pi, 0.48 * np.pi)

    camera_pos = camera_target + radius * np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.sin(elevation),
            np.cos(elevation) * np.sin(azimuth),
        ],
        dtype=np.float32,
    )


def print_controls():
    print("=" * 72)
    print("Complex rigid body scene (CPU solver + Taichi render)")
    print("LMB drag : apply force + torque to the selected body")
    print("TAB      : switch selected body")
    print("RMB drag : rotate camera around current target")
    print("WASDQE   : move camera")
    print("R        : randomize the scene")
    print("SPACE    : pause / resume")
    print("ESC      : quit")
    print("=" * 72)


def main():
    build_mesh_indices()
    randomize_complex_scene()
    print_controls()

    window = ti.ui.Window("Rigid Body Lab1 - Complex Scene (CPU Solver)", res=(1280, 900), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    paused = False
    dragging = False
    rotating = False
    selected_body = 0
    last_mouse_x = 0.0
    last_mouse_y = 0.0

    body_colors = [
        (0.86, 0.35, 0.30),
        (0.28, 0.60, 0.86),
        (0.30, 0.76, 0.42),
        (0.88, 0.66, 0.28),
        (0.72, 0.44, 0.82),
        (0.30, 0.78, 0.78),
    ]
    selected_body_color = (1.0, 0.93, 0.36)

    while window.running:
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.ESCAPE:
                window.running = False
            elif event.key == "r":
                randomize_complex_scene()
            elif event.key == ti.ui.SPACE:
                paused = not paused
            elif event.key == ti.ui.TAB:
                selected_body = (selected_body + 1) % N_BODIES
                print(f"Selected body: {selected_body}")

        update_camera_from_keyboard(window)

        if window.is_pressed(ti.ui.RMB):
            mouse_x, mouse_y = window.get_cursor_pos()
            if not rotating:
                rotating = True
                last_mouse_x = mouse_x
                last_mouse_y = mouse_y
            else:
                dx = mouse_x - last_mouse_x
                dy = mouse_y - last_mouse_y
                rotate_camera_from_mouse(dx, dy)
                last_mouse_x = mouse_x
                last_mouse_y = mouse_y
        else:
            rotating = False

        clear_applied_forces_cpu()
        if window.is_pressed(ti.ui.LMB):
            mouse_x, mouse_y = window.get_cursor_pos()
            if not dragging:
                dragging = True
                last_mouse_x = mouse_x
                last_mouse_y = mouse_y
            else:
                dx = mouse_x - last_mouse_x
                dy = mouse_y - last_mouse_y
                if abs(dx) + abs(dy) > 0.0:
                    apply_mouse_drag_force(dx, dy, selected_body)
                last_mouse_x = mouse_x
                last_mouse_y = mouse_y
        else:
            dragging = False

        if not paused:
            step_simulation()

        upload_render_state()
        update_mesh_vertices()
        update_selected_visuals(selected_body)

        camera.position(*camera_pos)
        camera.lookat(*camera_target)
        camera.up(0.0, 1.0, 0.0)

        scene.set_camera(camera)
        scene.ambient_light((0.66, 0.66, 0.66))
        scene.point_light((4.5, 5.5, 4.5), (1.25, 1.25, 1.25))

        canvas.set_background_color((0.82, 0.84, 0.88))
        scene.mesh(floor_vertices_ti, floor_indices_ti, color=(0.73, 0.73, 0.75))
        scene.lines(arena_line_verts, color=(0.38, 0.38, 0.38), width=1.2)
        scene.lines(axis_lines, color=(0.46, 0.46, 0.46), width=1.2)

        for body_id in range(N_BODIES):
            fill_color = selected_body_color if body_id == selected_body else body_colors[body_id % len(body_colors)]
            scene.mesh(
                mesh_vertices,
                mesh_indices,
                color=fill_color,
                index_offset=body_id * 36,
                index_count=36,
            )
            wire_color = (0.98, 0.82, 0.20) if body_id == selected_body else (0.0, 0.0, 0.0)
            scene.mesh(
                mesh_vertices,
                mesh_indices,
                color=wire_color,
                index_offset=body_id * 36,
                index_count=36,
                show_wireframe=True,
            )

        scene.particles(selected_com_vis, radius=0.035, color=(0.98, 0.82, 0.18))
        if dragging:
            scene.particles(force_anchor_vis, radius=0.028, color=(1.0, 0.84, 0.20))
            scene.lines(force_line, color=(1.0, 0.84, 0.20), width=4.0)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
