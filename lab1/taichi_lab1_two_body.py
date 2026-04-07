"""
Taichi Lab 1 - two rigid bodies with impulse-based collision response.

Features:
- two boxes with random initial state that is guaranteed to collide
- semi-implicit rigid body integration with quaternion orientation
- SAT-based OBB collision manifold
- impulse-based collision response at a single contact point
- WASDQE camera translation
- press R to randomize a new collision scene
- press SPACE to pause / resume
"""

import numpy as np
import taichi as ti


ti.init(arch=ti.gpu, default_fp=ti.f32)


N_BODIES = 2
FRAME_DT = 1.0 / 60.0
SUBSTEPS = 4
DT = FRAME_DT / SUBSTEPS
GRAVITY = ti.Vector([0.0, 0.0, 0.0])
LINEAR_DAMPING = 0.02
ANGULAR_DAMPING = 0.03
RESTITUTION = 0.65
FRICTION = 0.18
EPSILON = 1e-6
POSITION_CORRECTION_PERCENT = 0.75
POSITION_CORRECTION_SLOP = 0.003
CAMERA_MOVE_SPEED = 2.6
CAMERA_ROTATE_SPEED = 2.4


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
        [1.3, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.3, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.3],
    ],
    dtype=np.float32,
)


cube_local_verts = ti.Vector.field(3, dtype=ti.f32, shape=8)
cube_indices_ti = ti.field(dtype=ti.i32, shape=36)
mesh_vertices = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES * 8)
mesh_indices = ti.field(dtype=ti.i32, shape=N_BODIES * 36)
axis_lines = ti.Vector.field(3, dtype=ti.f32, shape=6)

position = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
velocity = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
angular_velocity = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
half_extent = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
mass = ti.field(dtype=ti.f32, shape=N_BODIES)
inv_mass = ti.field(dtype=ti.f32, shape=N_BODIES)
orientation = ti.Vector.field(4, dtype=ti.f32, shape=N_BODIES)
body_inertia_ref = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_BODIES)
body_inertia_ref_inv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_BODIES)

cube_local_verts.from_numpy(CUBE_LOCAL_VERTICES)
cube_indices_ti.from_numpy(CUBE_INDICES)
axis_lines.from_numpy(WORLD_AXIS_LINES)


@ti.func
def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return ti.Vector(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


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
def integrate_free_motion():
    for i in range(N_BODIES):
        velocity[i] += GRAVITY * DT
        velocity[i] *= ti.exp(-LINEAR_DAMPING * DT)
        position[i] += velocity[i] * DT

        q = quat_normalize(orientation[i])
        R = quat_to_matrix(q)
        omega = angular_velocity[i]
        I_world = R @ body_inertia_ref[i] @ R.transpose()
        I_world_inv = R @ body_inertia_ref_inv[i] @ R.transpose()

        angular_acc = I_world_inv @ (-omega.cross(I_world @ omega))
        omega += angular_acc * DT
        omega *= ti.exp(-ANGULAR_DAMPING * DT)
        angular_velocity[i] = omega

        omega_quat = ti.Vector([0.0, omega[0], omega[1], omega[2]])
        q = q + 0.5 * DT * quat_mul(omega_quat, q)
        orientation[i] = quat_normalize(q)


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
        for t in range(12):
            for v in range(3):
                mesh_indices[i * 36 + t * 3 + v] = i * 8 + cube_indices_ti[t * 3 + v]


def safe_normalize(v: np.ndarray):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        fallback = np.zeros_like(v, dtype=np.float32)
        fallback[0] = 1.0
        return fallback
    return (v / norm).astype(np.float32)


def axis_angle_to_quat(axis: np.ndarray, angle: float):
    axis = safe_normalize(axis)
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)


def quat_to_matrix_np(q: np.ndarray):
    q = safe_normalize(q)
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


def obb_support_radius(rotation_matrix: np.ndarray, ext: np.ndarray, axis: np.ndarray):
    return float(
        abs(np.dot(axis, rotation_matrix[:, 0])) * ext[0]
        + abs(np.dot(axis, rotation_matrix[:, 1])) * ext[1]
        + abs(np.dot(axis, rotation_matrix[:, 2])) * ext[2]
    )


def get_rotation_matrix(i: int):
    return quat_to_matrix_np(orientation[i].to_numpy())


def get_box_vertices(i: int):
    pos = position[i].to_numpy()
    rot = get_rotation_matrix(i)
    ext = half_extent[i].to_numpy()
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

    center_a = position[i].to_numpy()
    center_b = position[j].to_numpy()
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


def resolve_collision_impulse(i: int, j: int, normal: np.ndarray, penetration: float, contact: np.ndarray):
    pos_a = position[i].to_numpy()
    pos_b = position[j].to_numpy()
    vel_a = velocity[i].to_numpy()
    vel_b = velocity[j].to_numpy()
    omega_a = angular_velocity[i].to_numpy()
    omega_b = angular_velocity[j].to_numpy()

    rot_a = get_rotation_matrix(i)
    rot_b = get_rotation_matrix(j)
    inv_inertia_a = rot_a @ body_inertia_ref_inv[i].to_numpy() @ rot_a.T
    inv_inertia_b = rot_b @ body_inertia_ref_inv[j].to_numpy() @ rot_b.T

    inv_mass_a = float(inv_mass[i])
    inv_mass_b = float(inv_mass[j])
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

    j_n = -(1.0 + RESTITUTION) * vel_along_normal / denom_n
    impulse = j_n * normal

    tangent = relative_velocity - vel_along_normal * normal
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm > EPSILON:
        tangent = tangent / tangent_norm
        denom_t = (
            impulse_denominator(inv_mass_a, inv_inertia_a, r_a, tangent)
            + impulse_denominator(inv_mass_b, inv_inertia_b, r_b, tangent)
        )
        if denom_t > EPSILON:
            j_t = -np.dot(relative_velocity, tangent) / denom_t
            friction_limit = FRICTION * j_n
            j_t = np.clip(j_t, -friction_limit, friction_limit)
            impulse = impulse + j_t * tangent

    vel_a = vel_a - inv_mass_a * impulse
    vel_b = vel_b + inv_mass_b * impulse
    omega_a = omega_a - inv_inertia_a @ np.cross(r_a, impulse)
    omega_b = omega_b + inv_inertia_b @ np.cross(r_b, impulse)

    velocity[i] = vel_a.astype(np.float32)
    velocity[j] = vel_b.astype(np.float32)
    angular_velocity[i] = omega_a.astype(np.float32)
    angular_velocity[j] = omega_b.astype(np.float32)

    correction_mag = max(penetration - POSITION_CORRECTION_SLOP, 0.0)
    if correction_mag > 0.0:
        correction_mag *= POSITION_CORRECTION_PERCENT / (inv_mass_a + inv_mass_b)
        correction = correction_mag * normal
        position[i] = (pos_a - inv_mass_a * correction).astype(np.float32)
        position[j] = (pos_b + inv_mass_b * correction).astype(np.float32)


rng = np.random.default_rng()


def random_quaternion():
    axis = safe_normalize(rng.normal(size=3).astype(np.float32))
    angle = float(rng.uniform(0.0, np.pi))
    return axis_angle_to_quat(axis, angle)


def randomize_collision_pair():
    extents = rng.uniform(0.24, 0.42, size=(N_BODIES, 3)).astype(np.float32)
    masses = rng.uniform(0.9, 1.6, size=N_BODIES).astype(np.float32)
    quats = np.stack([random_quaternion(), random_quaternion()], axis=0).astype(np.float32)
    ang_vel = rng.uniform(-0.6, 0.6, size=(N_BODIES, 3)).astype(np.float32)

    collision_axis = safe_normalize(rng.normal(size=3).astype(np.float32))
    tangent_seed = rng.normal(size=3).astype(np.float32)
    tangent_u = tangent_seed - np.dot(tangent_seed, collision_axis) * collision_axis
    tangent_u = safe_normalize(tangent_u)
    tangent_v = safe_normalize(np.cross(collision_axis, tangent_u))
    shared_offset = rng.uniform(-0.15, 0.15) * tangent_u + rng.uniform(-0.15, 0.15) * tangent_v

    rot0 = quat_to_matrix_np(quats[0])
    rot1 = quat_to_matrix_np(quats[1])
    radius0 = obb_support_radius(rot0, extents[0], collision_axis)
    radius1 = obb_support_radius(rot1, extents[1], collision_axis)
    initial_gap = rng.uniform(0.45, 0.85)
    center_distance = radius0 + radius1 + initial_gap

    positions = np.zeros((N_BODIES, 3), dtype=np.float32)
    positions[0] = shared_offset - 0.5 * center_distance * collision_axis
    positions[1] = shared_offset + 0.5 * center_distance * collision_axis

    relative_speed = rng.uniform(0.85, 1.35)
    common_velocity = rng.uniform(-0.10, 0.10, size=3).astype(np.float32)
    velocities = np.zeros((N_BODIES, 3), dtype=np.float32)
    velocities[0] = common_velocity + 0.5 * relative_speed * collision_axis
    velocities[1] = common_velocity - 0.5 * relative_speed * collision_axis

    inertias = np.zeros((N_BODIES, 3, 3), dtype=np.float32)
    inertias_inv = np.zeros((N_BODIES, 3, 3), dtype=np.float32)
    for idx in range(N_BODIES):
        inertias[idx], inertias_inv[idx] = box_inertia_tensor(float(masses[idx]), extents[idx])

    position.from_numpy(positions)
    velocity.from_numpy(velocities)
    angular_velocity.from_numpy(ang_vel)
    half_extent.from_numpy(extents)
    mass.from_numpy(masses)
    inv_mass.from_numpy((1.0 / masses).astype(np.float32))
    orientation.from_numpy(quats)
    body_inertia_ref.from_numpy(inertias)
    body_inertia_ref_inv.from_numpy(inertias_inv)

    update_mesh_vertices()

    print("Randomized two-body collision scene")
    print(f"  collision_axis = {collision_axis}")
    print(f"  body0 position = {positions[0]}, velocity = {velocities[0]}")
    print(f"  body1 position = {positions[1]}, velocity = {velocities[1]}")


camera_pos = np.array([3.2, 2.1, 3.2], dtype=np.float32)
camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)


def update_camera_from_keyboard(window):
    global camera_pos, camera_target

    forward = camera_target - camera_pos
    forward = safe_normalize(forward)
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

    move_norm = np.linalg.norm(move)
    if move_norm > 1e-6:
        move = move / move_norm * (CAMERA_MOVE_SPEED * FRAME_DT)
        camera_pos = camera_pos + move
        camera_target = camera_target + move


def rotate_camera_from_mouse(dx: float, dy: float):
    global camera_pos, camera_target

    offset = camera_pos - camera_target
    radius = np.linalg.norm(offset)
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
    print("=" * 56)
    print("Two rigid bodies - impulse-based collision demo")
    print("WASDQE   : move camera (forward/back/left/right/up/down)")
    print("RMB drag : rotate camera around current target")
    print("R        : randomize a new guaranteed-collision scene")
    print("SPACE    : pause / resume")
    print("ESC      : quit")
    print("=" * 56)


def main():
    randomize_collision_pair()
    print_controls()

    window = ti.ui.Window("Rigid Body Lab1 - Two Bodies", res=(1280, 900), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    paused = False
    rotating = False
    last_mouse_x = 0.0
    last_mouse_y = 0.0

    colors = [(0.84, 0.34, 0.30), (0.26, 0.56, 0.86)]

    while window.running:
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.ESCAPE:
                window.running = False
            elif event.key == "r":
                randomize_collision_pair()
            elif event.key == ti.ui.SPACE:
                paused = not paused

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

        if not paused:
            for _ in range(SUBSTEPS):
                integrate_free_motion()
                ti.sync()

                collided, normal, penetration, contact = collision_manifold(0, 1)
                if collided:
                    resolve_collision_impulse(0, 1, normal, penetration, contact)

        update_mesh_vertices()

        camera.position(*camera_pos)
        camera.lookat(*camera_target)
        camera.up(0.0, 1.0, 0.0)

        scene.set_camera(camera)
        scene.ambient_light((0.66, 0.66, 0.66))
        scene.point_light((4.5, 5.0, 4.0), (1.2, 1.2, 1.2))

        canvas.set_background_color((0.82, 0.84, 0.88))
        scene.lines(axis_lines, color=(0.45, 0.45, 0.45), width=1.5)

        for body_id in range(N_BODIES):
            scene.mesh(
                mesh_vertices,
                mesh_indices,
                color=colors[body_id],
                index_offset=body_id * 36,
                index_count=36,
            )
            scene.mesh(
                mesh_vertices,
                mesh_indices,
                color=(0.0, 0.0, 0.0),
                index_offset=body_id * 36,
                index_count=36,
                show_wireframe=True,
            )

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
