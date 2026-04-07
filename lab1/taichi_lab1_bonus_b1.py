import argparse
import copy
import numpy as np
import taichi as ti


ti.init(arch=ti.gpu, default_fp=ti.f32)


N_BODIES = 5
FRAME_DT = 1.0 / 60.0
EPSILON = 1e-6
LINEAR_DAMPING = 0.00
ANGULAR_DAMPING = 0.00
POSITION_CORRECTION_PERCENT = 0.72
POSITION_CORRECTION_SLOP = 0.0008
CAMERA_MOVE_SPEED = 2.6
CAMERA_ROTATE_SPEED = 2.2
AUTO_LOG_FRAME = 40
SIMULATION_FRAMES = AUTO_LOG_FRAME

BOX_HALF_EXTENT = np.array([0.22, 0.18, 0.18], dtype=np.float32)
BOX_MASS = np.float32(1.0)
CHAIN_GAP = np.float32(0.002)
LAUNCH_GAP = np.float32(0.46)
IMPACT_SPEED = np.float32(1.25)


EXPERIMENTS = [
    {
        "id": "1",
        "name": "baseline_serial",
        "description": "1 substep, 1 solver iter, forward pair sweep",
        "substeps": 1,
        "solver_iters": 1,
        "pair_order": "forward",
        "restitution": 1.0,
        "friction": 0.0,
    },
    {
        "id": "2",
        "name": "more_iters",
        "description": "1 substep, 8 solver iters, forward pair sweep",
        "substeps": 1,
        "solver_iters": 8,
        "pair_order": "forward",
        "restitution": 1.0,
        "friction": 0.0,
    },
    {
        "id": "3",
        "name": "more_substeps_iters",
        "description": "4 substeps, 8 solver iters, forward pair sweep",
        "substeps": 4,
        "solver_iters": 8,
        "pair_order": "forward",
        "restitution": 1.0,
        "friction": 0.0,
    },
    {
        "id": "4",
        "name": "alternating_sweeps",
        "description": "4 substeps, 8 solver iters, alternating pair sweep",
        "substeps": 4,
        "solver_iters": 8,
        "pair_order": "alternating",
        "restitution": 1.0,
        "friction": 0.0,
    },
]
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
        [-1.0, 0.0, 0.0],
        [4.5, 0.0, 0.0],
        [0.0, -0.65, 0.0],
        [0.0, 0.65, 0.0],
        [0.0, 0.0, -0.65],
        [0.0, 0.0, 0.65],
    ],
    dtype=np.float32,
)

TRACK_LINES = np.array(
    [
        [-1.0, -0.23, -0.22],
        [4.5, -0.23, -0.22],
        [-1.0, -0.23, 0.22],
        [4.5, -0.23, 0.22],
    ],
    dtype=np.float32,
)


cube_local_verts = ti.Vector.field(3, dtype=ti.f32, shape=8)
mesh_vertices = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES * 8)
mesh_indices = ti.field(dtype=ti.i32, shape=N_BODIES * 36)
axis_lines = ti.Vector.field(3, dtype=ti.f32, shape=6)
track_lines = ti.Vector.field(3, dtype=ti.f32, shape=4)
center_points = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)

position = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
orientation = ti.Vector.field(4, dtype=ti.f32, shape=N_BODIES)
half_extent = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)

cube_local_verts.from_numpy(CUBE_LOCAL_VERTICES)
axis_lines.from_numpy(WORLD_AXIS_LINES)
track_lines.from_numpy(TRACK_LINES)


state_position = np.zeros((N_BODIES, 3), dtype=np.float32)
state_velocity = np.zeros((N_BODIES, 3), dtype=np.float32)
state_angular_velocity = np.zeros((N_BODIES, 3), dtype=np.float32)
state_half_extent = np.tile(BOX_HALF_EXTENT[None, :], (N_BODIES, 1)).astype(np.float32)
state_mass = np.full(N_BODIES, BOX_MASS, dtype=np.float32)
state_inv_mass = np.full(N_BODIES, 1.0 / BOX_MASS, dtype=np.float32)
state_orientation = np.zeros((N_BODIES, 4), dtype=np.float32)
state_orientation[:, 0] = 1.0
state_body_inertia_ref = np.zeros((N_BODIES, 3, 3), dtype=np.float32)
state_body_inertia_ref_inv = np.zeros((N_BODIES, 3, 3), dtype=np.float32)


def make_empty_metrics():
    return {
        "frame_count": 0,
        "sample_count": 0,
        "time_elapsed": 0.0,
        "initial_ke": 0.0,
        "peak_body_speed": np.zeros(N_BODIES, dtype=np.float32),
        "peak_body_sample": np.full(N_BODIES, -1, dtype=np.int32),
        "energy_history": [],
        "collision_count": 0,
        "auto_log_done": False,
    }


current_experiment_index = 0
current_experiment = copy.deepcopy(EXPERIMENTS[current_experiment_index])
metrics = make_empty_metrics()


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
        center_points[i] = pos
        for k in range(8):
            lv = cube_local_verts[k]
            local = ti.Vector([lv[0] * ext[0], lv[1] * ext[1], lv[2] * ext[2]])
            mesh_vertices[i * 8 + k] = rot @ local + pos


def build_mesh_indices():
    indices = np.zeros(N_BODIES * 36, dtype=np.int32)
    for body_id in range(N_BODIES):
        indices[body_id * 36 : (body_id + 1) * 36] = CUBE_INDICES + body_id * 8
    mesh_indices.from_numpy(indices)


def upload_render_state():
    position.from_numpy(state_position)
    orientation.from_numpy(state_orientation)
    half_extent.from_numpy(state_half_extent)


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


def obb_support_radius(rot: np.ndarray, ext: np.ndarray, axis: np.ndarray):
    return float(
        abs(np.dot(axis, rot[:, 0])) * ext[0]
        + abs(np.dot(axis, rot[:, 1])) * ext[1]
        + abs(np.dot(axis, rot[:, 2])) * ext[2]
    )


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

    # For the Newton's-cradle setup we want a contact point on the line of centers
    # rather than an arbitrary face corner. This avoids injecting artificial torque
    # when two aligned boxes collide head-on.
    radius_a = obb_support_radius(rot_a, state_half_extent[i], normal)
    radius_b = obb_support_radius(rot_b, state_half_extent[j], normal)
    face_center_a = center_a + radius_a * normal
    face_center_b = center_b - radius_b * normal
    contact = 0.5 * (face_center_a + face_center_b)

    return True, normal, float(min_overlap), contact.astype(np.float32)


def impulse_denominator(inv_mass_value: float, inv_inertia: np.ndarray, r: np.ndarray, direction: np.ndarray):
    angular_term = np.cross(inv_inertia @ np.cross(r, direction), r)
    return inv_mass_value + np.dot(angular_term, direction)


def integrate_free_motion_cpu(sub_dt: float):
    linear_decay = np.float32(np.exp(-LINEAR_DAMPING * sub_dt))
    angular_decay = np.float32(np.exp(-ANGULAR_DAMPING * sub_dt))

    for body_id in range(N_BODIES):
        state_position[body_id] += state_velocity[body_id] * sub_dt

        q = quat_normalize_np(state_orientation[body_id])
        rot = quat_to_matrix_np(q)
        omega = state_angular_velocity[body_id]
        inertia_world = rot @ state_body_inertia_ref[body_id] @ rot.T
        inertia_world_inv = rot @ state_body_inertia_ref_inv[body_id] @ rot.T

        angular_acc = inertia_world_inv @ (-np.cross(omega, inertia_world @ omega))
        omega = omega + angular_acc * sub_dt
        omega = omega * angular_decay
        state_angular_velocity[body_id] = omega.astype(np.float32)

        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=np.float32)
        q = q + 0.5 * sub_dt * quat_mul_np(omega_quat, q)
        state_orientation[body_id] = quat_normalize_np(q)
        state_velocity[body_id] *= linear_decay


def resolve_collision_impulse(i: int, j: int, normal: np.ndarray, penetration: float, contact: np.ndarray):
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
        return False

    r_a = contact - pos_a
    r_b = contact - pos_b
    v_contact_a = vel_a + np.cross(omega_a, r_a)
    v_contact_b = vel_b + np.cross(omega_b, r_b)
    relative_velocity = v_contact_b - v_contact_a

    vel_along_normal = float(np.dot(relative_velocity, normal))
    if vel_along_normal > 0.0:
        return False

    denom_n = (
        impulse_denominator(inv_mass_a, inv_inertia_a, r_a, normal)
        + impulse_denominator(inv_mass_b, inv_inertia_b, r_b, normal)
    )
    if denom_n < EPSILON:
        return False

    restitution = current_experiment["restitution"]
    friction = current_experiment["friction"]

    j_n = -(1.0 + restitution) * vel_along_normal / denom_n
    impulse = j_n * normal

    tangent = relative_velocity - vel_along_normal * normal
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm > EPSILON and friction > 0.0:
        tangent = tangent / tangent_norm
        denom_t = (
            impulse_denominator(inv_mass_a, inv_inertia_a, r_a, tangent)
            + impulse_denominator(inv_mass_b, inv_inertia_b, r_b, tangent)
        )
        if denom_t > EPSILON:
            j_t = -np.dot(relative_velocity, tangent) / denom_t
            friction_limit = friction * j_n
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

    return True


def compute_total_kinetic_energy():
    total = 0.0
    for body_id in range(N_BODIES):
        total += 0.5 * float(state_mass[body_id]) * float(np.dot(state_velocity[body_id], state_velocity[body_id]))
        rot = get_rotation_matrix(body_id)
        inertia_world = rot @ state_body_inertia_ref[body_id] @ rot.T
        omega = state_angular_velocity[body_id]
        total += 0.5 * float(omega @ (inertia_world @ omega))
    return total


def get_pair_sequence(sweep_index: int):
    pair_data = []
    for i in range(N_BODIES):
        for j in range(i + 1, N_BODIES):
            midpoint_x = 0.5 * (state_position[i, 0] + state_position[j, 0])
            pair_data.append((midpoint_x, i, j))

    pair_data.sort(key=lambda item: item[0])
    if current_experiment["pair_order"] == "alternating" and sweep_index % 2 == 1:
        pair_data.reverse()
    return [(i, j) for _, i, j in pair_data]


def record_peak_metrics():
    metrics["sample_count"] += 1
    speed_x = np.abs(state_velocity[:, 0])
    for body_id in range(N_BODIES):
        if speed_x[body_id] > metrics["peak_body_speed"][body_id]:
            metrics["peak_body_speed"][body_id] = speed_x[body_id]
            metrics["peak_body_sample"][body_id] = metrics["sample_count"]


def record_frame_metrics():
    metrics["frame_count"] += 1
    metrics["time_elapsed"] += FRAME_DT
    metrics["energy_history"].append(compute_total_kinetic_energy())


def step_simulation_frame():
    substeps = current_experiment["substeps"]
    solver_iters = current_experiment["solver_iters"]
    sub_dt = FRAME_DT / substeps

    for _ in range(substeps):
        integrate_free_motion_cpu(sub_dt)

        for solver_iter in range(solver_iters):
            for i, j in get_pair_sequence(solver_iter):
                collided, normal, penetration, contact = collision_manifold(i, j)
                if collided and resolve_collision_impulse(i, j, normal, penetration, contact):
                    metrics["collision_count"] += 1

        record_peak_metrics()

    record_frame_metrics()


def print_experiment_header(exp: dict):
    print("=" * 72)
    print(f"[B1] Experiment {exp['id']}: {exp['name']}")
    print(exp["description"])
    print(
        "params:",
        f"substeps={exp['substeps']}, solver_iters={exp['solver_iters']},",
        f"pair_order={exp['pair_order']}, restitution={exp['restitution']:.2f},",
        f"friction={exp['friction']:.2f}",
    )
    print("=" * 72)


def initialize_chain(experiment_index: int, verbose: bool = True):
    global current_experiment_index, current_experiment, metrics

    current_experiment_index = experiment_index
    current_experiment = copy.deepcopy(EXPERIMENTS[experiment_index])
    metrics = make_empty_metrics()

    inertia, inertia_inv = box_inertia_tensor(float(BOX_MASS), BOX_HALF_EXTENT)
    for body_id in range(N_BODIES):
        state_body_inertia_ref[body_id] = inertia
        state_body_inertia_ref_inv[body_id] = inertia_inv

    state_half_extent[:] = BOX_HALF_EXTENT
    state_mass[:] = BOX_MASS
    state_inv_mass[:] = 1.0 / BOX_MASS
    state_velocity.fill(0.0)
    state_angular_velocity.fill(0.0)
    state_orientation.fill(0.0)
    state_orientation[:, 0] = 1.0

    for body_id in range(N_BODIES):
        center_x = body_id * (2.0 * BOX_HALF_EXTENT[0] + CHAIN_GAP)
        state_position[body_id] = np.array([center_x, 0.0, 0.0], dtype=np.float32)

    state_position[0, 0] -= LAUNCH_GAP
    state_velocity[0] = np.array([IMPACT_SPEED, 0.0, 0.0], dtype=np.float32)

    metrics["initial_ke"] = compute_total_kinetic_energy()
    metrics["energy_history"].append(metrics["initial_ke"])

    upload_render_state()
    update_mesh_vertices()

    if verbose:
        print_experiment_header(current_experiment)


def summarize_experiment(frames_run: int):
    initial_ke = metrics["initial_ke"]
    final_ke = compute_total_kinetic_energy()
    peak_speed = metrics["peak_body_speed"]
    peak_sample = metrics["peak_body_sample"]
    sample_dt = FRAME_DT / current_experiment["substeps"]
    peak_time = np.full(N_BODIES, -1.0, dtype=np.float32)
    valid_samples = peak_sample >= 0
    peak_time[valid_samples] = peak_sample[valid_samples] * sample_dt

    transfer_order = [idx for idx in np.argsort(peak_sample) if peak_sample[idx] >= 0]
    order_str = " -> ".join(str(idx) for idx in transfer_order) if transfer_order else "none"

    print(f"Summary for {current_experiment['name']}")
    print(f"  frames_run       = {frames_run}")
    print(f"  substep_samples  = {metrics['sample_count']}")
    print(f"  collisions       = {metrics['collision_count']}")
    print(f"  initial_ke       = {initial_ke:.6f}")
    print(f"  final_ke         = {final_ke:.6f}")
    print(f"  final_ke_ratio   = {final_ke / max(initial_ke, EPSILON):.6f}")
    print(f"  transfer_order   = {order_str}")
    print(f"  last_body_peak_v = {peak_speed[-1]:.6f}")
    print(f"  body_peak_vx     = {np.array2string(peak_speed, precision=4)}")
    print(f"  body_peak_sample = {peak_sample}")
    print(f"  body_peak_time   = {np.array2string(peak_time, precision=4)}")


def maybe_auto_log():
    if metrics["auto_log_done"]:
        return
    if metrics["frame_count"] < AUTO_LOG_FRAME:
        return

    print(f"[auto-log] summary at frame {AUTO_LOG_FRAME}")
    summarize_experiment(AUTO_LOG_FRAME)
    metrics["auto_log_done"] = True


def run_batch(frames: int):
    print("# B1 batch comparison")
    for experiment_index, experiment in enumerate(EXPERIMENTS):
        initialize_chain(experiment_index, verbose=True)
        for _ in range(frames):
            step_simulation_frame()
            maybe_auto_log()
        if not metrics["auto_log_done"]:
            summarize_experiment(metrics["frame_count"])
        print()


camera_pos = np.array([1.5, 1.35, 3.2], dtype=np.float32)
camera_target = np.array([1.5, 0.0, 0.0], dtype=np.float32)


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
    print("B1 Newton's cradle exploration (CPU solver + Taichi render)")
    print("TAB      : cycle experiment preset and reset")
    print("R        : reset current experiment")
    print("P        : print current summary")
    print(f"auto-log : summarize once at frame {AUTO_LOG_FRAME}")
    print("SPACE    : pause / resume")
    print("WASDQE   : move camera")
    print("RMB drag : orbit camera")
    print("ESC      : quit")
    print("=" * 72)


def draw_status_panel(gui, paused: bool):
    gui.begin("Status", 0.02, 0.02, 0.26, 0.12)
    gui.text(f"Mode: {current_experiment['name']}")
    gui.text(f"Preset: {current_experiment['description']}")
    gui.text(f"State: {'paused' if paused else 'running'}")
    gui.end()


def run_interactive():
    build_mesh_indices()
    initialize_chain(0, verbose=True)
    print_controls()

    window = ti.ui.Window("Lab1 B1 - Newton's Cradle (CPU Solver)", res=(1280, 900), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    gui = window.get_gui()

    paused = False
    rotating = False
    last_mouse_x = 0.0
    last_mouse_y = 0.0

    body_colors = [
        (0.90, 0.40, 0.32),
        (0.97, 0.67, 0.25),
        (0.30, 0.75, 0.47),
        (0.26, 0.58, 0.88),
        (0.70, 0.44, 0.82),
    ]

    while window.running:
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.ESCAPE:
                window.running = False
            elif event.key == ti.ui.SPACE:
                paused = not paused
            elif event.key == ti.ui.TAB:
                next_experiment_index = (current_experiment_index + 1) % len(EXPERIMENTS)
                initialize_chain(next_experiment_index, verbose=True)
            elif event.key == "r":
                initialize_chain(current_experiment_index, verbose=True)
            elif event.key == "p":
                summarize_experiment(metrics["frame_count"])

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
            step_simulation_frame()
            maybe_auto_log()

        upload_render_state()
        update_mesh_vertices()

        camera.position(*camera_pos)
        camera.lookat(*camera_target)
        camera.up(0.0, 1.0, 0.0)

        scene.set_camera(camera)
        scene.ambient_light((0.66, 0.66, 0.66))
        scene.point_light((3.0, 4.2, 4.8), (1.15, 1.15, 1.15))

        canvas.set_background_color((0.82, 0.84, 0.88))
        scene.lines(axis_lines, color=(0.42, 0.42, 0.42), width=1.2)
        scene.lines(track_lines, color=(0.28, 0.28, 0.30), width=2.0)

        for body_id in range(N_BODIES):
            scene.mesh(
                mesh_vertices,
                mesh_indices,
                color=body_colors[body_id % len(body_colors)],
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

        scene.particles(center_points, radius=0.026, color=(0.98, 0.92, 0.20))
        canvas.scene(scene)
        draw_status_panel(gui, paused)
        window.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Lab1 Bonus B1 Newton's cradle experiment")
    parser.add_argument("--batch", action="store_true", help="run all presets without opening a window")
    parser.add_argument("--frames", type=int, default=SIMULATION_FRAMES, help="number of frames for batch mode")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.batch:
        build_mesh_indices()
        run_batch(args.frames)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
