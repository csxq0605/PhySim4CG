"""
Taichi Lab 3 bonus B1 - hyperelastic model comparison.

Three copies of the same tetrahedral soft body are simulated side by side:
- StVK
- Neo-Hookean
- Corotated

The same mouse handle displacement is applied to all three models so their
responses can be compared directly.
"""

import sys

import numpy as np
import taichi as ti


USE_CPU = "--cpu" in sys.argv

ti.init(arch=ti.cpu if USE_CPU else ti.gpu, default_fp=ti.f32)


MODEL_NAMES = ("StVK", "Neo-Hookean", "Corotated")
N_MODELS = 3
MODEL_OFFSETS = np.array(
    [
        [0.0, 0.0, -3.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0],
    ],
    dtype=np.float32,
)
MODEL_COLORS = np.array(
    [
        [0.88, 0.34, 0.26],
        [0.22, 0.55, 0.88],
        [0.25, 0.72, 0.42],
    ],
    dtype=np.float32,
)

GRID_X, GRID_Y, GRID_Z = 12, 3, 3
BODY_SIZE = np.array([8.0, 2.0, 2.0], dtype=np.float32)
BODY_ORIGIN = np.array([-4.0, 1.10, -1.0], dtype=np.float32)

YOUNG_MODULUS_DEFAULT = 20000.0
POISSON_RATIO = 0.2
DENSITY = 400.0
GRAVITY_DEFAULT = 0.05
DAMPING_DEFAULT = 2.2
FRAME_DT_DEFAULT = 1.0 / 60.0
SUBSTEPS_DEFAULT = 14

WINDOW_RES = (1280, 900)
FLOOR_Y = 0.0
MAX_SPEED = 24.0
DRAG_STIFFNESS_DEFAULT = 8500.0
DRAG_DAMPING_DEFAULT = 115.0
MOUSE_DRAG_WORLD_SCALE = 10.0
MAX_CONTROL_FORCE = 72000.0
CONTROL_HANDLE_RADIUS = 0.95
MAX_CONTROL_HANDLE_VERTICES = 32
CAMERA_ROTATE_SPEED = 2.4
CAMERA_FOV = 62.0
CAMERA_MIN_DISTANCE = 5.0
CAMERA_MAX_DISTANCE = 32.0
CAMERA_ZOOM_SPEED = 0.18


def lame_parameters(young: float, poisson: float):
    mu = young / (2.0 * (1.0 + poisson))
    la = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return float(mu), float(la)


def get_grid_vertex_id(i: int, j: int, k: int):
    return i * (GRID_Y + 1) * (GRID_Z + 1) + j * (GRID_Z + 1) + k


def create_base_mesh():
    vertices = []
    fixed = []
    dx = BODY_SIZE[0] / GRID_X
    dy = BODY_SIZE[1] / GRID_Y
    dz = BODY_SIZE[2] / GRID_Z

    for i in range(GRID_X + 1):
        for j in range(GRID_Y + 1):
            for k in range(GRID_Z + 1):
                vertices.append(BODY_ORIGIN + np.array([i * dx, j * dy, k * dz], dtype=np.float32))
                is_anchor = i == 0 and j in (GRID_Y // 2, GRID_Y // 2 + 1) and k in (GRID_Z // 2, GRID_Z // 2 + 1)
                fixed.append(1 if is_anchor else 0)

    vertices = np.array(vertices, dtype=np.float32)
    fixed = np.array(fixed, dtype=np.int32)

    tet_pattern = [
        (0, 4, 7, 6),
        (0, 3, 7, 6),
        (0, 4, 5, 6),
        (0, 1, 5, 6),
        (0, 3, 2, 6),
        (0, 1, 2, 6),
    ]
    tets = []
    for i in range(GRID_X):
        for j in range(GRID_Y):
            for k in range(GRID_Z):
                cell = [
                    get_grid_vertex_id(i, j, k),
                    get_grid_vertex_id(i + 1, j, k),
                    get_grid_vertex_id(i + 1, j + 1, k),
                    get_grid_vertex_id(i, j + 1, k),
                    get_grid_vertex_id(i, j, k + 1),
                    get_grid_vertex_id(i + 1, j, k + 1),
                    get_grid_vertex_id(i + 1, j + 1, k + 1),
                    get_grid_vertex_id(i, j + 1, k + 1),
                ]
                for local_tet in tet_pattern:
                    tet = [cell[idx] for idx in local_tet]
                    dm = np.column_stack(
                        (
                            vertices[tet[1]] - vertices[tet[0]],
                            vertices[tet[2]] - vertices[tet[0]],
                            vertices[tet[3]] - vertices[tet[0]],
                        )
                    )
                    if np.linalg.det(dm) < 0.0:
                        tet[1], tet[2] = tet[2], tet[1]
                    tets.append(tet)
    tets = np.array(tets, dtype=np.int32)

    dm_inv = np.zeros((len(tets), 3, 3), dtype=np.float32)
    tet_volume = np.zeros(len(tets), dtype=np.float32)
    vertex_mass = np.zeros(len(vertices), dtype=np.float32)
    for tet_id, tet in enumerate(tets):
        dm = np.column_stack(
            (
                vertices[tet[1]] - vertices[tet[0]],
                vertices[tet[2]] - vertices[tet[0]],
                vertices[tet[3]] - vertices[tet[0]],
            )
        )
        volume = abs(np.linalg.det(dm)) / 6.0
        dm_inv[tet_id] = np.linalg.inv(dm).astype(np.float32)
        tet_volume[tet_id] = np.float32(volume)
        for vertex_id in tet:
            vertex_mass[vertex_id] += DENSITY * volume / 4.0

    inv_mass = np.zeros_like(vertex_mass, dtype=np.float32)
    active = fixed == 0
    inv_mass[active] = 1.0 / np.maximum(vertex_mass[active], 1e-8)

    face_records = {}
    edge_set = set()
    tet_faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    tet_edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for tet in tets:
        for a, b, c in tet_faces:
            face = (int(tet[a]), int(tet[b]), int(tet[c]))
            key = tuple(sorted(face))
            if key not in face_records:
                face_records[key] = [1, face]
            else:
                face_records[key][0] += 1
        for a, b in tet_edges:
            edge_set.add(tuple(sorted((int(tet[a]), int(tet[b])))))

    surface_faces = []
    for count, face in face_records.values():
        if count == 1:
            surface_faces.extend(face)

    edge_indices = []
    for a, b in sorted(edge_set):
        edge_indices.extend([a, b])

    control_targets = [
        BODY_ORIGIN + np.array([BODY_SIZE[0], 0.5 * BODY_SIZE[1], 0.5 * BODY_SIZE[2]], dtype=np.float32),
        BODY_ORIGIN + np.array([BODY_SIZE[0], BODY_SIZE[1], 0.5 * BODY_SIZE[2]], dtype=np.float32),
        BODY_ORIGIN + np.array([BODY_SIZE[0], 0.5 * BODY_SIZE[1], BODY_SIZE[2]], dtype=np.float32),
        BODY_ORIGIN + np.array([0.72 * BODY_SIZE[0], BODY_SIZE[1], 0.5 * BODY_SIZE[2]], dtype=np.float32),
    ]
    control_vertices = []
    for target in control_targets:
        dist = np.linalg.norm(vertices - target[None, :], axis=1)
        control_vertices.append(int(np.argmin(dist)))

    fixed_vertices = np.nonzero(fixed != 0)[0].astype(np.int32)
    return (
        vertices,
        fixed,
        tets,
        dm_inv,
        tet_volume,
        vertex_mass.astype(np.float32),
        inv_mass.astype(np.float32),
        np.array(surface_faces, dtype=np.int32),
        np.array(edge_indices, dtype=np.int32),
        np.array(control_vertices, dtype=np.int32),
        fixed_vertices,
    )


def create_control_handles(vertices: np.ndarray, fixed: np.ndarray, control_vertices: np.ndarray):
    handle_indices = np.zeros((len(control_vertices), MAX_CONTROL_HANDLE_VERTICES), dtype=np.int32)
    handle_weights = np.zeros((len(control_vertices), MAX_CONTROL_HANDLE_VERTICES), dtype=np.float32)
    handle_offsets = np.zeros((len(control_vertices), MAX_CONTROL_HANDLE_VERTICES, 3), dtype=np.float32)

    for control_id, center_vertex in enumerate(control_vertices):
        center = vertices[center_vertex]
        dist = np.linalg.norm(vertices - center[None, :], axis=1)
        candidates = [
            int(i)
            for i in np.argsort(dist)
            if fixed[i] == 0 and dist[i] <= CONTROL_HANDLE_RADIUS
        ]
        if int(center_vertex) not in candidates:
            candidates.insert(0, int(center_vertex))
        candidates = candidates[:MAX_CONTROL_HANDLE_VERTICES]

        raw_weights = []
        for vertex_id in candidates:
            falloff = max(0.0, 1.0 - float(dist[vertex_id]) / CONTROL_HANDLE_RADIUS)
            raw_weights.append(max(0.05, falloff * falloff))
        weight_sum = max(float(np.sum(raw_weights)), 1e-8)

        for slot, vertex_id in enumerate(candidates):
            handle_indices[control_id, slot] = vertex_id
            handle_weights[control_id, slot] = np.float32(raw_weights[slot] / weight_sum)
            handle_offsets[control_id, slot] = vertices[vertex_id] - center
    return handle_indices, handle_weights, handle_offsets


(
    BASE_REST_POSITIONS,
    BASE_FIXED_FLAGS,
    TETS,
    DM_INV,
    TET_VOLUME,
    BASE_VERTEX_MASS,
    BASE_VERTEX_INV_MASS,
    BASE_SURFACE_FACE_INDICES,
    BASE_EDGE_INDICES,
    CONTROL_VERTICES,
    BASE_FIXED_VERTICES,
) = create_base_mesh()

(
    CONTROL_HANDLE_INDICES,
    CONTROL_HANDLE_WEIGHTS,
    CONTROL_HANDLE_OFFSETS,
) = create_control_handles(BASE_REST_POSITIONS, BASE_FIXED_FLAGS, CONTROL_VERTICES)

N_VERTS = BASE_REST_POSITIONS.shape[0]
N_TETS = TETS.shape[0]
N_FACE_INDICES = BASE_SURFACE_FACE_INDICES.shape[0]
N_EDGE_INDICES = BASE_EDGE_INDICES.shape[0]
N_CONTROL_VERTICES = CONTROL_VERTICES.shape[0]
N_FIXED_VERTICES = BASE_FIXED_VERTICES.shape[0]
TOTAL_VERTS = N_MODELS * N_VERTS
TOTAL_FACE_INDICES = N_MODELS * N_FACE_INDICES
TOTAL_EDGE_INDICES = N_MODELS * N_EDGE_INDICES
TOTAL_FIXED_VERTICES = N_MODELS * N_FIXED_VERTICES


def create_render_arrays():
    rest = np.zeros((TOTAL_VERTS, 3), dtype=np.float32)
    color = np.zeros((TOTAL_VERTS, 3), dtype=np.float32)
    fixed = np.zeros(TOTAL_VERTS, dtype=np.int32)
    mass = np.zeros(TOTAL_VERTS, dtype=np.float32)
    inv_mass = np.zeros(TOTAL_VERTS, dtype=np.float32)

    for model in range(N_MODELS):
        start = model * N_VERTS
        end = start + N_VERTS
        rest[start:end] = BASE_REST_POSITIONS + MODEL_OFFSETS[model]
        fixed[start:end] = BASE_FIXED_FLAGS
        mass[start:end] = BASE_VERTEX_MASS
        inv_mass[start:end] = BASE_VERTEX_INV_MASS
        t = (BASE_REST_POSITIONS[:, 0] - BASE_REST_POSITIONS[:, 0].min()) / BODY_SIZE[0]
        shade = (0.72 + 0.28 * t)[:, None]
        color[start:end] = MODEL_COLORS[model][None, :] * shade

    face_indices = []
    edge_indices = []
    fixed_positions = []
    for model in range(N_MODELS):
        offset = model * N_VERTS
        face_indices.extend((BASE_SURFACE_FACE_INDICES + offset).tolist())
        edge_indices.extend((BASE_EDGE_INDICES + offset).tolist())
        fixed_positions.extend(rest[offset + BASE_FIXED_VERTICES].tolist())

    floor_vertices = np.array(
        [
            [-4.8, FLOOR_Y, -5.2],
            [4.8, FLOOR_Y, -5.2],
            [4.8, FLOOR_Y, 5.2],
            [-4.8, FLOOR_Y, 5.2],
        ],
        dtype=np.float32,
    )
    floor_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    return (
        rest,
        color,
        fixed,
        mass,
        inv_mass,
        np.array(face_indices, dtype=np.int32),
        np.array(edge_indices, dtype=np.int32),
        np.array(fixed_positions, dtype=np.float32),
        floor_vertices,
        floor_indices,
    )


(
    REST_POSITIONS,
    VERTEX_COLORS,
    FIXED_FLAGS,
    VERTEX_MASS,
    VERTEX_INV_MASS,
    SURFACE_FACE_INDICES,
    EDGE_INDICES,
    FIXED_POINT_POSITIONS,
    FLOOR_VERTICES,
    FLOOR_INDICES,
) = create_render_arrays()


position = ti.Vector.field(3, dtype=ti.f32, shape=TOTAL_VERTS)
velocity = ti.Vector.field(3, dtype=ti.f32, shape=TOTAL_VERTS)
force = ti.Vector.field(3, dtype=ti.f32, shape=TOTAL_VERTS)
rest_position = ti.Vector.field(3, dtype=ti.f32, shape=TOTAL_VERTS)
vertex_color = ti.Vector.field(3, dtype=ti.f32, shape=TOTAL_VERTS)
mass = ti.field(dtype=ti.f32, shape=TOTAL_VERTS)
inv_mass = ti.field(dtype=ti.f32, shape=TOTAL_VERTS)
fixed_flag = ti.field(dtype=ti.i32, shape=TOTAL_VERTS)

tet_indices = ti.field(dtype=ti.i32, shape=(N_TETS, 4))
dm_inv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_TETS)
tet_volume = ti.field(dtype=ti.f32, shape=N_TETS)
control_handle_indices = ti.field(dtype=ti.i32, shape=(N_CONTROL_VERTICES, MAX_CONTROL_HANDLE_VERTICES))
control_handle_weights = ti.field(dtype=ti.f32, shape=(N_CONTROL_VERTICES, MAX_CONTROL_HANDLE_VERTICES))
control_handle_offsets = ti.Vector.field(3, dtype=ti.f32, shape=(N_CONTROL_VERTICES, MAX_CONTROL_HANDLE_VERTICES))
control_target_center = ti.Vector.field(3, dtype=ti.f32, shape=N_MODELS)

face_indices = ti.field(dtype=ti.i32, shape=TOTAL_FACE_INDICES)
edge_indices = ti.field(dtype=ti.i32, shape=TOTAL_EDGE_INDICES)
line_vertices = ti.Vector.field(3, dtype=ti.f32, shape=TOTAL_EDGE_INDICES)
selected_points = ti.Vector.field(3, dtype=ti.f32, shape=N_MODELS)
target_points = ti.Vector.field(3, dtype=ti.f32, shape=N_MODELS)
force_lines = ti.Vector.field(3, dtype=ti.f32, shape=N_MODELS * 2)
fixed_point_positions = ti.Vector.field(3, dtype=ti.f32, shape=TOTAL_FIXED_VERTICES)
floor_vertices = ti.Vector.field(3, dtype=ti.f32, shape=4)
floor_indices = ti.field(dtype=ti.i32, shape=6)


rest_position.from_numpy(REST_POSITIONS)
position.from_numpy(REST_POSITIONS)
velocity.from_numpy(np.zeros_like(REST_POSITIONS, dtype=np.float32))
force.from_numpy(np.zeros_like(REST_POSITIONS, dtype=np.float32))
vertex_color.from_numpy(VERTEX_COLORS)
mass.from_numpy(VERTEX_MASS)
inv_mass.from_numpy(VERTEX_INV_MASS)
fixed_flag.from_numpy(FIXED_FLAGS)
tet_indices.from_numpy(TETS)
dm_inv.from_numpy(DM_INV)
tet_volume.from_numpy(TET_VOLUME)
control_handle_indices.from_numpy(CONTROL_HANDLE_INDICES)
control_handle_weights.from_numpy(CONTROL_HANDLE_WEIGHTS)
control_handle_offsets.from_numpy(CONTROL_HANDLE_OFFSETS)
control_target_center.from_numpy(np.zeros((N_MODELS, 3), dtype=np.float32))
face_indices.from_numpy(SURFACE_FACE_INDICES)
edge_indices.from_numpy(EDGE_INDICES)
fixed_point_positions.from_numpy(FIXED_POINT_POSITIONS)
floor_vertices.from_numpy(FLOOR_VERTICES)
floor_indices.from_numpy(FLOOR_INDICES)


@ti.func
def global_vertex(model: ti.i32, local_vertex: ti.i32):
    return model * N_VERTS + local_vertex


@ti.func
def atomic_add_force(vertex_id: ti.i32, value):
    for c in ti.static(range(3)):
        ti.atomic_add(force[vertex_id][c], value[c])


@ti.func
def stvk_stress(deform_grad, mu: ti.f32, la: ti.f32):
    identity = ti.Matrix.identity(ti.f32, 3)
    green = 0.5 * (deform_grad.transpose() @ deform_grad - identity)
    trace = green[0, 0] + green[1, 1] + green[2, 2]
    second_piola = 2.0 * mu * green + la * trace * identity
    return deform_grad @ second_piola


@ti.func
def polar_rotation(deform_grad):
    u, sig, v = ti.svd(deform_grad)
    rot = u @ v.transpose()
    if rot.determinant() < 0.0:
        for i in ti.static(range(3)):
            u[i, 2] = -u[i, 2]
        rot = u @ v.transpose()
    return rot


@ti.func
def corrected_svd(deform_grad):
    u, sig, v = ti.svd(deform_grad)
    if (u @ v.transpose()).determinant() < 0.0:
        for i in ti.static(range(3)):
            u[i, 2] = -u[i, 2]
        sig[2, 2] = -sig[2, 2]
    return u, sig, v


@ti.func
def neo_hookean_stress(deform_grad, mu: ti.f32, la: ti.f32):
    identity = ti.Matrix.identity(ti.f32, 3)
    j = deform_grad.determinant()
    stress = ti.Matrix.zero(ti.f32, 3, 3)
    if j > 0.18:
        inv_t = deform_grad.inverse().transpose()
        stress = mu * (deform_grad - inv_t) + la * ti.log(j) * inv_t
    else:
        rot = polar_rotation(deform_grad)
        stress = 2.0 * mu * (deform_grad - rot) + la * (j - 1.0) * j * inv_t_safe(deform_grad, rot)
    return stress


@ti.func
def inv_t_safe(deform_grad, fallback_rot):
    result = fallback_rot
    if ti.abs(deform_grad.determinant()) > 0.18:
        result = deform_grad.inverse().transpose()
    return result


@ti.func
def corotated_stress(deform_grad, mu: ti.f32, la: ti.f32):
    identity = ti.Matrix.identity(ti.f32, 3)
    u, sig, v = corrected_svd(deform_grad)
    stretch = sig - identity
    trace = stretch[0, 0] + stretch[1, 1] + stretch[2, 2]
    principal_stress = 2.0 * mu * stretch + la * trace * identity
    return u @ principal_stress @ v.transpose()


@ti.kernel
def reset_state():
    for i in position:
        position[i] = rest_position[i]
        velocity[i] = ti.Vector([0.0, 0.0, 0.0])
        force[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def clear_forces(gravity_strength: ti.f32, damping: ti.f32):
    gravity = ti.Vector([0.0, -gravity_strength, 0.0])
    for i in position:
        if fixed_flag[i] != 0:
            force[i] = ti.Vector([0.0, 0.0, 0.0])
        else:
            force[i] = mass[i] * gravity - damping * mass[i] * velocity[i]


@ti.kernel
def compute_hyperelastic_forces(mu: ti.f32, la: ti.f32):
    for model, elem in ti.ndrange(N_MODELS, N_TETS):
        i0 = global_vertex(model, tet_indices[elem, 0])
        i1 = global_vertex(model, tet_indices[elem, 1])
        i2 = global_vertex(model, tet_indices[elem, 2])
        i3 = global_vertex(model, tet_indices[elem, 3])

        ds = ti.Matrix.cols(
            [
                position[i1] - position[i0],
                position[i2] - position[i0],
                position[i3] - position[i0],
            ]
        )
        deform_grad = ds @ dm_inv[elem]

        stress = ti.Matrix.zero(ti.f32, 3, 3)
        if model == 0:
            stress = stvk_stress(deform_grad, mu, la)
        elif model == 1:
            stress = neo_hookean_stress(deform_grad, mu, la)
        else:
            stress = corotated_stress(deform_grad, mu, la)

        h = -tet_volume[elem] * stress @ dm_inv[elem].transpose()
        f1 = ti.Vector([h[0, 0], h[1, 0], h[2, 0]])
        f2 = ti.Vector([h[0, 1], h[1, 1], h[2, 1]])
        f3 = ti.Vector([h[0, 2], h[1, 2], h[2, 2]])
        f0 = -f1 - f2 - f3

        atomic_add_force(i0, f0)
        atomic_add_force(i1, f1)
        atomic_add_force(i2, f2)
        atomic_add_force(i3, f3)


@ti.kernel
def add_control_spring_force(selected_control: ti.i32, stiffness: ti.f32, damping: ti.f32):
    for model in range(N_MODELS):
        target_center = control_target_center[model]
        for slot in ti.static(range(MAX_CONTROL_HANDLE_VERTICES)):
            weight = control_handle_weights[selected_control, slot]
            if weight > 0.0:
                local_vertex = control_handle_indices[selected_control, slot]
                vertex_id = global_vertex(model, local_vertex)
                if fixed_flag[vertex_id] == 0:
                    target = target_center + control_handle_offsets[selected_control, slot]
                    ext = weight * (stiffness * (target - position[vertex_id]) - damping * velocity[vertex_id])
                    force_norm = ext.norm()
                    force_limit = MAX_CONTROL_FORCE * weight
                    if force_norm > force_limit:
                        ext *= force_limit / force_norm
                    atomic_add_force(vertex_id, ext)


@ti.kernel
def integrate_explicit(dt: ti.f32):
    for i in position:
        if fixed_flag[i] != 0:
            position[i] = rest_position[i]
            velocity[i] = ti.Vector([0.0, 0.0, 0.0])
        else:
            velocity[i] += dt * force[i] * inv_mass[i]
            speed = velocity[i].norm()
            if speed > MAX_SPEED:
                velocity[i] *= MAX_SPEED / speed
            position[i] += dt * velocity[i]
            if position[i][1] < FLOOR_Y:
                position[i][1] = FLOOR_Y
                if velocity[i][1] < 0.0:
                    velocity[i][1] = 0.0
                    velocity[i][0] *= 0.82
                    velocity[i][2] *= 0.82


@ti.kernel
def update_line_vertices():
    for i in range(TOTAL_EDGE_INDICES):
        line_vertices[i] = position[edge_indices[i]]


@ti.kernel
def update_visual_helpers(selected_local: ti.i32, target_active: ti.i32):
    for model in range(N_MODELS):
        vertex_id = global_vertex(model, selected_local)
        selected_points[model] = position[vertex_id]
        if target_active != 0:
            target_points[model] = control_target_center[model]
            force_lines[2 * model] = position[vertex_id]
            force_lines[2 * model + 1] = control_target_center[model]
        else:
            target_points[model] = ti.Vector([100.0, 100.0, 100.0])
            force_lines[2 * model] = position[vertex_id]
            force_lines[2 * model + 1] = position[vertex_id]


def simulate_frame(
    frame_dt: float,
    substeps: int,
    young_modulus: float,
    gravity_strength: float,
    damping: float,
    selected_control: int,
    control_active: bool,
    drag_stiffness: float,
    drag_damping: float,
):
    mu, la = lame_parameters(young_modulus, POISSON_RATIO)
    dt = frame_dt / max(1, int(substeps))
    for _ in range(max(1, int(substeps))):
        clear_forces(float(gravity_strength), float(damping))
        compute_hyperelastic_forces(float(mu), float(la))
        if control_active:
            add_control_spring_force(int(selected_control), float(drag_stiffness), float(drag_damping))
        integrate_explicit(float(dt))


def safe_normalize(v: np.ndarray):
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        fallback = np.zeros_like(v, dtype=np.float32)
        fallback[0] = 1.0
        return fallback
    return (v / norm).astype(np.float32)


camera_pos = np.array([12.0, 6.4, 13.8], dtype=np.float32)
camera_target = np.array([0.0, 1.45, 0.0], dtype=np.float32)


def camera_basis():
    forward = safe_normalize(camera_target - camera_pos)
    right = safe_normalize(np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32)))
    up = safe_normalize(np.cross(right, forward))
    return forward, right, up


def rotate_camera_from_mouse(dx: float, dy: float):
    global camera_pos
    offset = camera_pos - camera_target
    radius = max(float(np.linalg.norm(offset)), 1e-6)
    azimuth = np.arctan2(offset[2], offset[0])
    horizontal = np.sqrt(offset[0] * offset[0] + offset[2] * offset[2])
    elevation = np.arctan2(offset[1], horizontal)
    azimuth -= dx * CAMERA_ROTATE_SPEED
    elevation -= dy * CAMERA_ROTATE_SPEED
    elevation = np.clip(elevation, -0.45 * np.pi, 0.45 * np.pi)
    camera_pos = camera_target + radius * np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.sin(elevation),
            np.cos(elevation) * np.sin(azimuth),
        ],
        dtype=np.float32,
    )


def zoom_camera(wheel_delta: float):
    global camera_pos
    offset = camera_pos - camera_target
    radius = max(float(np.linalg.norm(offset)), 1e-6)
    direction = offset / radius
    next_radius = radius * np.exp(-CAMERA_ZOOM_SPEED * wheel_delta)
    next_radius = float(np.clip(next_radius, CAMERA_MIN_DISTANCE, CAMERA_MAX_DISTANCE))
    camera_pos = camera_target + direction * next_radius


def upload_control_targets(targets: np.ndarray):
    control_target_center.from_numpy(targets.astype(np.float32))


def selected_global_vertex(model: int, selected_control: int):
    return model * N_VERTS + int(CONTROL_VERTICES[selected_control])


def print_controls():
    print("=" * 72)
    print("Taichi Lab 3 Bonus B1 - hyperelastic model comparison")
    print(f"models: {', '.join(MODEL_NAMES)}")
    print(f"mesh per model: {N_VERTS} vertices, {N_TETS} tets")
    print("anchor: a small left-end patch, so large rotations are not over-constrained")
    print("Controls:")
    print("  LMB drag : apply the same handle displacement to all models")
    print("  TAB / 1-4: switch selected Rest Shape control vertex")
    print("  RMB drag : orbit camera")
    print("  +/-      : zoom camera")
    print("  SPACE    : pause / resume")
    print("  R        : reset")
    print("  F/E/G    : toggle faces / tetra edges / fixed anchors")
    print("=" * 72)


def draw_gui(gui, paused, frame_dt, substeps, young_modulus, gravity, damping, drag_stiffness, drag_damping):
    gui.begin("Bonus B1 Controls", 0.02, 0.02, 0.30, 0.34)
    reset_requested = gui.button("Reset")
    paused = gui.checkbox("Paused", paused)
    frame_dt = gui.slider_float("frame dt", frame_dt, 1.0 / 240.0, 1.0 / 30.0)
    substeps = int(gui.slider_float("substeps", float(substeps), 6.0, 32.0))
    young_modulus = gui.slider_float("Young modulus", young_modulus, 3000.0, 60000.0)
    gravity = gui.slider_float("gravity", gravity, 0.0, 2.0)
    damping = gui.slider_float("damping", damping, 0.0, 8.0)
    drag_stiffness = gui.slider_float("drag stiffness", drag_stiffness, 300.0, 15000.0)
    drag_damping = gui.slider_float("drag damping", drag_damping, 0.0, 280.0)
    gui.text("Red: StVK")
    gui.text("Blue: Neo-Hookean")
    gui.text("Green: Corotated")
    gui.end()
    return paused, frame_dt, substeps, young_modulus, gravity, damping, drag_stiffness, drag_damping, reset_requested


def main():
    global camera_pos

    print_controls()
    reset_state()
    update_line_vertices()

    paused = False
    show_faces = True
    show_edges = True
    show_fixed = True
    frame_dt = FRAME_DT_DEFAULT
    substeps = SUBSTEPS_DEFAULT
    young_modulus = YOUNG_MODULUS_DEFAULT
    gravity = GRAVITY_DEFAULT
    damping = DAMPING_DEFAULT
    drag_stiffness = DRAG_STIFFNESS_DEFAULT
    drag_damping = DRAG_DAMPING_DEFAULT

    selected_control = 0
    print(f"Selected control 1/{N_CONTROL_VERTICES}: local vertex {CONTROL_VERTICES[selected_control]}")

    window = ti.ui.Window("Taichi FEM Bonus B1 - Hyperelastic Comparison", res=WINDOW_RES, vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    gui = window.get_gui()

    dragging_control = False
    rotating_camera = False
    last_mouse_x = 0.0
    last_mouse_y = 0.0
    drag_start_mouse_x = 0.0
    drag_start_mouse_y = 0.0
    drag_start_targets = np.zeros((N_MODELS, 3), dtype=np.float32)
    control_targets = np.zeros((N_MODELS, 3), dtype=np.float32)

    while window.running:
        for event in window.get_events(ti.ui.PRESS):
            key = event.key
            key_text = str(key).lower()
            if key == ti.ui.ESCAPE:
                window.running = False
            elif key == ti.ui.SPACE:
                paused = not paused
            elif key_text == "r":
                reset_state()
                dragging_control = False
            elif key == ti.ui.TAB:
                selected_control = (selected_control + 1) % N_CONTROL_VERTICES
                dragging_control = False
                print(f"Selected control {selected_control + 1}/{N_CONTROL_VERTICES}: local vertex {CONTROL_VERTICES[selected_control]}")
            elif key_text in ("1", "2", "3", "4"):
                candidate = int(key_text) - 1
                if candidate < N_CONTROL_VERTICES:
                    selected_control = candidate
                    dragging_control = False
                    print(f"Selected control {selected_control + 1}/{N_CONTROL_VERTICES}: local vertex {CONTROL_VERTICES[selected_control]}")
            elif key_text == "f":
                show_faces = not show_faces
            elif key_text == "e":
                show_edges = not show_edges
            elif key_text == "g":
                show_fixed = not show_fixed
            elif key_text in ("=", "+"):
                zoom_camera(1.0)
            elif key_text in ("-", "_"):
                zoom_camera(-1.0)

        if window.is_pressed("=") or window.is_pressed("+"):
            zoom_camera(0.08)
        if window.is_pressed("-") or window.is_pressed("_"):
            zoom_camera(-0.08)

        if window.is_pressed(ti.ui.RMB):
            mouse_x, mouse_y = window.get_cursor_pos()
            if not rotating_camera:
                rotating_camera = True
                last_mouse_x = mouse_x
                last_mouse_y = mouse_y
            else:
                dx = mouse_x - last_mouse_x
                dy = mouse_y - last_mouse_y
                rotate_camera_from_mouse(dx, dy)
                last_mouse_x = mouse_x
                last_mouse_y = mouse_y
        else:
            rotating_camera = False

        if window.is_pressed(ti.ui.LMB):
            mouse_x, mouse_y = window.get_cursor_pos()
            if not dragging_control:
                dragging_control = True
                last_mouse_x = mouse_x
                last_mouse_y = mouse_y
                drag_start_mouse_x = mouse_x
                drag_start_mouse_y = mouse_y
                current_pos = position.to_numpy()
                for model in range(N_MODELS):
                    drag_start_targets[model] = current_pos[selected_global_vertex(model, selected_control)]
                control_targets = drag_start_targets.copy()
                upload_control_targets(control_targets)
            else:
                dx = mouse_x - drag_start_mouse_x
                dy = mouse_y - drag_start_mouse_y
                _, right, up = camera_basis()
                displacement = MOUSE_DRAG_WORLD_SCALE * (dx * right + dy * up)
                control_targets = (drag_start_targets + displacement[None, :]).astype(np.float32)
                upload_control_targets(control_targets)
                last_mouse_x = mouse_x
                last_mouse_y = mouse_y
        else:
            dragging_control = False

        if not paused:
            simulate_frame(
                frame_dt,
                substeps,
                young_modulus,
                gravity,
                damping,
                selected_control,
                dragging_control,
                drag_stiffness,
                drag_damping,
            )

        update_line_vertices()
        update_visual_helpers(int(CONTROL_VERTICES[selected_control]), 1 if dragging_control else 0)

        camera.position(*camera_pos)
        camera.lookat(*camera_target)
        camera.up(0.0, 1.0, 0.0)
        camera.fov(CAMERA_FOV)

        scene.set_camera(camera)
        scene.ambient_light((0.68, 0.68, 0.68))
        scene.point_light((2.8, 7.0, 5.0), (1.2, 1.2, 1.2))
        canvas.set_background_color((0.83, 0.85, 0.88))

        scene.mesh(floor_vertices, indices=floor_indices, color=(0.70, 0.72, 0.74), two_sided=True)
        if show_faces:
            scene.mesh(position, indices=face_indices, per_vertex_color=vertex_color, two_sided=True)
        if show_edges:
            scene.lines(line_vertices, color=(0.05, 0.06, 0.07), width=1.0)
        if show_fixed:
            scene.particles(fixed_point_positions, radius=0.055, color=(0.88, 0.18, 0.16))

        scene.particles(selected_points, radius=0.12, color=(1.0, 0.86, 0.16))
        if dragging_control:
            scene.particles(target_points, radius=0.085, color=(1.0, 0.54, 0.08))
            scene.lines(force_lines, color=(1.0, 0.76, 0.08), width=4.0)

        canvas.scene(scene)
        (
            paused,
            frame_dt,
            substeps,
            young_modulus,
            gravity,
            damping,
            drag_stiffness,
            drag_damping,
            reset_requested,
        ) = draw_gui(gui, paused, frame_dt, substeps, young_modulus, gravity, damping, drag_stiffness, drag_damping)
        if reset_requested:
            reset_state()
            dragging_control = False
        window.show()


if __name__ == "__main__":
    main()
