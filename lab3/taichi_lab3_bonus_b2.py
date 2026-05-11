"""
Taichi Lab 3 bonus B2 - FEM cloth simulation.

The cloth is a 2D manifold embedded in 3D.  Each triangular element stores
2D rest-shape coordinates, so its deformation gradient is a 3x2 matrix.
"""

import sys

import numpy as np
import taichi as ti


USE_CPU = "--cpu" in sys.argv

ti.init(arch=ti.cpu if USE_CPU else ti.gpu, default_fp=ti.f32)


CLOTH_NX, CLOTH_NY = 28, 28
CLOTH_SIZE = np.array([2.0, 2.0], dtype=np.float32)
CLOTH_ORIGIN = np.array([-1.0, 2.45, -1.0], dtype=np.float32)

YOUNG_MODULUS_DEFAULT = 50.0
POISSON_RATIO = 0.3
AREA_DENSITY = 0.5
GRAVITY_DEFAULT = 9.8
DAMPING_DEFAULT = 0.65
FRAME_DT_DEFAULT = 1.0 / 60.0
SUBSTEPS_DEFAULT = 18

WINDOW_RES = (1280, 900)
FLOOR_Y = 0.0
MAX_SPEED = 16.0
DRAG_STIFFNESS_DEFAULT = 180.0
DRAG_DAMPING_DEFAULT = 8.0
MOUSE_DRAG_WORLD_SCALE = 4.0
MAX_CONTROL_FORCE = 800.0
CONTROL_HANDLE_RADIUS = 0.18
MAX_CONTROL_HANDLE_VERTICES = 48
CAMERA_ROTATE_SPEED = 2.4
CAMERA_FOV = 54.0
CAMERA_MIN_DISTANCE = 2.0
CAMERA_MAX_DISTANCE = 12.0
CAMERA_ZOOM_SPEED = 0.18


def lame_parameters(young: float, poisson: float):
    mu = young / (2.0 * (1.0 + poisson))
    la = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return float(mu), float(la)


def get_vertex_id(i: int, j: int):
    return j * (CLOTH_NX + 1) + i


def create_cloth_mesh():
    rest_uv = []
    rest_positions = []
    fixed = []
    colors = []
    dx = CLOTH_SIZE[0] / CLOTH_NX
    dy = CLOTH_SIZE[1] / CLOTH_NY

    for j in range(CLOTH_NY + 1):
        for i in range(CLOTH_NX + 1):
            u = i * dx
            v = j * dy
            rest_uv.append([u, v])
            rest_positions.append([CLOTH_ORIGIN[0] + u, CLOTH_ORIGIN[1], CLOTH_ORIGIN[2] + v])
            fixed.append(1 if j == 0 and (i == 0 or i == CLOTH_NX) else 0)
            t = j / max(1, CLOTH_NY)
            colors.append([0.16 + 0.52 * t, 0.48 + 0.32 * (1.0 - t), 0.90])

    rest_uv = np.array(rest_uv, dtype=np.float32)
    rest_positions = np.array(rest_positions, dtype=np.float32)
    fixed = np.array(fixed, dtype=np.int32)
    colors = np.array(colors, dtype=np.float32)

    triangles = []
    for j in range(CLOTH_NY):
        for i in range(CLOTH_NX):
            v00 = get_vertex_id(i, j)
            v10 = get_vertex_id(i + 1, j)
            v01 = get_vertex_id(i, j + 1)
            v11 = get_vertex_id(i + 1, j + 1)
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])
    triangles = np.array(triangles, dtype=np.int32)

    dm_inv = np.zeros((len(triangles), 2, 2), dtype=np.float32)
    tri_area = np.zeros(len(triangles), dtype=np.float32)
    mass = np.zeros(len(rest_positions), dtype=np.float32)

    for tri_id, tri in enumerate(triangles):
        uv0, uv1, uv2 = rest_uv[tri[0]], rest_uv[tri[1]], rest_uv[tri[2]]
        dm = np.column_stack((uv1 - uv0, uv2 - uv0))
        area = abs(np.linalg.det(dm)) * 0.5
        if area <= 1e-10:
            raise ValueError(f"Degenerate cloth triangle {tri_id}")
        dm_inv[tri_id] = np.linalg.inv(dm).astype(np.float32)
        tri_area[tri_id] = np.float32(area)
        for vertex_id in tri:
            mass[vertex_id] += AREA_DENSITY * area / 3.0

    inv_mass = np.zeros_like(mass, dtype=np.float32)
    active = fixed == 0
    inv_mass[active] = 1.0 / np.maximum(mass[active], 1e-8)

    edge_set = set()
    for tri in triangles:
        edge_set.add(tuple(sorted((int(tri[0]), int(tri[1])))))
        edge_set.add(tuple(sorted((int(tri[1]), int(tri[2])))))
        edge_set.add(tuple(sorted((int(tri[2]), int(tri[0])))))
    edge_indices = []
    for a, b in sorted(edge_set):
        edge_indices.extend([a, b])

    control_targets_uv = [
        np.array([0.5 * CLOTH_SIZE[0], CLOTH_SIZE[1]], dtype=np.float32),
        np.array([0.0, CLOTH_SIZE[1]], dtype=np.float32),
        np.array([CLOTH_SIZE[0], CLOTH_SIZE[1]], dtype=np.float32),
        np.array([0.5 * CLOTH_SIZE[0], 0.62 * CLOTH_SIZE[1]], dtype=np.float32),
    ]
    control_vertices = []
    for target in control_targets_uv:
        dist = np.linalg.norm(rest_uv - target[None, :], axis=1)
        control_vertices.append(int(np.argmin(dist)))

    fixed_vertices = np.nonzero(fixed != 0)[0].astype(np.int32)
    return (
        rest_uv,
        rest_positions,
        colors,
        fixed,
        triangles,
        dm_inv,
        tri_area,
        mass.astype(np.float32),
        inv_mass.astype(np.float32),
        triangles.reshape(-1).astype(np.int32),
        np.array(edge_indices, dtype=np.int32),
        np.array(control_vertices, dtype=np.int32),
        fixed_vertices,
    )


def create_control_handles(rest_uv: np.ndarray, rest_positions: np.ndarray, fixed: np.ndarray, control_vertices: np.ndarray):
    handle_indices = np.zeros((len(control_vertices), MAX_CONTROL_HANDLE_VERTICES), dtype=np.int32)
    handle_weights = np.zeros((len(control_vertices), MAX_CONTROL_HANDLE_VERTICES), dtype=np.float32)
    handle_offsets = np.zeros((len(control_vertices), MAX_CONTROL_HANDLE_VERTICES, 3), dtype=np.float32)

    for control_id, center_vertex in enumerate(control_vertices):
        center_uv = rest_uv[center_vertex]
        center_pos = rest_positions[center_vertex]
        dist = np.linalg.norm(rest_uv - center_uv[None, :], axis=1)
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
            handle_offsets[control_id, slot] = rest_positions[vertex_id] - center_pos
    return handle_indices, handle_weights, handle_offsets


(
    REST_UV,
    REST_POSITIONS,
    VERTEX_COLORS,
    FIXED_FLAGS,
    TRIANGLES,
    DM_INV,
    TRI_AREA,
    VERTEX_MASS,
    VERTEX_INV_MASS,
    FACE_INDICES,
    EDGE_INDICES,
    CONTROL_VERTICES,
    FIXED_VERTICES,
) = create_cloth_mesh()

(
    CONTROL_HANDLE_INDICES,
    CONTROL_HANDLE_WEIGHTS,
    CONTROL_HANDLE_OFFSETS,
) = create_control_handles(REST_UV, REST_POSITIONS, FIXED_FLAGS, CONTROL_VERTICES)

N_VERTS = REST_POSITIONS.shape[0]
N_TRIS = TRIANGLES.shape[0]
N_FACE_INDICES = FACE_INDICES.shape[0]
N_EDGE_INDICES = EDGE_INDICES.shape[0]
N_CONTROL_VERTICES = CONTROL_VERTICES.shape[0]
N_FIXED_VERTICES = FIXED_VERTICES.shape[0]

FLOOR_VERTICES = np.array(
    [
        [-1.35, FLOOR_Y, -1.15],
        [1.35, FLOOR_Y, -1.15],
        [1.35, FLOOR_Y, 1.15],
        [-1.35, FLOOR_Y, 1.15],
    ],
    dtype=np.float32,
)
FLOOR_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)


position = ti.Vector.field(3, dtype=ti.f32, shape=N_VERTS)
velocity = ti.Vector.field(3, dtype=ti.f32, shape=N_VERTS)
force = ti.Vector.field(3, dtype=ti.f32, shape=N_VERTS)
rest_position = ti.Vector.field(3, dtype=ti.f32, shape=N_VERTS)
vertex_color = ti.Vector.field(3, dtype=ti.f32, shape=N_VERTS)
mass = ti.field(dtype=ti.f32, shape=N_VERTS)
inv_mass = ti.field(dtype=ti.f32, shape=N_VERTS)
fixed_flag = ti.field(dtype=ti.i32, shape=N_VERTS)

tri_indices = ti.field(dtype=ti.i32, shape=(N_TRIS, 3))
dm_inv = ti.Matrix.field(2, 2, dtype=ti.f32, shape=N_TRIS)
tri_area = ti.field(dtype=ti.f32, shape=N_TRIS)
control_handle_indices = ti.field(dtype=ti.i32, shape=(N_CONTROL_VERTICES, MAX_CONTROL_HANDLE_VERTICES))
control_handle_weights = ti.field(dtype=ti.f32, shape=(N_CONTROL_VERTICES, MAX_CONTROL_HANDLE_VERTICES))
control_handle_offsets = ti.Vector.field(3, dtype=ti.f32, shape=(N_CONTROL_VERTICES, MAX_CONTROL_HANDLE_VERTICES))

face_indices = ti.field(dtype=ti.i32, shape=N_FACE_INDICES)
edge_indices = ti.field(dtype=ti.i32, shape=N_EDGE_INDICES)
line_vertices = ti.Vector.field(3, dtype=ti.f32, shape=N_EDGE_INDICES)
selected_point = ti.Vector.field(3, dtype=ti.f32, shape=1)
control_target_point = ti.Vector.field(3, dtype=ti.f32, shape=1)
force_line = ti.Vector.field(3, dtype=ti.f32, shape=2)
fixed_point_positions = ti.Vector.field(3, dtype=ti.f32, shape=N_FIXED_VERTICES)
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
tri_indices.from_numpy(TRIANGLES)
dm_inv.from_numpy(DM_INV)
tri_area.from_numpy(TRI_AREA)
control_handle_indices.from_numpy(CONTROL_HANDLE_INDICES)
control_handle_weights.from_numpy(CONTROL_HANDLE_WEIGHTS)
control_handle_offsets.from_numpy(CONTROL_HANDLE_OFFSETS)
face_indices.from_numpy(FACE_INDICES)
edge_indices.from_numpy(EDGE_INDICES)
fixed_point_positions.from_numpy(REST_POSITIONS[FIXED_VERTICES])
floor_vertices.from_numpy(FLOOR_VERTICES)
floor_indices.from_numpy(FLOOR_INDICES)


@ti.func
def atomic_add_force(vertex_id: ti.i32, value):
    for c in ti.static(range(3)):
        ti.atomic_add(force[vertex_id][c], value[c])


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
def compute_cloth_fem_forces(mu: ti.f32, la: ti.f32):
    identity = ti.Matrix.identity(ti.f32, 2)
    for elem in range(N_TRIS):
        i0 = tri_indices[elem, 0]
        i1 = tri_indices[elem, 1]
        i2 = tri_indices[elem, 2]

        ds = ti.Matrix.cols([position[i1] - position[i0], position[i2] - position[i0]])
        deform_grad = ds @ dm_inv[elem]
        green = 0.5 * (deform_grad.transpose() @ deform_grad - identity)
        trace = green[0, 0] + green[1, 1]
        second_piola = 2.0 * mu * green + la * trace * identity
        stress = deform_grad @ second_piola

        h = -tri_area[elem] * stress @ dm_inv[elem].transpose()
        f1 = ti.Vector([h[0, 0], h[1, 0], h[2, 0]])
        f2 = ti.Vector([h[0, 1], h[1, 1], h[2, 1]])
        f0 = -f1 - f2

        atomic_add_force(i0, f0)
        atomic_add_force(i1, f1)
        atomic_add_force(i2, f2)


@ti.kernel
def add_control_spring_force(
    selected_control: ti.i32,
    target_x: ti.f32,
    target_y: ti.f32,
    target_z: ti.f32,
    stiffness: ti.f32,
    damping: ti.f32,
):
    target_center = ti.Vector([target_x, target_y, target_z])
    for slot in range(MAX_CONTROL_HANDLE_VERTICES):
        weight = control_handle_weights[selected_control, slot]
        if weight > 0.0:
            vertex_id = control_handle_indices[selected_control, slot]
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
    for i in range(N_EDGE_INDICES):
        line_vertices[i] = position[edge_indices[i]]


@ti.kernel
def update_visual_helpers(selected_vertex: ti.i32, target_x: ti.f32, target_y: ti.f32, target_z: ti.f32, target_active: ti.i32):
    selected_point[0] = position[selected_vertex]
    if target_active != 0:
        target = ti.Vector([target_x, target_y, target_z])
        control_target_point[0] = target
        force_line[0] = position[selected_vertex]
        force_line[1] = target
    else:
        control_target_point[0] = ti.Vector([100.0, 100.0, 100.0])
        force_line[0] = position[selected_vertex]
        force_line[1] = position[selected_vertex]


def simulate_frame(
    frame_dt: float,
    substeps: int,
    young_modulus: float,
    gravity_strength: float,
    damping: float,
    selected_control: int,
    control_target: np.ndarray,
    control_active: bool,
    drag_stiffness: float,
    drag_damping: float,
):
    mu, la = lame_parameters(young_modulus, POISSON_RATIO)
    dt = frame_dt / max(1, int(substeps))
    tx, ty, tz = [float(v) for v in control_target]
    for _ in range(max(1, int(substeps))):
        clear_forces(float(gravity_strength), float(damping))
        compute_cloth_fem_forces(float(mu), float(la))
        if control_active:
            add_control_spring_force(
                int(selected_control),
                tx,
                ty,
                tz,
                float(drag_stiffness),
                float(drag_damping),
            )
        integrate_explicit(float(dt))


def safe_normalize(v: np.ndarray):
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        fallback = np.zeros_like(v, dtype=np.float32)
        fallback[0] = 1.0
        return fallback
    return (v / norm).astype(np.float32)


camera_pos = np.array([2.8, 2.7, 4.0], dtype=np.float32)
camera_target = np.array([0.0, 1.35, 0.0], dtype=np.float32)


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


def print_controls():
    print("=" * 72)
    print("Taichi Lab 3 Bonus B2 - FEM cloth simulation")
    print(f"cloth mesh: {CLOTH_NX}x{CLOTH_NY} cells, {N_VERTS} vertices, {N_TRIS} triangles")
    print("initial state: horizontal cloth, only two edge corners are fixed")
    print("Controls:")
    print("  LMB drag : move a cloth control handle")
    print("  TAB / 1-4: switch selected Rest Shape control vertex")
    print("  RMB drag : orbit camera")
    print("  +/-      : zoom camera")
    print("  SPACE    : pause / resume")
    print("  R        : reset")
    print("  F/E/G    : toggle cloth faces / edges / fixed anchors")
    print("=" * 72)


def draw_gui(gui, paused, frame_dt, substeps, young_modulus, gravity, damping, drag_stiffness, drag_damping):
    gui.begin("Bonus B2 Cloth Controls", 0.02, 0.02, 0.30, 0.30)
    reset_requested = gui.button("Reset")
    paused = gui.checkbox("Paused", paused)
    frame_dt = gui.slider_float("frame dt", frame_dt, 1.0 / 240.0, 1.0 / 30.0)
    substeps = int(gui.slider_float("substeps", float(substeps), 6.0, 36.0))
    young_modulus = gui.slider_float("Young modulus", young_modulus, 5.0, 180.0)
    gravity = gui.slider_float("gravity", gravity, 0.0, 12.0)
    damping = gui.slider_float("damping", damping, 0.0, 3.0)
    drag_stiffness = gui.slider_float("drag stiffness", drag_stiffness, 20.0, 800.0)
    drag_damping = gui.slider_float("drag damping", drag_damping, 0.0, 40.0)
    gui.text("Triangle FEM uses F in R^(3x2)")
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
    selected_vertex = int(CONTROL_VERTICES[selected_control])
    print(f"Selected control 1/{N_CONTROL_VERTICES}: vertex {selected_vertex}, rest uv={REST_UV[selected_vertex]}")

    window = ti.ui.Window("Taichi FEM Bonus B2 - Cloth", res=WINDOW_RES, vsync=True)
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
    drag_start_target = np.zeros(3, dtype=np.float32)
    control_target = np.zeros(3, dtype=np.float32)

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
                selected_vertex = int(CONTROL_VERTICES[selected_control])
                dragging_control = False
                print(f"Selected control {selected_control + 1}/{N_CONTROL_VERTICES}: vertex {selected_vertex}, rest uv={REST_UV[selected_vertex]}")
            elif key_text in ("1", "2", "3", "4"):
                candidate = int(key_text) - 1
                if candidate < N_CONTROL_VERTICES:
                    selected_control = candidate
                    selected_vertex = int(CONTROL_VERTICES[selected_control])
                    dragging_control = False
                    print(f"Selected control {selected_control + 1}/{N_CONTROL_VERTICES}: vertex {selected_vertex}, rest uv={REST_UV[selected_vertex]}")
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
                drag_start_target = position.to_numpy()[selected_vertex].astype(np.float32)
                control_target = drag_start_target.copy()
            else:
                dx = mouse_x - drag_start_mouse_x
                dy = mouse_y - drag_start_mouse_y
                _, right, up = camera_basis()
                control_target = (
                    drag_start_target
                    + MOUSE_DRAG_WORLD_SCALE * (dx * right + dy * up)
                ).astype(np.float32)
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
                control_target,
                dragging_control,
                drag_stiffness,
                drag_damping,
            )

        update_line_vertices()
        update_visual_helpers(
            selected_vertex,
            float(control_target[0]),
            float(control_target[1]),
            float(control_target[2]),
            1 if dragging_control else 0,
        )

        camera.position(*camera_pos)
        camera.lookat(*camera_target)
        camera.up(0.0, 1.0, 0.0)
        camera.fov(CAMERA_FOV)

        scene.set_camera(camera)
        scene.ambient_light((0.68, 0.68, 0.68))
        scene.point_light((2.2, 4.0, 3.0), (1.15, 1.15, 1.15))
        canvas.set_background_color((0.83, 0.85, 0.88))

        scene.mesh(floor_vertices, indices=floor_indices, color=(0.70, 0.72, 0.74), two_sided=True)
        if show_faces:
            scene.mesh(position, indices=face_indices, per_vertex_color=vertex_color, two_sided=True)
        if show_edges:
            scene.lines(line_vertices, color=(0.05, 0.06, 0.07), width=0.8)
        if show_fixed:
            scene.particles(fixed_point_positions, radius=0.025, color=(0.88, 0.18, 0.16))

        scene.particles(selected_point, radius=0.055, color=(1.0, 0.86, 0.16))
        if dragging_control:
            scene.particles(control_target_point, radius=0.045, color=(1.0, 0.54, 0.08))
            scene.lines(force_line, color=(1.0, 0.76, 0.08), width=4.0)

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
