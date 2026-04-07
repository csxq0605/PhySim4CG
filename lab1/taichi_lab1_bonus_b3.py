"""
Taichi Lab 1 - Bonus B3 mixed geometry scene.

Features:
- four different rigid bodies: box, sphere, cone, and convex hull
- CPU/NumPy rigid-body stepping with impulse-based collision response
- gravity, floor, and four fixed arena walls
- left mouse drag applies force at an off-center local point
- right mouse drag rotates the camera, WASDQE moves the camera
- TAB cycles the selected body, R resets the scene, SPACE pauses
"""

import numpy as np
import taichi as ti


ti.init(arch=ti.gpu, default_fp=ti.f32)


BODY_BOX = 0
BODY_SPHERE = 1
BODY_CONE = 2
BODY_HULL = 3
N_BODIES = 4
SELECTABLE_BODIES = [BODY_SPHERE, BODY_CONE]

FRAME_DT = 1.0 / 60.0
SUBSTEPS = 3
SOLVER_ITERS = 2
DT = FRAME_DT / SUBSTEPS

GRAVITY = np.array([0.0, -9.8, 0.0], dtype=np.float32)
LINEAR_DAMPING = 0.03
ANGULAR_DAMPING = 0.05
BODY_RESTITUTION = 0.42
BODY_FRICTION = 0.20
PLANE_RESTITUTION = 0.24
PLANE_FRICTION = 0.30
EPSILON = 1e-6
POSITION_CORRECTION_PERCENT = 0.68
POSITION_CORRECTION_SLOP = 0.002
PLANE_POSITION_CORRECTION_PERCENT = 0.98
PLANE_POSITION_CORRECTION_SLOP = 0.0005
DRAG_FORCE_SCALE = 1200.0
FORCE_LINE_SCALE = 0.0035
CAMERA_MOVE_SPEED = 2.8
CAMERA_ROTATE_SPEED = 2.4
BOUND_SPHERE_MARGIN = 0.03
SPHERE_RENDER_RADIUS_SCALE = 1.00
FLOOR_SPHERE_CLEARANCE = 0.010
SPHERE_RENDER_LIFT = 0.004

ARENA_HALF_X = 2.25
ARENA_HALF_Z = 2.25
ARENA_HEIGHT = 3.2
FLOOR_Y = 0.0
FLOOR_SIZE = 2.8

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
        [-FLOOR_SIZE, FLOOR_Y, -FLOOR_SIZE],
        [FLOOR_SIZE, FLOOR_Y, -FLOOR_SIZE],
        [FLOOR_SIZE, FLOOR_Y, FLOOR_SIZE],
        [-FLOOR_SIZE, FLOOR_Y, FLOOR_SIZE],
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


def safe_normalize(v: np.ndarray):
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        fallback = np.zeros_like(v, dtype=np.float32)
        fallback[0] = 1.0
        return fallback
    return (v / norm).astype(np.float32)


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


def quat_normalize_np(q: np.ndarray):
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


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


def axis_angle_to_quat(axis: np.ndarray, angle: float):
    axis = safe_normalize(axis)
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)


def orient_triangles_outward(vertices: np.ndarray, triangles: np.ndarray):
    tris = np.asarray(triangles, dtype=np.int32).reshape(-1, 3).copy()
    centroid = vertices.mean(axis=0)
    for tri in tris:
        v0, v1, v2 = vertices[tri]
        normal = np.cross(v1 - v0, v2 - v0)
        face_center = (v0 + v1 + v2) / 3.0
        if np.dot(normal, face_center - centroid) < 0.0:
            tri[1], tri[2] = tri[2], tri[1]
    return tris


def extract_unique_edges(triangles: np.ndarray):
    edges = set()
    for tri in triangles:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i = int(tri[a])
            j = int(tri[b])
            edges.add((min(i, j), max(i, j)))
    return np.array(sorted(edges), dtype=np.int32)


def make_box_geom(ext: np.ndarray):
    vertices = np.array(
        [
            [-ext[0], -ext[1], -ext[2]],
            [ext[0], -ext[1], -ext[2]],
            [ext[0], ext[1], -ext[2]],
            [-ext[0], ext[1], -ext[2]],
            [-ext[0], -ext[1], ext[2]],
            [ext[0], -ext[1], ext[2]],
            [ext[0], ext[1], ext[2]],
            [-ext[0], ext[1], ext[2]],
        ],
        dtype=np.float32,
    )
    triangles = orient_triangles_outward(
        vertices,
        np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 3, 7],
                [0, 7, 4],
                [1, 6, 2],
                [1, 5, 6],
                [0, 4, 5],
                [0, 5, 1],
                [3, 2, 6],
                [3, 6, 7],
            ],
            dtype=np.int32,
        ),
    )
    return vertices, triangles


def make_cone_geom(radius: float, height: float, segments: int):
    apex = np.array([[0.0, 0.75 * height, 0.0]], dtype=np.float32)
    base_center = np.array([[0.0, -0.25 * height, 0.0]], dtype=np.float32)
    ring = []
    for idx in range(segments):
        angle = 2.0 * np.pi * idx / segments
        ring.append([radius * np.cos(angle), -0.25 * height, radius * np.sin(angle)])
    vertices = np.vstack([apex, base_center, np.array(ring, dtype=np.float32)])
    triangles = []
    for idx in range(segments):
        curr = 2 + idx
        nxt = 2 + (idx + 1) % segments
        triangles.append([0, curr, nxt])
        triangles.append([1, nxt, curr])
    return vertices.astype(np.float32), orient_triangles_outward(vertices, np.array(triangles, dtype=np.int32))


def make_tetra_geom(scale: float):
    vertices = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    )
    vertices = scale * vertices / np.sqrt(3.0)
    triangles = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int32)
    return vertices.astype(np.float32), orient_triangles_outward(vertices, triangles)


def make_uv_sphere_geom(radius: float, stacks: int, slices: int):
    vertices = []
    triangles = []
    for stack in range(stacks + 1):
        phi = np.pi * stack / stacks
        y = radius * np.cos(phi)
        ring_radius = radius * np.sin(phi)
        for slc in range(slices):
            theta = 2.0 * np.pi * slc / slices
            x = ring_radius * np.cos(theta)
            z = ring_radius * np.sin(theta)
            vertices.append([x, y, z])

    for stack in range(stacks):
        ring_start = stack * slices
        next_ring_start = (stack + 1) * slices
        for slc in range(slices):
            curr = ring_start + slc
            nxt = ring_start + (slc + 1) % slices
            curr_next = next_ring_start + slc
            nxt_next = next_ring_start + (slc + 1) % slices
            if stack != 0:
                triangles.append([curr, curr_next, nxt])
            if stack != stacks - 1:
                triangles.append([nxt, curr_next, nxt_next])
    return np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.int32)


def point_cloud_inertia_tensor(mass_value: float, vertices: np.ndarray):
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    point_mass = mass_value / max(len(centered), 1)
    inertia = np.zeros((3, 3), dtype=np.float32)
    for point in centered:
        sq_norm = float(np.dot(point, point))
        inertia += point_mass * (sq_norm * np.eye(3, dtype=np.float32) - np.outer(point, point))
    inertia += 1e-4 * np.eye(3, dtype=np.float32)
    return inertia.astype(np.float32), np.linalg.inv(inertia).astype(np.float32)


def sphere_inertia_tensor(mass_value: float, radius_value: float):
    diagonal = 0.4 * mass_value * radius_value * radius_value
    inertia = np.diag([diagonal, diagonal, diagonal]).astype(np.float32)
    inertia_inv = np.diag([1.0 / diagonal, 1.0 / diagonal, 1.0 / diagonal]).astype(np.float32)
    return inertia, inertia_inv


def build_hull_spec(name: str, vertices: np.ndarray, triangles: np.ndarray, mass_value: float, color):
    edges = extract_unique_edges(triangles)
    ext = np.max(np.abs(vertices), axis=0).astype(np.float32)
    inertia, inertia_inv = point_cloud_inertia_tensor(mass_value, vertices)
    return {
        "name": name,
        "kind": "hull",
        "local_vertices": vertices.astype(np.float32),
        "triangles": triangles.astype(np.int32),
        "edges": edges.astype(np.int32),
        "mass": np.float32(mass_value),
        "inertia": inertia,
        "inertia_inv": inertia_inv,
        "bound_radius": np.float32(np.max(np.linalg.norm(vertices, axis=1))),
        "color": color,
        "local_anchor": np.array([0.60 * ext[0], 0.18 * ext[1], 0.42 * ext[2]], dtype=np.float32),
    }


def build_box_spec(ext: np.ndarray, mass_value: float, color):
    vertices, triangles = make_box_geom(ext)
    edges = extract_unique_edges(triangles)
    size = 2.0 * ext
    ixx = mass_value * (size[1] ** 2 + size[2] ** 2) / 12.0
    iyy = mass_value * (size[0] ** 2 + size[2] ** 2) / 12.0
    izz = mass_value * (size[0] ** 2 + size[1] ** 2) / 12.0
    inertia = np.diag([ixx, iyy, izz]).astype(np.float32)
    inertia_inv = np.diag([1.0 / ixx, 1.0 / iyy, 1.0 / izz]).astype(np.float32)
    return {
        "name": "box",
        "kind": "hull",
        "local_vertices": vertices.astype(np.float32),
        "triangles": triangles.astype(np.int32),
        "edges": edges.astype(np.int32),
        "mass": np.float32(mass_value),
        "inertia": inertia,
        "inertia_inv": inertia_inv,
        "bound_radius": np.float32(np.max(np.linalg.norm(vertices, axis=1))),
        "color": color,
        "local_anchor": np.array([0.60 * ext[0], 0.18 * ext[1], 0.42 * ext[2]], dtype=np.float32),
    }


def build_sphere_spec(radius: float, mass_value: float, color):
    inertia, inertia_inv = sphere_inertia_tensor(mass_value, radius)
    return {
        "name": "sphere",
        "kind": "sphere",
        "radius": np.float32(radius),
        "mass": np.float32(mass_value),
        "inertia": inertia,
        "inertia_inv": inertia_inv,
        "bound_radius": np.float32(radius),
        "color": color,
        "local_anchor": np.array([0.55 * radius, 0.18 * radius, 0.40 * radius], dtype=np.float32),
    }


SPHERE_RADIUS = 0.23
CONE_VERTS, CONE_TRIS = make_cone_geom(radius=0.22, height=0.62, segments=10)
HULL_VERTS, HULL_TRIS = make_tetra_geom(scale=0.46)
SPHERE_VERTS, SPHERE_TRIS = make_uv_sphere_geom(radius=SPHERE_RADIUS, stacks=10, slices=16)

BODY_SPECS = [
    build_box_spec(np.array([0.32, 0.24, 0.20], dtype=np.float32), 1.35, (0.86, 0.35, 0.30)),
    build_sphere_spec(SPHERE_RADIUS, 1.00, (0.28, 0.60, 0.86)),
    build_hull_spec("cone", CONE_VERTS, CONE_TRIS, 1.18, (0.30, 0.76, 0.42)),
    build_hull_spec("convex_hull", HULL_VERTS, HULL_TRIS, 1.08, (0.88, 0.66, 0.28)),
]


def build_mesh_layout():
    vertex_offsets = [-1] * N_BODIES
    index_offsets = [-1] * N_BODIES
    index_counts = [0] * N_BODIES
    total_vertices = 0
    total_indices = 0
    index_parts = []
    for body_id, spec in enumerate(BODY_SPECS):
        if spec["kind"] != "hull":
            continue
        vertex_offsets[body_id] = total_vertices
        index_offsets[body_id] = total_indices
        tri_indices = spec["triangles"].reshape(-1) + total_vertices
        index_parts.append(tri_indices.astype(np.int32))
        index_counts[body_id] = len(tri_indices)
        total_vertices += len(spec["local_vertices"])
        total_indices += len(tri_indices)
    index_buffer = np.concatenate(index_parts).astype(np.int32) if total_indices > 0 else np.zeros(0, dtype=np.int32)
    return vertex_offsets, index_offsets, index_counts, total_vertices, total_indices, index_buffer


BODY_VERTEX_OFFSET, BODY_INDEX_OFFSET, BODY_INDEX_COUNT, TOTAL_MESH_VERTICES, TOTAL_MESH_INDICES, INITIAL_INDEX_BUFFER = build_mesh_layout()

mesh_vertices = ti.Vector.field(3, dtype=ti.f32, shape=max(TOTAL_MESH_VERTICES, 1))
mesh_indices = ti.field(dtype=ti.i32, shape=max(TOTAL_MESH_INDICES, 1))
sphere_mesh_vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(SPHERE_VERTS))
sphere_mesh_indices = ti.field(dtype=ti.i32, shape=SPHERE_TRIS.size)
sphere_center_vis = ti.Vector.field(3, dtype=ti.f32, shape=1)
axis_lines = ti.Vector.field(3, dtype=ti.f32, shape=len(WORLD_AXIS_LINES))
arena_line_verts = ti.Vector.field(3, dtype=ti.f32, shape=len(ARENA_LINE_VERTICES))
floor_vertices_ti = ti.Vector.field(3, dtype=ti.f32, shape=4)
floor_indices_ti = ti.field(dtype=ti.i32, shape=6)
selected_com_vis = ti.Vector.field(3, dtype=ti.f32, shape=1)
force_anchor_vis = ti.Vector.field(3, dtype=ti.f32, shape=1)
force_line = ti.Vector.field(3, dtype=ti.f32, shape=2)

axis_lines.from_numpy(WORLD_AXIS_LINES)
arena_line_verts.from_numpy(ARENA_LINE_VERTICES)
floor_vertices_ti.from_numpy(FLOOR_VERTICES)
floor_indices_ti.from_numpy(FLOOR_INDICES)
if TOTAL_MESH_INDICES > 0:
    mesh_indices.from_numpy(INITIAL_INDEX_BUFFER)
sphere_mesh_indices.from_numpy(SPHERE_TRIS.reshape(-1).astype(np.int32))

state_position = np.zeros((N_BODIES, 3), dtype=np.float32)
state_velocity = np.zeros((N_BODIES, 3), dtype=np.float32)
state_angular_velocity = np.zeros((N_BODIES, 3), dtype=np.float32)
state_orientation = np.zeros((N_BODIES, 4), dtype=np.float32)
state_orientation[:, 0] = 1.0
state_mass = np.array([spec["mass"] for spec in BODY_SPECS], dtype=np.float32)
state_inv_mass = (1.0 / state_mass).astype(np.float32)
state_inertia_ref = np.stack([spec["inertia"] for spec in BODY_SPECS], axis=0).astype(np.float32)
state_inertia_ref_inv = np.stack([spec["inertia_inv"] for spec in BODY_SPECS], axis=0).astype(np.float32)
state_applied_force = np.zeros((N_BODIES, 3), dtype=np.float32)
state_force_local_point = np.zeros((N_BODIES, 3), dtype=np.float32)

rng = np.random.default_rng()
camera_pos = np.array([4.2, 2.7, 4.2], dtype=np.float32)
camera_target = np.array([0.0, 1.1, 0.0], dtype=np.float32)


def random_quaternion():
    axis = safe_normalize(rng.normal(size=3).astype(np.float32))
    angle = float(rng.uniform(0.0, np.pi))
    return axis_angle_to_quat(axis, angle)


def get_rotation_matrix(body_id: int):
    return quat_to_matrix_np(state_orientation[body_id])


def get_hull_world_vertices(body_id: int):
    spec = BODY_SPECS[body_id]
    return spec["local_vertices"] @ get_rotation_matrix(body_id).T + state_position[body_id]


def clear_applied_forces():
    state_applied_force.fill(0.0)
    state_force_local_point.fill(0.0)


def set_applied_force(body_id: int, force: np.ndarray, local_point: np.ndarray):
    state_applied_force[body_id] = force.astype(np.float32)
    state_force_local_point[body_id] = local_point.astype(np.float32)


def update_selected_visuals(selected_body: int):
    pos = state_position[selected_body]
    rot = get_rotation_matrix(selected_body)
    anchor_world = pos + rot @ state_force_local_point[selected_body]
    line_end = anchor_world + state_applied_force[selected_body] * FORCE_LINE_SCALE
    selected_com_vis.from_numpy(pos[None, :].astype(np.float32))
    force_anchor_vis.from_numpy(anchor_world[None, :].astype(np.float32))
    force_line.from_numpy(np.vstack([anchor_world, line_end]).astype(np.float32))


def update_render_state():
    if TOTAL_MESH_VERTICES > 0:
        world_vertices = np.zeros((TOTAL_MESH_VERTICES, 3), dtype=np.float32)
        for body_id, spec in enumerate(BODY_SPECS):
            if spec["kind"] != "hull":
                continue
            start = BODY_VERTEX_OFFSET[body_id]
            stop = start + len(spec["local_vertices"])
            world_vertices[start:stop] = spec["local_vertices"] @ get_rotation_matrix(body_id).T + state_position[body_id]
        mesh_vertices.from_numpy(world_vertices)
    sphere_vis = state_position[BODY_SPHERE].copy()
    sphere_vis[1] = max(
        float(sphere_vis[1] + SPHERE_RENDER_LIFT),
        float(FLOOR_Y + BODY_SPECS[BODY_SPHERE]["radius"] + FLOOR_SPHERE_CLEARANCE + SPHERE_RENDER_LIFT),
    )
    sphere_center_vis.from_numpy(sphere_vis[None, :].astype(np.float32))
    sphere_mesh_vertices.from_numpy((SPHERE_VERTS + sphere_vis[None, :]).astype(np.float32))


def impulse_denominator(inv_mass_value: float, inv_inertia: np.ndarray, r: np.ndarray, direction: np.ndarray):
    angular_term = np.cross(inv_inertia @ np.cross(r, direction), r)
    return inv_mass_value + np.dot(angular_term, direction)


def integrate_free_motion_cpu():
    linear_decay = np.float32(np.exp(-LINEAR_DAMPING * DT))
    angular_decay = np.float32(np.exp(-ANGULAR_DAMPING * DT))
    for body_id in range(N_BODIES):
        force = state_applied_force[body_id]
        linear_acc = GRAVITY + force * state_inv_mass[body_id]
        state_velocity[body_id] += linear_acc * DT
        state_velocity[body_id] *= linear_decay
        state_position[body_id] += state_velocity[body_id] * DT

        q = quat_normalize_np(state_orientation[body_id])
        rot = quat_to_matrix_np(q)
        omega = state_angular_velocity[body_id]
        inertia_world = rot @ state_inertia_ref[body_id] @ rot.T
        inertia_world_inv = rot @ state_inertia_ref_inv[body_id] @ rot.T
        torque = np.cross(rot @ state_force_local_point[body_id], force)
        angular_acc = inertia_world_inv @ (torque - np.cross(omega, inertia_world @ omega))
        omega = omega + angular_acc * DT
        omega = omega * angular_decay
        state_angular_velocity[body_id] = omega.astype(np.float32)

        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=np.float32)
        q = q + 0.5 * DT * quat_mul_np(omega_quat, q)
        state_orientation[body_id] = quat_normalize_np(q)


def closest_point_on_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = point - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + (d1 / (d1 - d3)) * ab

    cp = point - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + (d2 / (d2 - d6)) * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


def collision_hull_sphere(hull_id: int, sphere_id: int):
    spec = BODY_SPECS[hull_id]
    verts = get_hull_world_vertices(hull_id)
    center = state_position[sphere_id]
    radius = float(BODY_SPECS[sphere_id]["radius"])

    best_dist2 = np.inf
    best_point = None
    inside = True
    best_signed = -np.inf
    best_normal = None
    best_plane_point = None

    for tri in spec["triangles"]:
        v0, v1, v2 = verts[tri]
        normal = safe_normalize(np.cross(v1 - v0, v2 - v0))
        signed = float(np.dot(center - v0, normal))
        if signed > POSITION_CORRECTION_SLOP:
            inside = False
        if signed > best_signed:
            best_signed = signed
            best_normal = normal
            best_plane_point = center - signed * normal

        closest = closest_point_on_triangle(center, v0, v1, v2)
        dist2 = float(np.dot(center - closest, center - closest))
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_point = closest

    if inside and best_normal is not None:
        normal = (-best_normal).astype(np.float32)
        penetration = radius - best_signed
        sphere_point = center - normal * radius
        contact = 0.5 * (best_plane_point + sphere_point)
        return True, normal, float(penetration), contact.astype(np.float32)

    if best_point is None:
        return False, None, 0.0, None

    distance = float(np.sqrt(best_dist2))
    if distance >= radius:
        return False, None, 0.0, None

    normal = safe_normalize(center - best_point) if distance > EPSILON else safe_normalize(center - state_position[hull_id])
    penetration = radius - distance
    sphere_point = center - normal * radius
    contact = 0.5 * (best_point + sphere_point)
    return True, normal.astype(np.float32), float(penetration), contact.astype(np.float32)


def collision_sphere_sphere(i: int, j: int):
    center_a = state_position[i]
    center_b = state_position[j]
    radius_a = float(BODY_SPECS[i]["radius"])
    radius_b = float(BODY_SPECS[j]["radius"])
    delta = center_b - center_a
    distance = float(np.linalg.norm(delta))
    limit = radius_a + radius_b
    if distance >= limit:
        return False, None, 0.0, None
    normal = (delta / distance).astype(np.float32) if distance > EPSILON else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    penetration = limit - distance
    point_a = center_a + normal * radius_a
    point_b = center_b - normal * radius_b
    contact = 0.5 * (point_a + point_b)
    return True, normal, float(penetration), contact.astype(np.float32)


def collision_hull_hull(i: int, j: int):
    spec_i = BODY_SPECS[i]
    spec_j = BODY_SPECS[j]
    verts_i = get_hull_world_vertices(i)
    verts_j = get_hull_world_vertices(j)
    axes = []

    for tri in spec_i["triangles"]:
        v0, v1, v2 = verts_i[tri]
        normal = np.cross(v1 - v0, v2 - v0)
        if np.dot(normal, normal) > EPSILON:
            axes.append(safe_normalize(normal))
    for tri in spec_j["triangles"]:
        v0, v1, v2 = verts_j[tri]
        normal = np.cross(v1 - v0, v2 - v0)
        if np.dot(normal, normal) > EPSILON:
            axes.append(safe_normalize(normal))

    for a, b in spec_i["edges"]:
        edge_i = verts_i[b] - verts_i[a]
        for c, d in spec_j["edges"]:
            edge_j = verts_j[d] - verts_j[c]
            cross_axis = np.cross(edge_i, edge_j)
            if np.dot(cross_axis, cross_axis) > EPSILON:
                axes.append(safe_normalize(cross_axis))

    min_overlap = np.inf
    best_axis = None
    center_i = state_position[i]
    center_j = state_position[j]
    direction_ij = center_j - center_i

    for axis in axes:
        proj_i = verts_i @ axis
        proj_j = verts_j @ axis
        min_i, max_i = float(np.min(proj_i)), float(np.max(proj_i))
        min_j, max_j = float(np.min(proj_j)), float(np.max(proj_j))
        if max_i < min_j - EPSILON or max_j < min_i - EPSILON:
            return False, None, 0.0, None
        overlap = min(max_i - min_j, max_j - min_i)
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = axis

    if best_axis is None:
        return False, None, 0.0, None

    if np.dot(best_axis, direction_ij) < 0.0:
        best_axis = -best_axis
    normal = safe_normalize(best_axis)
    support_i = verts_i[int(np.argmax(verts_i @ normal))]
    support_j = verts_j[int(np.argmin(verts_j @ normal))]
    contact = 0.5 * (support_i + support_j)
    return True, normal.astype(np.float32), float(min_overlap), contact.astype(np.float32)


def collision_manifold(i: int, j: int):
    center_delta = state_position[j] - state_position[i]
    max_dist = float(BODY_SPECS[i]["bound_radius"] + BODY_SPECS[j]["bound_radius"] + BOUND_SPHERE_MARGIN)
    if np.dot(center_delta, center_delta) > max_dist * max_dist:
        return False, None, 0.0, None

    kind_i = BODY_SPECS[i]["kind"]
    kind_j = BODY_SPECS[j]["kind"]
    if kind_i == "sphere" and kind_j == "sphere":
        return collision_sphere_sphere(i, j)
    if kind_i == "hull" and kind_j == "sphere":
        return collision_hull_sphere(i, j)
    if kind_i == "sphere" and kind_j == "hull":
        collided, normal, penetration, contact = collision_hull_sphere(j, i)
        if not collided:
            return False, None, 0.0, None
        return True, (-normal).astype(np.float32), penetration, contact
    return collision_hull_hull(i, j)


def resolve_body_body_impulse(i: int, j: int, normal: np.ndarray, penetration: float, contact: np.ndarray):
    pos_a = state_position[i]
    pos_b = state_position[j]
    vel_a = state_velocity[i]
    vel_b = state_velocity[j]
    omega_a = state_angular_velocity[i]
    omega_b = state_angular_velocity[j]
    rot_a = get_rotation_matrix(i)
    rot_b = get_rotation_matrix(j)
    inv_inertia_a = rot_a @ state_inertia_ref_inv[i] @ rot_a.T
    inv_inertia_b = rot_b @ state_inertia_ref_inv[j] @ rot_b.T
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

    denom_n = impulse_denominator(inv_mass_a, inv_inertia_a, r_a, normal) + impulse_denominator(inv_mass_b, inv_inertia_b, r_b, normal)
    if denom_n < EPSILON:
        return
    j_n = -(1.0 + BODY_RESTITUTION) * vel_along_normal / denom_n
    impulse = j_n * normal

    tangent = relative_velocity - vel_along_normal * normal
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm > EPSILON:
        tangent = tangent / tangent_norm
        denom_t = impulse_denominator(inv_mass_a, inv_inertia_a, r_a, tangent) + impulse_denominator(inv_mass_b, inv_inertia_b, r_b, tangent)
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
        correction = (POSITION_CORRECTION_PERCENT * correction_mag / (inv_mass_a + inv_mass_b)) * normal
        state_position[i] = (pos_a - inv_mass_a * correction).astype(np.float32)
        state_position[j] = (pos_b + inv_mass_b * correction).astype(np.float32)


def resolve_body_plane_impulse(body_id: int, plane_point: np.ndarray, plane_normal: np.ndarray):
    pos = state_position[body_id]
    vel = state_velocity[body_id]
    omega = state_angular_velocity[body_id]
    rot = get_rotation_matrix(body_id)
    inv_inertia = rot @ state_inertia_ref_inv[body_id] @ rot.T
    inv_mass_value = float(state_inv_mass[body_id])

    if BODY_SPECS[body_id]["kind"] == "sphere":
        radius = float(BODY_SPECS[body_id]["radius"])
        signed_distance = float(np.dot(pos - plane_point, plane_normal) - radius)
        if signed_distance >= 0.0:
            return
        penetration = -signed_distance
        contact = pos - plane_normal * radius
    else:
        verts = get_hull_world_vertices(body_id)
        signed_distances = (verts - plane_point) @ plane_normal
        min_idx = int(np.argmin(signed_distances))
        min_signed = float(signed_distances[min_idx])
        if min_signed >= 0.0:
            return
        penetration = -min_signed
        contact = verts[min_idx]

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

    correction_mag = max(penetration - PLANE_POSITION_CORRECTION_SLOP, 0.0)
    if correction_mag > 0.0:
        state_position[body_id] = (pos + PLANE_POSITION_CORRECTION_PERCENT * correction_mag * plane_normal).astype(np.float32)

    if BODY_SPECS[body_id]["kind"] == "sphere" and plane_normal[1] > 0.99:
        radius = float(BODY_SPECS[body_id]["radius"])
        min_center_y = float(plane_point[1] + radius + FLOOR_SPHERE_CLEARANCE)
        if state_position[body_id, 1] < min_center_y:
            state_position[body_id, 1] = np.float32(min_center_y)
        if state_velocity[body_id, 1] < 0.0:
            state_velocity[body_id, 1] = np.float32(0.0)


def randomize_complex_scene():
    global camera_target

    base_positions = np.array(
        [
            [-0.95, 0.78, -0.30],
            [0.10, 1.48, 0.05],
            [0.98, 0.95, -0.10],
            [-0.18, 1.92, 0.86],
        ],
        dtype=np.float32,
    )
    base_velocities = np.array(
        [
            [0.18, 0.00, 0.12],
            [-0.22, 0.00, -0.10],
            [-0.05, 0.00, 0.18],
            [0.10, -0.06, -0.16],
        ],
        dtype=np.float32,
    )

    state_position[:] = base_positions
    state_position[:, 0] += rng.uniform(-0.08, 0.08, size=N_BODIES).astype(np.float32)
    state_position[:, 2] += rng.uniform(-0.08, 0.08, size=N_BODIES).astype(np.float32)
    state_position[:, 1] += rng.uniform(-0.03, 0.08, size=N_BODIES).astype(np.float32)

    state_velocity[:] = base_velocities + rng.uniform(-0.08, 0.08, size=(N_BODIES, 3)).astype(np.float32)
    state_velocity[:, 1] = np.clip(state_velocity[:, 1], -0.10, 0.10)
    state_angular_velocity[:] = rng.uniform(-0.45, 0.45, size=(N_BODIES, 3)).astype(np.float32)
    state_orientation.fill(0.0)
    state_orientation[:, 0] = 1.0
    state_orientation[BODY_BOX] = random_quaternion()
    state_orientation[BODY_CONE] = random_quaternion()
    state_orientation[BODY_HULL] = random_quaternion()

    clear_applied_forces()
    camera_target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    update_render_state()

    print("Randomized B3 mixed-geometry scene")
    for body_id, spec in enumerate(BODY_SPECS):
        print(f"  {spec['name']}: pos={state_position[body_id]}, vel={state_velocity[body_id]}")


def apply_mouse_drag_force(dx: float, dy: float, selected_body: int):
    local_anchor = BODY_SPECS[selected_body]["local_anchor"]
    forward = safe_normalize(camera_target - camera_pos)
    right = safe_normalize(np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32)))
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    force = DRAG_FORCE_SCALE * (dx * right + dy * up)
    set_applied_force(selected_body, force.astype(np.float32), local_anchor.astype(np.float32))


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
        camera_pos[:] = camera_pos + move
        camera_target[:] = camera_target + move


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
    print("=" * 64)
    print("Bonus B3 - mixed geometry complex scene")
    print("LMB drag : apply force + torque to the selected body")
    print("TAB      : switch selected body (sphere / cone)")
    print("RMB drag : rotate camera around current target")
    print("WASDQE   : move camera")
    print("R        : reset the random scene")
    print("SPACE    : pause / resume")
    print("ESC      : quit")
    print("=" * 64)


def draw_status_panel(gui, selected_body: int, paused: bool):
    gui.begin("Status", 0.02, 0.02, 0.22, 0.12)
    gui.text(f"Mode: {BODY_SPECS[selected_body]['name']}")
    gui.text("Selectable: sphere / cone")
    gui.text(f"State: {'paused' if paused else 'running'}")
    gui.end()


def main():
    randomize_complex_scene()
    print_controls()

    window = ti.ui.Window("Rigid Body Lab1 - Bonus B3", res=(1280, 900), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    gui = window.get_gui()

    paused = False
    dragging = False
    rotating = False
    selected_body_idx = 0
    selected_body = SELECTABLE_BODIES[selected_body_idx]
    last_mouse_x = 0.0
    last_mouse_y = 0.0
    selected_body_color = (1.0, 0.93, 0.36)

    while window.running:
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.ESCAPE:
                window.running = False
            elif event.key == "r":
                randomize_complex_scene()
                selected_body_idx = 0
                selected_body = SELECTABLE_BODIES[selected_body_idx]
            elif event.key == ti.ui.SPACE:
                paused = not paused
            elif event.key == ti.ui.TAB:
                selected_body_idx = (selected_body_idx + 1) % len(SELECTABLE_BODIES)
                selected_body = SELECTABLE_BODIES[selected_body_idx]
                print(f"Selected body: {selected_body} ({BODY_SPECS[selected_body]['name']})")

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

        clear_applied_forces()
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
            for _ in range(SUBSTEPS):
                integrate_free_motion_cpu()
                for _ in range(SOLVER_ITERS):
                    for i in range(N_BODIES):
                        for j in range(i + 1, N_BODIES):
                            collided, normal, penetration, contact = collision_manifold(i, j)
                            if collided:
                                resolve_body_body_impulse(i, j, normal, penetration, contact)
                    for body_id in range(N_BODIES):
                        for plane_point, plane_normal in zip(PLANE_POINTS, PLANE_NORMALS):
                            resolve_body_plane_impulse(body_id, plane_point, plane_normal)

        update_render_state()
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

        for body_id, spec in enumerate(BODY_SPECS):
            fill_color = selected_body_color if body_id == selected_body else spec["color"]
            if spec["kind"] == "sphere":
                scene.mesh(
                    sphere_mesh_vertices,
                    sphere_mesh_indices,
                    color=fill_color,
                )
                wire_color = (0.98, 0.82, 0.20) if body_id == selected_body else (0.0, 0.0, 0.0)
                scene.mesh(
                    sphere_mesh_vertices,
                    sphere_mesh_indices,
                    color=wire_color,
                    show_wireframe=True,
                )
            else:
                scene.mesh(mesh_vertices, mesh_indices, color=fill_color, index_offset=BODY_INDEX_OFFSET[body_id], index_count=BODY_INDEX_COUNT[body_id])
                wire_color = (0.98, 0.82, 0.20) if body_id == selected_body else (0.0, 0.0, 0.0)
                scene.mesh(
                    mesh_vertices,
                    mesh_indices,
                    color=wire_color,
                    index_offset=BODY_INDEX_OFFSET[body_id],
                    index_count=BODY_INDEX_COUNT[body_id],
                    show_wireframe=True,
                )

        scene.particles(selected_com_vis, radius=0.035, color=(0.98, 0.82, 0.18))
        if dragging:
            scene.particles(force_anchor_vis, radius=0.028, color=(1.0, 0.84, 0.20))
            scene.lines(force_line, color=(1.0, 0.84, 0.20), width=4.0)

        canvas.scene(scene)
        draw_status_panel(gui, selected_body, paused)
        window.show()


if __name__ == "__main__":
    main()
