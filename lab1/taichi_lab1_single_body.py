"""
Taichi Lab 1 - single rigid body demo.

Features:
- one rigid box with random initial state
- semi-implicit Euler integration:
  explicit update for v / omega, implicit update for x / q
- orientation stored as a quaternion
- left mouse drag applies a force at an off-center point, which also
  generates torque
- press R to randomize the body again
- press SPACE to pause / resume
"""

import numpy as np
import taichi as ti


ti.init(arch=ti.gpu, default_fp=ti.f32)


N_BODIES = 1
FRAME_DT = 1.0 / 60.0
SUBSTEPS = 4
DT = FRAME_DT / SUBSTEPS
GRAVITY = ti.Vector([0.0, 0.0, 0.0])
LINEAR_DAMPING = 0.22
ANGULAR_DAMPING = 0.28
DRAG_FORCE_SCALE = 900.0
FORCE_LINE_SCALE = 0.004
AUTO_RESET_RADIUS = 6.0
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
        [1.2, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.2, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.2],
    ],
    dtype=np.float32,
)


cube_local_verts = ti.Vector.field(3, dtype=ti.f32, shape=8)
mesh_vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
mesh_indices = ti.field(dtype=ti.i32, shape=36)
axis_lines = ti.Vector.field(3, dtype=ti.f32, shape=6)
force_line = ti.Vector.field(3, dtype=ti.f32, shape=2)
force_anchor_vis = ti.Vector.field(3, dtype=ti.f32, shape=1)
com_vis = ti.Vector.field(3, dtype=ti.f32, shape=1)

position = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
velocity = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
angular_velocity = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
half_extent = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
mass = ti.field(dtype=ti.f32, shape=N_BODIES)
inv_mass = ti.field(dtype=ti.f32, shape=N_BODIES)
orientation = ti.Vector.field(4, dtype=ti.f32, shape=N_BODIES)
body_inertia_ref = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_BODIES)
body_inertia_ref_inv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_BODIES)
applied_force = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
force_local_point = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)

cube_local_verts.from_numpy(CUBE_LOCAL_VERTICES)
mesh_indices.from_numpy(CUBE_INDICES)
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
def clear_applied_force():
    applied_force[0] = ti.Vector([0.0, 0.0, 0.0])
    force_local_point[0] = ti.Vector([0.0, 0.0, 0.0])
    force_anchor_vis[0] = position[0]
    force_line[0] = position[0]
    force_line[1] = position[0]


@ti.kernel
def set_applied_force(
    force: ti.types.vector(3, ti.f32),
    local_point: ti.types.vector(3, ti.f32),
):
    applied_force[0] = force
    force_local_point[0] = local_point


@ti.kernel
def integrate():
    for i in range(N_BODIES):
        # Semi-implicit Euler for translation:
        # first update v explicitly, then use the new v to update x.
        linear_acc = GRAVITY + applied_force[i] * inv_mass[i]
        velocity[i] += linear_acc * DT
        velocity[i] *= ti.exp(-LINEAR_DAMPING * DT)
        position[i] += velocity[i] * DT

        q = quat_normalize(orientation[i])
        R = quat_to_matrix(q)
        omega = angular_velocity[i]
        torque = (R @ force_local_point[i]).cross(applied_force[i])
        I_world = R @ body_inertia_ref[i] @ R.transpose()
        I_world_inv = R @ body_inertia_ref_inv[i] @ R.transpose()

        # Semi-implicit Euler for rotation:
        # first update omega explicitly, then use the new omega to update q.
        angular_acc = I_world_inv @ (torque - omega.cross(I_world @ omega))
        omega += angular_acc * DT
        omega *= ti.exp(-ANGULAR_DAMPING * DT)
        angular_velocity[i] = omega

        omega_quat = ti.Vector([0.0, omega[0], omega[1], omega[2]])
        q = q + 0.5 * DT * quat_mul(omega_quat, q)
        orientation[i] = quat_normalize(q)


@ti.kernel
def update_mesh_vertices():
    pos = position[0]
    rot = quat_to_matrix(orientation[0])
    ext = half_extent[0]
    for k in range(8):
        lv = cube_local_verts[k]
        local = ti.Vector([lv[0] * ext[0], lv[1] * ext[1], lv[2] * ext[2]])
        mesh_vertices[k] = rot @ local + pos
    com_vis[0] = pos
    force_anchor_vis[0] = pos + rot @ force_local_point[0]
    force_line[0] = force_anchor_vis[0]
    force_line[1] = force_anchor_vis[0] + applied_force[0] * FORCE_LINE_SCALE


def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)


def box_inertia_tensor(mass_value: float, ext: np.ndarray):
    size = 2.0 * ext
    ixx = mass_value * (size[1] ** 2 + size[2] ** 2) / 12.0
    iyy = mass_value * (size[0] ** 2 + size[2] ** 2) / 12.0
    izz = mass_value * (size[0] ** 2 + size[1] ** 2) / 12.0
    inertia = np.diag([ixx, iyy, izz]).astype(np.float32)
    inertia_inv = np.diag([1.0 / ixx, 1.0 / iyy, 1.0 / izz]).astype(np.float32)
    return inertia, inertia_inv


rng = np.random.default_rng()


def randomize_single_body():
    ext = rng.uniform(0.22, 0.42, size=3).astype(np.float32)
    pos = rng.uniform([-0.35, -0.20, -0.35], [0.35, 0.35, 0.35]).astype(np.float32)
    vel = rng.uniform(-0.28, 0.28, size=3).astype(np.float32)
    ang_vel = rng.uniform(-0.85, 0.85, size=3).astype(np.float32)
    body_mass = np.float32(rng.uniform(0.8, 1.6))

    axis = rng.normal(size=3).astype(np.float32)
    axis /= np.linalg.norm(axis)
    angle = float(rng.uniform(0.0, np.pi))
    quat = axis_angle_to_quat(axis, angle)

    inertia, inertia_inv = box_inertia_tensor(float(body_mass), ext)

    position.from_numpy(pos[None, :])
    velocity.from_numpy(vel[None, :])
    orientation.from_numpy(quat[None, :])
    angular_velocity.from_numpy(ang_vel[None, :])
    half_extent.from_numpy(ext[None, :])
    mass.from_numpy(np.array([body_mass], dtype=np.float32))
    inv_mass.from_numpy(np.array([1.0 / body_mass], dtype=np.float32))
    body_inertia_ref.from_numpy(inertia[None, :, :])
    body_inertia_ref_inv.from_numpy(inertia_inv[None, :, :])

    clear_applied_force()
    update_mesh_vertices()

    print("Randomized initial state")
    print(f"  position = {pos}")
    print(f"  velocity = {vel}")
    print(f"  angular_velocity = {ang_vel}")
    print(f"  orientation(q) = {quat}")
    print(f"  half_extent = {ext}")
    print(f"  mass = {float(body_mass):.3f}")


camera_pos = np.array([3.0, 2.0, 3.0], dtype=np.float32)
camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)


def apply_mouse_drag_force(dx: float, dy: float):
    ext = half_extent[0].to_numpy()

    local_anchor = np.array([0.65 * ext[0], 0.20 * ext[1], 0.45 * ext[2]], dtype=np.float32)
    camera_forward = camera_target - camera_pos
    camera_forward /= np.linalg.norm(camera_forward)
    camera_right = np.cross(camera_forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    camera_right /= np.linalg.norm(camera_right)
    camera_up = np.cross(camera_right, camera_forward)
    camera_up /= np.linalg.norm(camera_up)

    # Match screen drag direction: downward mouse drag should push downward.
    force = DRAG_FORCE_SCALE * (dx * camera_right + dy * camera_up)
    set_applied_force(ti.Vector(force.astype(np.float32)), ti.Vector(local_anchor))


def update_camera_from_keyboard(window):
    global camera_pos, camera_target

    forward = camera_target - camera_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    right /= np.linalg.norm(right)
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
    print("=" * 48)
    print("Single rigid body demo")
    print("LMB drag : apply force at an off-center local point")
    print("          torque is computed by tau = (R r) x f")
    print("WASDQE   : move camera (forward/back/left/right/up/down)")
    print("RMB drag : rotate camera around current target")
    print("R        : randomize initial state")
    print("SPACE    : pause / resume")
    print("ESC      : quit")
    print("=" * 48)


def main():
    randomize_single_body()
    print_controls()

    window = ti.ui.Window("Rigid Body Lab1 - Single Body", res=(1280, 900), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    paused = False
    dragging = False
    rotating = False
    last_mouse_x = 0.0
    last_mouse_y = 0.0

    while window.running:
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.ESCAPE:
                window.running = False
            elif event.key == "r":
                randomize_single_body()
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

        if window.is_pressed(ti.ui.LMB):
            mouse_x, mouse_y = window.get_cursor_pos()
            if not dragging:
                dragging = True
                last_mouse_x = mouse_x
                last_mouse_y = mouse_y
                clear_applied_force()
            else:
                dx = mouse_x - last_mouse_x
                dy = mouse_y - last_mouse_y
                if abs(dx) + abs(dy) > 0.0:
                    apply_mouse_drag_force(dx, dy)
                else:
                    clear_applied_force()
                last_mouse_x = mouse_x
                last_mouse_y = mouse_y
        else:
            dragging = False
            clear_applied_force()

        if not paused:
            for _ in range(SUBSTEPS):
                integrate()

            if np.linalg.norm(position[0].to_numpy()) > AUTO_RESET_RADIUS:
                randomize_single_body()

        update_mesh_vertices()

        camera.position(*camera_pos)
        camera.lookat(*camera_target)
        camera.up(0.0, 1.0, 0.0)

        scene.set_camera(camera)
        scene.ambient_light((0.65, 0.65, 0.65))
        scene.point_light((4.0, 5.0, 4.0), (1.2, 1.2, 1.2))

        canvas.set_background_color((0.82, 0.84, 0.88))
        scene.lines(axis_lines, color=(0.45, 0.45, 0.45), width=1.5)

        scene.mesh(mesh_vertices, mesh_indices, color=(0.32, 0.68, 0.84))
        scene.mesh(
            mesh_vertices,
            mesh_indices,
            color=(0.0, 0.0, 0.0),
            show_wireframe=True,
        )
        scene.particles(com_vis, radius=0.035, color=(0.95, 0.35, 0.30))

        if dragging:
            scene.particles(force_anchor_vis, radius=0.03, color=(1.0, 0.82, 0.22))
            scene.lines(force_line, color=(1.0, 0.82, 0.22), width=4.0)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
