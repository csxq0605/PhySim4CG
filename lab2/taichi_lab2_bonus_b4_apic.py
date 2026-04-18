"""
Taichi FLIP / APIC fluid demo - Lab 2 bonus B4

Implemented features:
- baseline Lab 2 FLIP solver configuration
- PIC / FLIP blending through flipRatio
- APIC transfer on the same MAC grid
"""

import numpy as np
import taichi as ti


ti.init(arch=ti.gpu, default_fp=ti.f32)


# Domain and simulation constants
GRID_RES = 24
DOMAIN_MIN = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
DOMAIN_MAX = np.array([0.5, 0.5, 0.5], dtype=np.float32)
DOMAIN_SIZE = DOMAIN_MAX - DOMAIN_MIN
CELL_SIZE = DOMAIN_SIZE[0] / GRID_RES
INV_CELL_SIZE = 1.0 / CELL_SIZE

DEFAULT_FRAME_DT = 0.025
DEFAULT_SUBSTEPS = 3
DEFAULT_PRESSURE_ITERS = 30
DEFAULT_OVER_RELAXATION = 1.9
DEFAULT_FLIP_RATIO = 0.95
DEFAULT_COLOR_MODE = 1
VELOCITY_EXTRAPOLATION_ITERS = GRID_RES
COMPONENT_LABEL_RELAX_ITERS = GRID_RES * 3
TRANSFER_MODE_BLEND = 1
TRANSFER_MODE_APIC = 1
FLUID_DENSITY = 1.0
DRIFT_COMPENSATION_GAIN = 1.0
GRAVITY = ti.Vector([0.0, -9.81, 0.0])
WINDOW_RES = (1440, 960)
CAMERA_ROTATE_SPEED = 2.4
BASE_PARTICLE_COLOR = np.array([0.18, 0.52, 0.88], dtype=np.float32)

COLOR_MODE_SPEED = 1
COLOR_MODE_DENSITY = 2
COLOR_MODE_PRESSURE = 3

PARTICLE_RADIUS = 0.27 * CELL_SIZE
PARTICLE_DRAW_RADIUS = PARTICLE_RADIUS * 0.76
WATER_RATIO = np.array([0.56, 0.78, 0.56], dtype=np.float32)
PARTICLE_MIN_BOUND = DOMAIN_MIN + CELL_SIZE + PARTICLE_RADIUS
PARTICLE_MAX_BOUND = DOMAIN_MAX - CELL_SIZE - PARTICLE_RADIUS
DOMAIN_MIN_X = float(DOMAIN_MIN[0])
DOMAIN_MIN_Y = float(DOMAIN_MIN[1])
DOMAIN_MIN_Z = float(DOMAIN_MIN[2])
BOUNDARY_MIN_X = float(PARTICLE_MIN_BOUND[0])
BOUNDARY_MIN_Y = float(PARTICLE_MIN_BOUND[1])
BOUNDARY_MIN_Z = float(PARTICLE_MIN_BOUND[2])
BOUNDARY_MAX_X = float(PARTICLE_MAX_BOUND[0])
BOUNDARY_MAX_Y = float(PARTICLE_MAX_BOUND[1])
BOUNDARY_MAX_Z = float(PARTICLE_MAX_BOUND[2])
HIDDEN_PARTICLE_POS = np.array([10.0, 10.0, 10.0], dtype=np.float32)
CHEAP_CULL_MAX_PARTICLES = 2

EMPTY_CELL = 0
FLUID_CELL = 1
SOLID_CELL = 2
MAX_CELL_COMPONENTS = GRID_RES * GRID_RES * GRID_RES

SEPARATION_CELL_SIZE = 2.2 * PARTICLE_RADIUS
SEPARATION_GRID_RES = np.ceil(DOMAIN_SIZE / SEPARATION_CELL_SIZE).astype(np.int32) + 1
SEPARATION_GRID_X = int(SEPARATION_GRID_RES[0])
SEPARATION_GRID_Y = int(SEPARATION_GRID_RES[1])
SEPARATION_GRID_Z = int(SEPARATION_GRID_RES[2])
MAX_PARTICLES_PER_SEP_CELL = 64


def create_initial_particle_block():
    tank = DOMAIN_SIZE
    fluid_size = WATER_RATIO * tank
    dx = 2.0 * PARTICLE_RADIUS
    dy = np.sqrt(3.0) * PARTICLE_RADIUS
    dz = dx

    usable = fluid_size - 2.0 * CELL_SIZE - 2.0 * PARTICLE_RADIUS
    num_x = max(1, int(np.floor(usable[0] / dx)))
    num_y = max(1, int(np.floor(usable[1] / dy)))
    num_z = max(1, int(np.floor(usable[2] / dz)))

    positions = []
    base = DOMAIN_MIN + CELL_SIZE + PARTICLE_RADIUS
    for i in range(num_x):
        for j in range(num_y):
            for k in range(num_z):
                x = base[0] + dx * i + (PARTICLE_RADIUS if j % 2 == 1 else 0.0)
                y = base[1] + dy * j
                z = base[2] + dz * k + (PARTICLE_RADIUS if j % 2 == 1 else 0.0)
                positions.append([x, y, z])
    return np.array(positions, dtype=np.float32)


def create_tank_line_vertices():
    x0, y0, z0 = DOMAIN_MIN.tolist()
    x1, y1, z1 = DOMAIN_MAX.tolist()
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    line_vertices = np.zeros((len(edges) * 2, 3), dtype=np.float32)
    for edge_id, (a, b) in enumerate(edges):
        line_vertices[2 * edge_id] = corners[a]
        line_vertices[2 * edge_id + 1] = corners[b]
    return line_vertices


INITIAL_PARTICLE_POS = create_initial_particle_block()
INITIAL_PARTICLE_VEL = np.zeros_like(INITIAL_PARTICLE_POS, dtype=np.float32)
INITIAL_PARTICLE_COLOR = np.tile(
    BASE_PARTICLE_COLOR[None, :],
    (INITIAL_PARTICLE_POS.shape[0], 1),
)
TANK_LINE_VERTICES = create_tank_line_vertices()
N_PARTICLES = INITIAL_PARTICLE_POS.shape[0]


# Particle fields
particle_pos = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)
particle_vel = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)
particle_color = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)
particle_c = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_PARTICLES)
particle_active = ti.field(dtype=ti.i32, shape=N_PARTICLES)


# MAC grid fields
grid_u = ti.field(dtype=ti.f32, shape=(GRID_RES + 1, GRID_RES, GRID_RES))
grid_v = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES + 1, GRID_RES))
grid_w = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES, GRID_RES + 1))

grid_u_prev = ti.field(dtype=ti.f32, shape=(GRID_RES + 1, GRID_RES, GRID_RES))
grid_v_prev = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES + 1, GRID_RES))
grid_w_prev = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES, GRID_RES + 1))

grid_u_weight = ti.field(dtype=ti.f32, shape=(GRID_RES + 1, GRID_RES, GRID_RES))
grid_v_weight = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES + 1, GRID_RES))
grid_w_weight = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES, GRID_RES + 1))
grid_u_tmp = ti.field(dtype=ti.f32, shape=(GRID_RES + 1, GRID_RES, GRID_RES))
grid_v_tmp = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES + 1, GRID_RES))
grid_w_tmp = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES, GRID_RES + 1))
grid_u_valid = ti.field(dtype=ti.i32, shape=(GRID_RES + 1, GRID_RES, GRID_RES))
grid_v_valid = ti.field(dtype=ti.i32, shape=(GRID_RES, GRID_RES + 1, GRID_RES))
grid_w_valid = ti.field(dtype=ti.i32, shape=(GRID_RES, GRID_RES, GRID_RES + 1))
grid_u_valid_tmp = ti.field(dtype=ti.i32, shape=(GRID_RES + 1, GRID_RES, GRID_RES))
grid_v_valid_tmp = ti.field(dtype=ti.i32, shape=(GRID_RES, GRID_RES + 1, GRID_RES))
grid_w_valid_tmp = ti.field(dtype=ti.i32, shape=(GRID_RES, GRID_RES, GRID_RES + 1))

cell_pressure = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES, GRID_RES))
cell_type = ti.field(dtype=ti.i32, shape=(GRID_RES, GRID_RES, GRID_RES))
cell_particle_density = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES, GRID_RES))
cell_particle_count = ti.field(dtype=ti.i32, shape=(GRID_RES, GRID_RES, GRID_RES))
cell_component_label = ti.field(dtype=ti.i32, shape=(GRID_RES, GRID_RES, GRID_RES))
cell_component_label_tmp = ti.field(dtype=ti.i32, shape=(GRID_RES, GRID_RES, GRID_RES))
component_particle_sum = ti.field(dtype=ti.i32, shape=MAX_CELL_COMPONENTS)

rest_density = ti.field(dtype=ti.f32, shape=())
max_speed_value = ti.field(dtype=ti.f32, shape=())
max_density_value = ti.field(dtype=ti.f32, shape=())
max_pressure_value = ti.field(dtype=ti.f32, shape=())
main_component_label = ti.field(dtype=ti.i32, shape=())
main_component_particle_sum = ti.field(dtype=ti.i32, shape=())


# Render fields
tank_line_verts = ti.Vector.field(3, dtype=ti.f32, shape=TANK_LINE_VERTICES.shape[0])


# Particle separation helpers
separation_count = ti.field(
    dtype=ti.i32,
    shape=(SEPARATION_GRID_X, SEPARATION_GRID_Y, SEPARATION_GRID_Z),
)
separation_particles = ti.field(
    dtype=ti.i32,
    shape=(
        SEPARATION_GRID_X,
        SEPARATION_GRID_Y,
        SEPARATION_GRID_Z,
        MAX_PARTICLES_PER_SEP_CELL,
    ),
)
particle_separation_delta = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)


@ti.func
def centered_cell_coord(pos):
    return ti.Vector(
        [
            (pos[0] - DOMAIN_MIN_X) * INV_CELL_SIZE - 0.5,
            (pos[1] - DOMAIN_MIN_Y) * INV_CELL_SIZE - 0.5,
            (pos[2] - DOMAIN_MIN_Z) * INV_CELL_SIZE - 0.5,
        ]
    )


@ti.func
def particle_cell_index(pos):
    coord = (pos - ti.Vector([DOMAIN_MIN_X, DOMAIN_MIN_Y, DOMAIN_MIN_Z])) * INV_CELL_SIZE
    return ti.Vector(
        [
            ti.max(0, ti.min(GRID_RES - 1, ti.cast(ti.floor(coord[0]), ti.i32))),
            ti.max(0, ti.min(GRID_RES - 1, ti.cast(ti.floor(coord[1]), ti.i32))),
            ti.max(0, ti.min(GRID_RES - 1, ti.cast(ti.floor(coord[2]), ti.i32))),
        ]
    )


@ti.func
def cell_linear_index(i, j, k):
    return (i * GRID_RES + j) * GRID_RES + k


@ti.func
def separation_cell_coord(pos):
    return ti.Vector(
        [
            (pos[0] - DOMAIN_MIN_X) / SEPARATION_CELL_SIZE,
            (pos[1] - DOMAIN_MIN_Y) / SEPARATION_CELL_SIZE,
            (pos[2] - DOMAIN_MIN_Z) / SEPARATION_CELL_SIZE,
        ]
    )


@ti.func
def u_face_coord(pos):
    return ti.Vector(
        [
            (pos[0] - DOMAIN_MIN_X) * INV_CELL_SIZE,
            (pos[1] - DOMAIN_MIN_Y) * INV_CELL_SIZE - 0.5,
            (pos[2] - DOMAIN_MIN_Z) * INV_CELL_SIZE - 0.5,
        ]
    )


@ti.func
def v_face_coord(pos):
    return ti.Vector(
        [
            (pos[0] - DOMAIN_MIN_X) * INV_CELL_SIZE - 0.5,
            (pos[1] - DOMAIN_MIN_Y) * INV_CELL_SIZE,
            (pos[2] - DOMAIN_MIN_Z) * INV_CELL_SIZE - 0.5,
        ]
    )


@ti.func
def w_face_coord(pos):
    return ti.Vector(
        [
            (pos[0] - DOMAIN_MIN_X) * INV_CELL_SIZE - 0.5,
            (pos[1] - DOMAIN_MIN_Y) * INV_CELL_SIZE - 0.5,
            (pos[2] - DOMAIN_MIN_Z) * INV_CELL_SIZE,
        ]
    )


@ti.func
def sample_u_field(field: ti.template(), pos):
    coord = u_face_coord(pos)
    base = ti.cast(ti.floor(coord), ti.i32)
    frac = coord - ti.cast(base, ti.f32)
    value = 0.0
    for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
        idx = base + ti.Vector([ox, oy, oz])
        if (
            0 <= idx[0] < GRID_RES + 1
            and 0 <= idx[1] < GRID_RES
            and 0 <= idx[2] < GRID_RES
        ):
            wx = frac[0] if ox == 1 else 1.0 - frac[0]
            wy = frac[1] if oy == 1 else 1.0 - frac[1]
            wz = frac[2] if oz == 1 else 1.0 - frac[2]
            value += wx * wy * wz * field[idx[0], idx[1], idx[2]]
    return value


@ti.func
def sample_v_field(field: ti.template(), pos):
    coord = v_face_coord(pos)
    base = ti.cast(ti.floor(coord), ti.i32)
    frac = coord - ti.cast(base, ti.f32)
    value = 0.0
    for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
        idx = base + ti.Vector([ox, oy, oz])
        if (
            0 <= idx[0] < GRID_RES
            and 0 <= idx[1] < GRID_RES + 1
            and 0 <= idx[2] < GRID_RES
        ):
            wx = frac[0] if ox == 1 else 1.0 - frac[0]
            wy = frac[1] if oy == 1 else 1.0 - frac[1]
            wz = frac[2] if oz == 1 else 1.0 - frac[2]
            value += wx * wy * wz * field[idx[0], idx[1], idx[2]]
    return value


@ti.func
def sample_w_field(field: ti.template(), pos):
    coord = w_face_coord(pos)
    base = ti.cast(ti.floor(coord), ti.i32)
    frac = coord - ti.cast(base, ti.f32)
    value = 0.0
    for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
        idx = base + ti.Vector([ox, oy, oz])
        if (
            0 <= idx[0] < GRID_RES
            and 0 <= idx[1] < GRID_RES
            and 0 <= idx[2] < GRID_RES + 1
        ):
            wx = frac[0] if ox == 1 else 1.0 - frac[0]
            wy = frac[1] if oy == 1 else 1.0 - frac[1]
            wz = frac[2] if oz == 1 else 1.0 - frac[2]
            value += wx * wy * wz * field[idx[0], idx[1], idx[2]]
    return value


@ti.func
def sample_centered_scalar_field(field: ti.template(), pos):
    coord = centered_cell_coord(pos)
    base = ti.cast(ti.floor(coord), ti.i32)
    frac = coord - ti.cast(base, ti.f32)
    value = 0.0
    weight_sum = 0.0
    for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
        idx = base + ti.Vector([ox, oy, oz])
        if 0 <= idx[0] < GRID_RES and 0 <= idx[1] < GRID_RES and 0 <= idx[2] < GRID_RES:
            weight = trilinear_weight(frac, ox, oy, oz)
            value += weight * field[idx[0], idx[1], idx[2]]
            weight_sum += weight
    if weight_sum > 0.0:
        value /= weight_sum
    return value


@ti.func
def u_face_world_pos(i, j, k):
    return ti.Vector(
        [
            DOMAIN_MIN_X + ti.cast(i, ti.f32) * CELL_SIZE,
            DOMAIN_MIN_Y + (ti.cast(j, ti.f32) + 0.5) * CELL_SIZE,
            DOMAIN_MIN_Z + (ti.cast(k, ti.f32) + 0.5) * CELL_SIZE,
        ]
    )


@ti.func
def v_face_world_pos(i, j, k):
    return ti.Vector(
        [
            DOMAIN_MIN_X + (ti.cast(i, ti.f32) + 0.5) * CELL_SIZE,
            DOMAIN_MIN_Y + ti.cast(j, ti.f32) * CELL_SIZE,
            DOMAIN_MIN_Z + (ti.cast(k, ti.f32) + 0.5) * CELL_SIZE,
        ]
    )


@ti.func
def w_face_world_pos(i, j, k):
    return ti.Vector(
        [
            DOMAIN_MIN_X + (ti.cast(i, ti.f32) + 0.5) * CELL_SIZE,
            DOMAIN_MIN_Y + (ti.cast(j, ti.f32) + 0.5) * CELL_SIZE,
            DOMAIN_MIN_Z + ti.cast(k, ti.f32) * CELL_SIZE,
        ]
    )


@ti.func
def trilinear_weight(frac, ox, oy, oz):
    wx = frac[0] if ox == 1 else 1.0 - frac[0]
    wy = frac[1] if oy == 1 else 1.0 - frac[1]
    wz = frac[2] if oz == 1 else 1.0 - frac[2]
    return wx * wy * wz


@ti.func
def trilinear_weight_grad(frac, ox, oy, oz):
    sign_x = 1.0 if ox == 1 else -1.0
    sign_y = 1.0 if oy == 1 else -1.0
    sign_z = 1.0 if oz == 1 else -1.0
    wx = frac[0] if ox == 1 else 1.0 - frac[0]
    wy = frac[1] if oy == 1 else 1.0 - frac[1]
    wz = frac[2] if oz == 1 else 1.0 - frac[2]
    return ti.Vector(
        [
            sign_x * INV_CELL_SIZE * wy * wz,
            sign_y * INV_CELL_SIZE * wx * wz,
            sign_z * INV_CELL_SIZE * wx * wy,
        ]
    )


@ti.func
def cell_is_fluid(i, j, k):
    return cell_type[i, j, k] == FLUID_CELL


@ti.func
def u_face_is_solid(i, j, k):
    return (
        i == 0
        or i == GRID_RES
        or cell_type[i - 1, j, k] == SOLID_CELL
        or cell_type[i, j, k] == SOLID_CELL
    )


@ti.func
def v_face_is_solid(i, j, k):
    return (
        j == 0
        or j == GRID_RES
        or cell_type[i, j - 1, k] == SOLID_CELL
        or cell_type[i, j, k] == SOLID_CELL
    )


@ti.func
def w_face_is_solid(i, j, k):
    return (
        k == 0
        or k == GRID_RES
        or cell_type[i, j, k - 1] == SOLID_CELL
        or cell_type[i, j, k] == SOLID_CELL
    )


@ti.func
def color_ramp(t):
    t = ti.max(0.0, ti.min(1.0, t))
    c0 = ti.Vector([0.16, 0.30, 0.84])
    c1 = ti.Vector([0.15, 0.77, 0.73])
    c2 = ti.Vector([0.98, 0.82, 0.22])
    c3 = ti.Vector([0.94, 0.30, 0.19])
    color = c3
    if t < 0.33:
        local_t = t / 0.33
        color = c0 * (1.0 - local_t) + c1 * local_t
    elif t < 0.66:
        local_t = (t - 0.33) / 0.33
        color = c1 * (1.0 - local_t) + c2 * local_t
    else:
        local_t = (t - 0.66) / 0.34
        color = c2 * (1.0 - local_t) + c3 * local_t
    return color


@ti.kernel
def reset_grid_fields():
    for i, j, k in grid_u:
        grid_u[i, j, k] = 0.0
        grid_u_prev[i, j, k] = 0.0
        grid_u_weight[i, j, k] = 0.0

    for i, j, k in grid_v:
        grid_v[i, j, k] = 0.0
        grid_v_prev[i, j, k] = 0.0
        grid_v_weight[i, j, k] = 0.0

    for i, j, k in grid_w:
        grid_w[i, j, k] = 0.0
        grid_w_prev[i, j, k] = 0.0
        grid_w_weight[i, j, k] = 0.0

    for i, j, k in cell_pressure:
        cell_pressure[i, j, k] = 0.0
        cell_type[i, j, k] = 0
        cell_particle_density[i, j, k] = 0.0
        cell_particle_count[i, j, k] = 0
        cell_component_label[i, j, k] = -1
        cell_component_label_tmp[i, j, k] = -1

    for idx in component_particle_sum:
        component_particle_sum[idx] = 0

    main_component_label[None] = -1
    main_component_particle_sum[None] = 0


@ti.kernel
def clear_particle_affine():
    for p in particle_c:
        particle_c[p] = ti.Matrix.zero(ti.f32, 3, 3)


@ti.kernel
def activate_all_particles():
    for p in particle_active:
        particle_active[p] = 1


@ti.kernel
def reset_visualization_stats():
    max_speed_value[None] = 1e-5
    max_density_value[None] = 1e-5
    max_pressure_value[None] = 1e-5


@ti.kernel
def compute_visualization_stats():
    for p in particle_vel:
        if particle_active[p] != 0:
            ti.atomic_max(max_speed_value[None], particle_vel[p].norm())

    for i, j, k in cell_type:
        if cell_is_fluid(i, j, k):
            ti.atomic_max(max_density_value[None], cell_particle_density[i, j, k])
            ti.atomic_max(max_pressure_value[None], ti.abs(cell_pressure[i, j, k]))


@ti.kernel
def update_particle_colors(color_mode: ti.i32):
    rest_density_value = ti.max(rest_density[None], 1e-5)
    max_speed = ti.max(max_speed_value[None], 1e-5)
    max_density = ti.max(max_density_value[None], rest_density_value)
    max_pressure = ti.max(max_pressure_value[None], 1e-5)

    for p in particle_pos:
        color = ti.Vector([BASE_PARTICLE_COLOR[0], BASE_PARTICLE_COLOR[1], BASE_PARTICLE_COLOR[2]])
        if particle_active[p] != 0:
            pos = particle_pos[p]
            if color_mode == COLOR_MODE_SPEED:
                t = particle_vel[p].norm() / max_speed
                color = color_ramp(t)
            elif color_mode == COLOR_MODE_DENSITY:
                density = sample_centered_scalar_field(cell_particle_density, pos)
                t = density / max_density
                color = color_ramp(t)
            elif color_mode == COLOR_MODE_PRESSURE:
                pressure = sample_centered_scalar_field(cell_pressure, pos)
                t = 0.5 + 0.5 * pressure / max_pressure
                color = color_ramp(t)

        particle_color[p] = color


@ti.kernel
def reset_particle_separation_state():
    for i, j, k in separation_count:
        separation_count[i, j, k] = 0

    for p in particle_separation_delta:
        particle_separation_delta[p] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def build_particle_separation_grid():
    for p in particle_pos:
        if particle_active[p] != 0:
            coord = separation_cell_coord(particle_pos[p])
            i = ti.max(0, ti.min(SEPARATION_GRID_X - 1, ti.cast(ti.floor(coord[0]), ti.i32)))
            j = ti.max(0, ti.min(SEPARATION_GRID_Y - 1, ti.cast(ti.floor(coord[1]), ti.i32)))
            k = ti.max(0, ti.min(SEPARATION_GRID_Z - 1, ti.cast(ti.floor(coord[2]), ti.i32)))
            slot = ti.atomic_add(separation_count[i, j, k], 1)
            if slot < MAX_PARTICLES_PER_SEP_CELL:
                separation_particles[i, j, k, slot] = p


@ti.kernel
def compute_particle_separation():
    min_dist = 2.0 * PARTICLE_RADIUS
    min_dist2 = min_dist * min_dist

    for p in particle_pos:
        if particle_active[p] != 0:
            pos_p = particle_pos[p]
            coord = separation_cell_coord(pos_p)
            ci = ti.max(0, ti.min(SEPARATION_GRID_X - 1, ti.cast(ti.floor(coord[0]), ti.i32)))
            cj = ti.max(0, ti.min(SEPARATION_GRID_Y - 1, ti.cast(ti.floor(coord[1]), ti.i32)))
            ck = ti.max(0, ti.min(SEPARATION_GRID_Z - 1, ti.cast(ti.floor(coord[2]), ti.i32)))

            for ox, oy, oz in ti.static(ti.ndrange(3, 3, 3)):
                ni = ci + ox - 1
                nj = cj + oy - 1
                nk = ck + oz - 1
                if (
                    0 <= ni < SEPARATION_GRID_X
                    and 0 <= nj < SEPARATION_GRID_Y
                    and 0 <= nk < SEPARATION_GRID_Z
                ):
                    count = ti.min(separation_count[ni, nj, nk], MAX_PARTICLES_PER_SEP_CELL)
                    for s in range(count):
                        q = separation_particles[ni, nj, nk, s]
                        if particle_active[q] != 0 and q > p:
                            d = pos_p - particle_pos[q]
                            dist2 = d.dot(d)
                            if 1e-12 < dist2 < min_dist2:
                                dist = ti.sqrt(dist2)
                                corr = 0.5 * (min_dist - dist) * d / dist
                                for c in ti.static(range(3)):
                                    ti.atomic_add(particle_separation_delta[p][c], corr[c])
                                    ti.atomic_add(particle_separation_delta[q][c], -corr[c])


@ti.kernel
def apply_particle_separation():
    for p in particle_pos:
        if particle_active[p] != 0:
            particle_pos[p] += particle_separation_delta[p]


@ti.kernel
def initialize_cell_types():
    for i, j, k in cell_type:
        if (
            i == 0
            or j == 0
            or k == 0
            or i == GRID_RES - 1
            or j == GRID_RES - 1
            or k == GRID_RES - 1
        ):
            cell_type[i, j, k] = SOLID_CELL
        else:
            cell_type[i, j, k] = EMPTY_CELL
        cell_pressure[i, j, k] = 0.0
        cell_particle_count[i, j, k] = 0
        cell_component_label[i, j, k] = -1
        cell_component_label_tmp[i, j, k] = -1


@ti.kernel
def mark_fluid_cells_from_particles():
    for p in particle_pos:
        if particle_active[p] != 0:
            idx = particle_cell_index(particle_pos[p])
            if cell_type[idx[0], idx[1], idx[2]] != SOLID_CELL:
                cell_type[idx[0], idx[1], idx[2]] = FLUID_CELL
                ti.atomic_add(cell_particle_count[idx[0], idx[1], idx[2]], 1)


@ti.kernel
def initialize_component_labels():
    for i, j, k in cell_type:
        label = -1
        if cell_is_fluid(i, j, k):
            label = cell_linear_index(i, j, k)
        cell_component_label[i, j, k] = label
        cell_component_label_tmp[i, j, k] = label


@ti.kernel
def relax_component_labels():
    for i, j, k in cell_type:
        label = cell_component_label[i, j, k]
        next_label = label
        if label >= 0:
            if i > 0 and cell_component_label[i - 1, j, k] >= 0:
                next_label = ti.min(next_label, cell_component_label[i - 1, j, k])
            if i + 1 < GRID_RES and cell_component_label[i + 1, j, k] >= 0:
                next_label = ti.min(next_label, cell_component_label[i + 1, j, k])
            if j > 0 and cell_component_label[i, j - 1, k] >= 0:
                next_label = ti.min(next_label, cell_component_label[i, j - 1, k])
            if j + 1 < GRID_RES and cell_component_label[i, j + 1, k] >= 0:
                next_label = ti.min(next_label, cell_component_label[i, j + 1, k])
            if k > 0 and cell_component_label[i, j, k - 1] >= 0:
                next_label = ti.min(next_label, cell_component_label[i, j, k - 1])
            if k + 1 < GRID_RES and cell_component_label[i, j, k + 1] >= 0:
                next_label = ti.min(next_label, cell_component_label[i, j, k + 1])
        cell_component_label_tmp[i, j, k] = next_label


@ti.kernel
def apply_component_labels():
    for i, j, k in cell_type:
        cell_component_label[i, j, k] = cell_component_label_tmp[i, j, k]


@ti.kernel
def reset_component_particle_sums():
    for idx in component_particle_sum:
        component_particle_sum[idx] = 0
    main_component_label[None] = MAX_CELL_COMPONENTS
    main_component_particle_sum[None] = 0


@ti.kernel
def accumulate_component_particle_sums():
    for i, j, k in cell_type:
        label = cell_component_label[i, j, k]
        if label >= 0:
            ti.atomic_add(component_particle_sum[label], cell_particle_count[i, j, k])


@ti.kernel
def find_main_component_particle_sum():
    for idx in component_particle_sum:
        ti.atomic_max(main_component_particle_sum[None], component_particle_sum[idx])


@ti.kernel
def find_main_component_label():
    for idx in component_particle_sum:
        total = component_particle_sum[idx]
        if total > 0 and total == main_component_particle_sum[None]:
            ti.atomic_min(main_component_label[None], idx)


@ti.kernel
def finalize_main_component_label():
    if main_component_label[None] == MAX_CELL_COMPONENTS:
        main_component_label[None] = -1


@ti.kernel
def cull_isolated_fluid_cells(max_particles: ti.i32):
    main_label = main_component_label[None]
    for i, j, k in cell_type:
        if (
            cell_type[i, j, k] == FLUID_CELL
            and cell_component_label[i, j, k] != main_label
            and cell_particle_count[i, j, k] <= max_particles
        ):
            cell_type[i, j, k] = EMPTY_CELL


@ti.kernel
def deactivate_particles_in_empty_cells():
    hidden_pos = ti.Vector(
        [HIDDEN_PARTICLE_POS[0], HIDDEN_PARTICLE_POS[1], HIDDEN_PARTICLE_POS[2]]
    )
    zero_vel = ti.Vector([0.0, 0.0, 0.0])
    zero_affine = ti.Matrix.zero(ti.f32, 3, 3)
    for p in particle_pos:
        if particle_active[p] != 0:
            idx = particle_cell_index(particle_pos[p])
            if cell_type[idx[0], idx[1], idx[2]] == EMPTY_CELL:
                particle_active[p] = 0
                particle_pos[p] = hidden_pos
                particle_vel[p] = zero_vel
                particle_c[p] = zero_affine


@ti.kernel
def scatter_particles_to_grid():
    for p in particle_pos:
        if particle_active[p] != 0:
            pos = particle_pos[p]
            vel = particle_vel[p]

            base_u = ti.cast(ti.floor(u_face_coord(pos)), ti.i32)
            frac_u = u_face_coord(pos) - ti.cast(base_u, ti.f32)
            for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
                idx = base_u + ti.Vector([ox, oy, oz])
                if (
                    0 <= idx[0] < GRID_RES + 1
                    and 0 <= idx[1] < GRID_RES
                    and 0 <= idx[2] < GRID_RES
                ):
                    wx = frac_u[0] if ox == 1 else 1.0 - frac_u[0]
                    wy = frac_u[1] if oy == 1 else 1.0 - frac_u[1]
                    wz = frac_u[2] if oz == 1 else 1.0 - frac_u[2]
                    weight = wx * wy * wz
                    ti.atomic_add(grid_u[idx[0], idx[1], idx[2]], weight * vel[0])
                    ti.atomic_add(grid_u_weight[idx[0], idx[1], idx[2]], weight)

            base_v = ti.cast(ti.floor(v_face_coord(pos)), ti.i32)
            frac_v = v_face_coord(pos) - ti.cast(base_v, ti.f32)
            for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
                idx = base_v + ti.Vector([ox, oy, oz])
                if (
                    0 <= idx[0] < GRID_RES
                    and 0 <= idx[1] < GRID_RES + 1
                    and 0 <= idx[2] < GRID_RES
                ):
                    wx = frac_v[0] if ox == 1 else 1.0 - frac_v[0]
                    wy = frac_v[1] if oy == 1 else 1.0 - frac_v[1]
                    wz = frac_v[2] if oz == 1 else 1.0 - frac_v[2]
                    weight = wx * wy * wz
                    ti.atomic_add(grid_v[idx[0], idx[1], idx[2]], weight * vel[1])
                    ti.atomic_add(grid_v_weight[idx[0], idx[1], idx[2]], weight)

            base_w = ti.cast(ti.floor(w_face_coord(pos)), ti.i32)
            frac_w = w_face_coord(pos) - ti.cast(base_w, ti.f32)
            for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
                idx = base_w + ti.Vector([ox, oy, oz])
                if (
                    0 <= idx[0] < GRID_RES
                    and 0 <= idx[1] < GRID_RES
                    and 0 <= idx[2] < GRID_RES + 1
                ):
                    wx = frac_w[0] if ox == 1 else 1.0 - frac_w[0]
                    wy = frac_w[1] if oy == 1 else 1.0 - frac_w[1]
                    wz = frac_w[2] if oz == 1 else 1.0 - frac_w[2]
                    weight = wx * wy * wz
                    ti.atomic_add(grid_w[idx[0], idx[1], idx[2]], weight * vel[2])
                    ti.atomic_add(grid_w_weight[idx[0], idx[1], idx[2]], weight)


@ti.kernel
def scatter_particles_to_grid_apic():
    for p in particle_pos:
        if particle_active[p] != 0:
            pos = particle_pos[p]
            vel = particle_vel[p]
            affine = particle_c[p]

            coord_u = u_face_coord(pos)
            base_u = ti.cast(ti.floor(coord_u), ti.i32)
            frac_u = coord_u - ti.cast(base_u, ti.f32)
            for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
                idx = base_u + ti.Vector([ox, oy, oz])
                if (
                    0 <= idx[0] < GRID_RES + 1
                    and 0 <= idx[1] < GRID_RES
                    and 0 <= idx[2] < GRID_RES
                ):
                    weight = trilinear_weight(frac_u, ox, oy, oz)
                    offset = u_face_world_pos(idx[0], idx[1], idx[2]) - pos
                    value = vel[0] + affine[0, 0] * offset[0] + affine[0, 1] * offset[1] + affine[0, 2] * offset[2]
                    ti.atomic_add(grid_u[idx[0], idx[1], idx[2]], weight * value)
                    ti.atomic_add(grid_u_weight[idx[0], idx[1], idx[2]], weight)

            coord_v = v_face_coord(pos)
            base_v = ti.cast(ti.floor(coord_v), ti.i32)
            frac_v = coord_v - ti.cast(base_v, ti.f32)
            for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
                idx = base_v + ti.Vector([ox, oy, oz])
                if (
                    0 <= idx[0] < GRID_RES
                    and 0 <= idx[1] < GRID_RES + 1
                    and 0 <= idx[2] < GRID_RES
                ):
                    weight = trilinear_weight(frac_v, ox, oy, oz)
                    offset = v_face_world_pos(idx[0], idx[1], idx[2]) - pos
                    value = vel[1] + affine[1, 0] * offset[0] + affine[1, 1] * offset[1] + affine[1, 2] * offset[2]
                    ti.atomic_add(grid_v[idx[0], idx[1], idx[2]], weight * value)
                    ti.atomic_add(grid_v_weight[idx[0], idx[1], idx[2]], weight)

            coord_w = w_face_coord(pos)
            base_w = ti.cast(ti.floor(coord_w), ti.i32)
            frac_w = coord_w - ti.cast(base_w, ti.f32)
            for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
                idx = base_w + ti.Vector([ox, oy, oz])
                if (
                    0 <= idx[0] < GRID_RES
                    and 0 <= idx[1] < GRID_RES
                    and 0 <= idx[2] < GRID_RES + 1
                ):
                    weight = trilinear_weight(frac_w, ox, oy, oz)
                    offset = w_face_world_pos(idx[0], idx[1], idx[2]) - pos
                    value = vel[2] + affine[2, 0] * offset[0] + affine[2, 1] * offset[1] + affine[2, 2] * offset[2]
                    ti.atomic_add(grid_w[idx[0], idx[1], idx[2]], weight * value)
                    ti.atomic_add(grid_w_weight[idx[0], idx[1], idx[2]], weight)


@ti.kernel
def normalize_grid_velocities():
    for i, j, k in grid_u:
        weight = grid_u_weight[i, j, k]
        if weight > 0.0:
            grid_u[i, j, k] /= weight
        else:
            grid_u[i, j, k] = 0.0

    for i, j, k in grid_v:
        weight = grid_v_weight[i, j, k]
        if weight > 0.0:
            grid_v[i, j, k] /= weight
        else:
            grid_v[i, j, k] = 0.0

    for i, j, k in grid_w:
        weight = grid_w_weight[i, j, k]
        if weight > 0.0:
            grid_w[i, j, k] /= weight
        else:
            grid_w[i, j, k] = 0.0


@ti.kernel
def initialize_velocity_extrapolation():
    for i, j, k in grid_u:
        valid = 0
        if grid_u_weight[i, j, k] > 0.0 and not u_face_is_solid(i, j, k):
            valid = 1
        grid_u_valid[i, j, k] = valid
        grid_u_valid_tmp[i, j, k] = valid
        grid_u_tmp[i, j, k] = grid_u[i, j, k]

    for i, j, k in grid_v:
        valid = 0
        if grid_v_weight[i, j, k] > 0.0 and not v_face_is_solid(i, j, k):
            valid = 1
        grid_v_valid[i, j, k] = valid
        grid_v_valid_tmp[i, j, k] = valid
        grid_v_tmp[i, j, k] = grid_v[i, j, k]

    for i, j, k in grid_w:
        valid = 0
        if grid_w_weight[i, j, k] > 0.0 and not w_face_is_solid(i, j, k):
            valid = 1
        grid_w_valid[i, j, k] = valid
        grid_w_valid_tmp[i, j, k] = valid
        grid_w_tmp[i, j, k] = grid_w[i, j, k]


@ti.kernel
def extrapolate_velocity_pass():
    for i, j, k in grid_u:
        grid_u_tmp[i, j, k] = grid_u[i, j, k]
        grid_u_valid_tmp[i, j, k] = grid_u_valid[i, j, k]
        if grid_u_valid[i, j, k] == 0 and not u_face_is_solid(i, j, k):
            total = 0.0
            count = 0
            if i > 0 and grid_u_valid[i - 1, j, k] == 1:
                total += grid_u[i - 1, j, k]
                count += 1
            if i + 1 < GRID_RES + 1 and grid_u_valid[i + 1, j, k] == 1:
                total += grid_u[i + 1, j, k]
                count += 1
            if j > 0 and grid_u_valid[i, j - 1, k] == 1:
                total += grid_u[i, j - 1, k]
                count += 1
            if j + 1 < GRID_RES and grid_u_valid[i, j + 1, k] == 1:
                total += grid_u[i, j + 1, k]
                count += 1
            if k > 0 and grid_u_valid[i, j, k - 1] == 1:
                total += grid_u[i, j, k - 1]
                count += 1
            if k + 1 < GRID_RES and grid_u_valid[i, j, k + 1] == 1:
                total += grid_u[i, j, k + 1]
                count += 1
            if count > 0:
                grid_u_tmp[i, j, k] = total / ti.cast(count, ti.f32)
                grid_u_valid_tmp[i, j, k] = 1

    for i, j, k in grid_v:
        grid_v_tmp[i, j, k] = grid_v[i, j, k]
        grid_v_valid_tmp[i, j, k] = grid_v_valid[i, j, k]
        if grid_v_valid[i, j, k] == 0 and not v_face_is_solid(i, j, k):
            total = 0.0
            count = 0
            if i > 0 and grid_v_valid[i - 1, j, k] == 1:
                total += grid_v[i - 1, j, k]
                count += 1
            if i + 1 < GRID_RES and grid_v_valid[i + 1, j, k] == 1:
                total += grid_v[i + 1, j, k]
                count += 1
            if j > 0 and grid_v_valid[i, j - 1, k] == 1:
                total += grid_v[i, j - 1, k]
                count += 1
            if j + 1 < GRID_RES + 1 and grid_v_valid[i, j + 1, k] == 1:
                total += grid_v[i, j + 1, k]
                count += 1
            if k > 0 and grid_v_valid[i, j, k - 1] == 1:
                total += grid_v[i, j, k - 1]
                count += 1
            if k + 1 < GRID_RES and grid_v_valid[i, j, k + 1] == 1:
                total += grid_v[i, j, k + 1]
                count += 1
            if count > 0:
                grid_v_tmp[i, j, k] = total / ti.cast(count, ti.f32)
                grid_v_valid_tmp[i, j, k] = 1

    for i, j, k in grid_w:
        grid_w_tmp[i, j, k] = grid_w[i, j, k]
        grid_w_valid_tmp[i, j, k] = grid_w_valid[i, j, k]
        if grid_w_valid[i, j, k] == 0 and not w_face_is_solid(i, j, k):
            total = 0.0
            count = 0
            if i > 0 and grid_w_valid[i - 1, j, k] == 1:
                total += grid_w[i - 1, j, k]
                count += 1
            if i + 1 < GRID_RES and grid_w_valid[i + 1, j, k] == 1:
                total += grid_w[i + 1, j, k]
                count += 1
            if j > 0 and grid_w_valid[i, j - 1, k] == 1:
                total += grid_w[i, j - 1, k]
                count += 1
            if j + 1 < GRID_RES and grid_w_valid[i, j + 1, k] == 1:
                total += grid_w[i, j + 1, k]
                count += 1
            if k > 0 and grid_w_valid[i, j, k - 1] == 1:
                total += grid_w[i, j, k - 1]
                count += 1
            if k + 1 < GRID_RES + 1 and grid_w_valid[i, j, k + 1] == 1:
                total += grid_w[i, j, k + 1]
                count += 1
            if count > 0:
                grid_w_tmp[i, j, k] = total / ti.cast(count, ti.f32)
                grid_w_valid_tmp[i, j, k] = 1


@ti.kernel
def apply_extrapolated_velocities():
    for i, j, k in grid_u:
        grid_u[i, j, k] = grid_u_tmp[i, j, k]
        grid_u_valid[i, j, k] = grid_u_valid_tmp[i, j, k]

    for i, j, k in grid_v:
        grid_v[i, j, k] = grid_v_tmp[i, j, k]
        grid_v_valid[i, j, k] = grid_v_valid_tmp[i, j, k]

    for i, j, k in grid_w:
        grid_w[i, j, k] = grid_w_tmp[i, j, k]
        grid_w_valid[i, j, k] = grid_w_valid_tmp[i, j, k]


@ti.kernel
def apply_solid_velocity_constraints():
    for i, j, k in grid_u:
        left_solid = i == 0 or cell_type[i - 1, j, k] == SOLID_CELL
        right_solid = i == GRID_RES or cell_type[i, j, k] == SOLID_CELL
        if left_solid or right_solid:
            grid_u[i, j, k] = 0.0

    for i, j, k in grid_v:
        down_solid = j == 0 or cell_type[i, j - 1, k] == SOLID_CELL
        up_solid = j == GRID_RES or cell_type[i, j, k] == SOLID_CELL
        if down_solid or up_solid:
            grid_v[i, j, k] = 0.0

    for i, j, k in grid_w:
        back_solid = k == 0 or cell_type[i, j, k - 1] == SOLID_CELL
        front_solid = k == GRID_RES or cell_type[i, j, k] == SOLID_CELL
        if back_solid or front_solid:
            grid_w[i, j, k] = 0.0


@ti.kernel
def copy_grid_to_previous():
    for i, j, k in grid_u:
        grid_u_prev[i, j, k] = grid_u[i, j, k]

    for i, j, k in grid_v:
        grid_v_prev[i, j, k] = grid_v[i, j, k]

    for i, j, k in grid_w:
        grid_w_prev[i, j, k] = grid_w[i, j, k]


@ti.kernel
def grid_to_particles(flip_ratio: ti.f32):
    for p in particle_pos:
        if particle_active[p] != 0:
            pos = particle_pos[p]
            old_vel = particle_vel[p]

            pic_velocity = ti.Vector(
                [
                    sample_u_field(grid_u, pos),
                    sample_v_field(grid_v, pos),
                    sample_w_field(grid_w, pos),
                ]
            )
            flip_delta = ti.Vector(
                [
                    sample_u_field(grid_u, pos) - sample_u_field(grid_u_prev, pos),
                    sample_v_field(grid_v, pos) - sample_v_field(grid_v_prev, pos),
                    sample_w_field(grid_w, pos) - sample_w_field(grid_w_prev, pos),
                ]
            )
            flip_velocity = old_vel + flip_delta
            particle_vel[p] = (1.0 - flip_ratio) * pic_velocity + flip_ratio * flip_velocity


@ti.kernel
def grid_to_particles_apic():
    for p in particle_pos:
        if particle_active[p] != 0:
            pos = particle_pos[p]

            pic_u = 0.0
            grad_u = ti.Vector([0.0, 0.0, 0.0])
            coord_u = u_face_coord(pos)
            base_u = ti.cast(ti.floor(coord_u), ti.i32)
            frac_u = coord_u - ti.cast(base_u, ti.f32)
            for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
                idx = base_u + ti.Vector([ox, oy, oz])
                if (
                    0 <= idx[0] < GRID_RES + 1
                    and 0 <= idx[1] < GRID_RES
                    and 0 <= idx[2] < GRID_RES
                ):
                    weight = trilinear_weight(frac_u, ox, oy, oz)
                    grad = trilinear_weight_grad(frac_u, ox, oy, oz)
                    value = grid_u[idx[0], idx[1], idx[2]]
                    pic_u += weight * value
                    grad_u += value * grad

            pic_v = 0.0
            grad_v = ti.Vector([0.0, 0.0, 0.0])
            coord_v = v_face_coord(pos)
            base_v = ti.cast(ti.floor(coord_v), ti.i32)
            frac_v = coord_v - ti.cast(base_v, ti.f32)
            for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
                idx = base_v + ti.Vector([ox, oy, oz])
                if (
                    0 <= idx[0] < GRID_RES
                    and 0 <= idx[1] < GRID_RES + 1
                    and 0 <= idx[2] < GRID_RES
                ):
                    weight = trilinear_weight(frac_v, ox, oy, oz)
                    grad = trilinear_weight_grad(frac_v, ox, oy, oz)
                    value = grid_v[idx[0], idx[1], idx[2]]
                    pic_v += weight * value
                    grad_v += value * grad

            pic_w = 0.0
            grad_w = ti.Vector([0.0, 0.0, 0.0])
            coord_w = w_face_coord(pos)
            base_w = ti.cast(ti.floor(coord_w), ti.i32)
            frac_w = coord_w - ti.cast(base_w, ti.f32)
            for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
                idx = base_w + ti.Vector([ox, oy, oz])
                if (
                    0 <= idx[0] < GRID_RES
                    and 0 <= idx[1] < GRID_RES
                    and 0 <= idx[2] < GRID_RES + 1
                ):
                    weight = trilinear_weight(frac_w, ox, oy, oz)
                    grad = trilinear_weight_grad(frac_w, ox, oy, oz)
                    value = grid_w[idx[0], idx[1], idx[2]]
                    pic_w += weight * value
                    grad_w += value * grad

            particle_vel[p] = ti.Vector([pic_u, pic_v, pic_w])
            particle_c[p] = ti.Matrix(
                [
                    [grad_u[0], grad_u[1], grad_u[2]],
                    [grad_v[0], grad_v[1], grad_v[2]],
                    [grad_w[0], grad_w[1], grad_w[2]],
                ]
            )


@ti.kernel
def clear_density_grid():
    for i, j, k in cell_particle_density:
        cell_particle_density[i, j, k] = 0.0


@ti.kernel
def scatter_particle_density():
    for p in particle_pos:
        if particle_active[p] != 0:
            pos = particle_pos[p]
            coord = centered_cell_coord(pos)
            base = ti.cast(ti.floor(coord), ti.i32)
            frac = coord - ti.cast(base, ti.f32)

            for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
                idx = base + ti.Vector([ox, oy, oz])
                if (
                    0 <= idx[0] < GRID_RES
                    and 0 <= idx[1] < GRID_RES
                    and 0 <= idx[2] < GRID_RES
                ):
                    wx = frac[0] if ox == 1 else 1.0 - frac[0]
                    wy = frac[1] if oy == 1 else 1.0 - frac[1]
                    wz = frac[2] if oz == 1 else 1.0 - frac[2]
                    ti.atomic_add(cell_particle_density[idx[0], idx[1], idx[2]], wx * wy * wz)


@ti.kernel
def initialize_rest_density():
    total_density = 0.0
    total_cells = 0
    for i, j, k in cell_particle_density:
        if cell_is_fluid(i, j, k):
            total_density += cell_particle_density[i, j, k]
            total_cells += 1

    if total_cells > 0:
        rest_density[None] = total_density / ti.cast(total_cells, ti.f32)


@ti.kernel
def pressure_projection_iteration(dt: ti.f32, over_relaxation: ti.f32, compensate_drift: ti.i32):
    ti.loop_config(serialize=True)
    for i, j, k in ti.ndrange(GRID_RES, GRID_RES, GRID_RES):
        if cell_is_fluid(i, j, k):
            sx0 = 0.0 if i == 0 or cell_type[i - 1, j, k] == SOLID_CELL else 1.0
            sx1 = 0.0 if i == GRID_RES - 1 or cell_type[i + 1, j, k] == SOLID_CELL else 1.0
            sy0 = 0.0 if j == 0 or cell_type[i, j - 1, k] == SOLID_CELL else 1.0
            sy1 = 0.0 if j == GRID_RES - 1 or cell_type[i, j + 1, k] == SOLID_CELL else 1.0
            sz0 = 0.0 if k == 0 or cell_type[i, j, k - 1] == SOLID_CELL else 1.0
            sz1 = 0.0 if k == GRID_RES - 1 or cell_type[i, j, k + 1] == SOLID_CELL else 1.0

            denom = sx0 + sx1 + sy0 + sy1 + sz0 + sz1
            if denom > 0.0:
                divergence = (
                    grid_u[i + 1, j, k]
                    - grid_u[i, j, k]
                    + grid_v[i, j + 1, k]
                    - grid_v[i, j, k]
                    + grid_w[i, j, k + 1]
                    - grid_w[i, j, k]
                )
                if compensate_drift != 0 and rest_density[None] > 0.0:
                    density_error = cell_particle_density[i, j, k] - rest_density[None]
                    if density_error > 0.0:
                        divergence -= DRIFT_COMPENSATION_GAIN * density_error
                rhs = FLUID_DENSITY * CELL_SIZE / dt * divergence
                neighbor_sum = (
                    sx0 * cell_pressure[i - 1, j, k]
                    + sx1 * cell_pressure[i + 1, j, k]
                    + sy0 * cell_pressure[i, j - 1, k]
                    + sy1 * cell_pressure[i, j + 1, k]
                    + sz0 * cell_pressure[i, j, k - 1]
                    + sz1 * cell_pressure[i, j, k + 1]
                )
                pressure_target = (neighbor_sum - rhs) / denom
                cell_pressure[i, j, k] += over_relaxation * (
                    pressure_target - cell_pressure[i, j, k]
                )


@ti.kernel
def apply_pressure_gradient(dt: ti.f32):
    scale = dt / (FLUID_DENSITY * CELL_SIZE)

    for i, j, k in grid_u:
        if i == 0 or i == GRID_RES:
            grid_u[i, j, k] = 0.0
        else:
            left_type = cell_type[i - 1, j, k]
            right_type = cell_type[i, j, k]
            if left_type == SOLID_CELL or right_type == SOLID_CELL:
                grid_u[i, j, k] = 0.0
            elif left_type == EMPTY_CELL and right_type == EMPTY_CELL:
                grid_u[i, j, k] = 0.0
            else:
                p_left = cell_pressure[i - 1, j, k] if left_type == FLUID_CELL else 0.0
                p_right = cell_pressure[i, j, k] if right_type == FLUID_CELL else 0.0
                grid_u[i, j, k] -= scale * (p_right - p_left)

    for i, j, k in grid_v:
        if j == 0 or j == GRID_RES:
            grid_v[i, j, k] = 0.0
        else:
            down_type = cell_type[i, j - 1, k]
            up_type = cell_type[i, j, k]
            if down_type == SOLID_CELL or up_type == SOLID_CELL:
                grid_v[i, j, k] = 0.0
            elif down_type == EMPTY_CELL and up_type == EMPTY_CELL:
                grid_v[i, j, k] = 0.0
            else:
                p_down = cell_pressure[i, j - 1, k] if down_type == FLUID_CELL else 0.0
                p_up = cell_pressure[i, j, k] if up_type == FLUID_CELL else 0.0
                grid_v[i, j, k] -= scale * (p_up - p_down)

    for i, j, k in grid_w:
        if k == 0 or k == GRID_RES:
            grid_w[i, j, k] = 0.0
        else:
            back_type = cell_type[i, j, k - 1]
            front_type = cell_type[i, j, k]
            if back_type == SOLID_CELL or front_type == SOLID_CELL:
                grid_w[i, j, k] = 0.0
            elif back_type == EMPTY_CELL and front_type == EMPTY_CELL:
                grid_w[i, j, k] = 0.0
            else:
                p_back = cell_pressure[i, j, k - 1] if back_type == FLUID_CELL else 0.0
                p_front = cell_pressure[i, j, k] if front_type == FLUID_CELL else 0.0
                grid_w[i, j, k] -= scale * (p_front - p_back)


def initialize_scene():
    particle_pos.from_numpy(INITIAL_PARTICLE_POS)
    particle_vel.from_numpy(INITIAL_PARTICLE_VEL)
    particle_color.from_numpy(INITIAL_PARTICLE_COLOR)
    activate_all_particles()
    tank_line_verts.from_numpy(TANK_LINE_VERTICES)
    reset_grid_fields()
    clear_particle_affine()
    reset_visualization_stats()
    rest_density[None] = 0.0


@ti.kernel
def integrate_particles(dt: ti.f32):
    for p in particle_pos:
        if particle_active[p] != 0:
            particle_vel[p] += GRAVITY * dt
            particle_pos[p] += particle_vel[p] * dt


def push_particles_apart(num_iters: int):
    for _ in range(num_iters):
        reset_particle_separation_state()
        build_particle_separation_grid()
        compute_particle_separation()
        apply_particle_separation()
        handle_particle_collisions()


@ti.kernel
def handle_particle_collisions():
    for p in particle_pos:
        if particle_active[p] != 0:
            pos = particle_pos[p]
            vel = particle_vel[p]
            collided = 0

            if pos[0] < BOUNDARY_MIN_X:
                pos[0] = BOUNDARY_MIN_X
                collided = 1
                if vel[0] < 0.0:
                    vel[0] = 0.0
            elif pos[0] > BOUNDARY_MAX_X:
                pos[0] = BOUNDARY_MAX_X
                collided = 1
                if vel[0] > 0.0:
                    vel[0] = 0.0

            if pos[1] < BOUNDARY_MIN_Y:
                pos[1] = BOUNDARY_MIN_Y
                collided = 1
                if vel[1] < 0.0:
                    vel[1] = 0.0
            elif pos[1] > BOUNDARY_MAX_Y:
                pos[1] = BOUNDARY_MAX_Y
                collided = 1
                if vel[1] > 0.0:
                    vel[1] = 0.0

            if pos[2] < BOUNDARY_MIN_Z:
                pos[2] = BOUNDARY_MIN_Z
                collided = 1
                if vel[2] < 0.0:
                    vel[2] = 0.0
            elif pos[2] > BOUNDARY_MAX_Z:
                pos[2] = BOUNDARY_MAX_Z
                collided = 1
                if vel[2] > 0.0:
                    vel[2] = 0.0

            particle_pos[p] = pos
            particle_vel[p] = vel
            if collided != 0:
                particle_c[p] = ti.Matrix.zero(ti.f32, 3, 3)


def cull_isolated_spray_particles():
    initialize_component_labels()
    for _ in range(COMPONENT_LABEL_RELAX_ITERS):
        relax_component_labels()
        apply_component_labels()
    reset_component_particle_sums()
    accumulate_component_particle_sums()
    find_main_component_particle_sum()
    find_main_component_label()
    finalize_main_component_label()
    cull_isolated_fluid_cells(CHEAP_CULL_MAX_PARTICLES)
    deactivate_particles_in_empty_cells()


def extrapolate_velocity_fields():
    initialize_velocity_extrapolation()
    for _ in range(VELOCITY_EXTRAPOLATION_ITERS):
        extrapolate_velocity_pass()
        apply_extrapolated_velocities()


def transfer_velocities(to_grid: bool, flip_ratio: float, transfer_mode: int):
    if to_grid:
        reset_grid_fields()
        initialize_cell_types()
        mark_fluid_cells_from_particles()
        cull_isolated_spray_particles()
        if transfer_mode == TRANSFER_MODE_APIC:
            scatter_particles_to_grid_apic()
        else:
            scatter_particles_to_grid()
        normalize_grid_velocities()
        apply_solid_velocity_constraints()
        extrapolate_velocity_fields()
        apply_solid_velocity_constraints()
        copy_grid_to_previous()
    else:
        if transfer_mode == TRANSFER_MODE_APIC:
            grid_to_particles_apic()
        else:
            grid_to_particles(flip_ratio)


def update_particle_density():
    clear_density_grid()
    scatter_particle_density()
    if rest_density[None] <= 0.0:
        initialize_rest_density()


def solve_incompressibility(
    num_iters: int,
    dt: float,
    over_relaxation: float,
    compensate_drift: bool,
):
    for _ in range(num_iters):
        pressure_projection_iteration(
            dt,
            over_relaxation,
            1 if compensate_drift else 0,
        )
    apply_pressure_gradient(dt)
    apply_solid_velocity_constraints()
    extrapolate_velocity_fields()
    apply_solid_velocity_constraints()


def simulate_frame(
    frame_dt: float,
    num_substeps: int,
    flip_ratio: float,
    transfer_mode: int,
    num_particle_iters: int,
    num_pressure_iters: int,
    over_relaxation: float,
    separate_particles: bool,
    compensate_drift: bool,
):
    sub_dt = frame_dt / max(1, num_substeps)
    for _ in range(num_substeps):
        integrate_particles(sub_dt)
        handle_particle_collisions()
        if separate_particles:
            push_particles_apart(num_particle_iters)
        handle_particle_collisions()
        transfer_velocities(True, flip_ratio, transfer_mode)
        update_particle_density()
        solve_incompressibility(
            num_pressure_iters,
            sub_dt,
            over_relaxation,
            compensate_drift,
        )
        transfer_velocities(False, flip_ratio, transfer_mode)


def refresh_particle_visualization(color_mode: int):
    reset_visualization_stats()
    compute_visualization_stats()
    update_particle_colors(color_mode)


camera_pos = np.array([1.30, 0.82, 1.42], dtype=np.float32)
camera_target = np.array([0.0, -0.08, 0.0], dtype=np.float32)


def rotate_camera_from_mouse(dx: float, dy: float):
    global camera_pos

    offset = camera_pos - camera_target
    radius = max(np.linalg.norm(offset), 1e-6)
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


def transfer_mode_name(transfer_mode: int, flip_ratio: float) -> str:
    if transfer_mode == TRANSFER_MODE_APIC:
        return "APIC"
    if flip_ratio <= 0.01:
        return "PIC"
    if flip_ratio >= 0.99:
        return "FLIP"
    return f"PIC/FLIP ({flip_ratio:.2f})"


def next_transfer_mode(transfer_mode: int) -> int:
    return (transfer_mode + 1) % 2


def color_mode_name(color_mode: int) -> str:
    if color_mode == COLOR_MODE_SPEED:
        return "speed"
    if color_mode == COLOR_MODE_DENSITY:
        return "density"
    if color_mode == COLOR_MODE_PRESSURE:
        return "pressure"
    return "base"


def next_color_mode(color_mode: int) -> int:
    return (color_mode + 1) % 4


def print_controls():
    print("=" * 64)
    print("Taichi Lab 2 - APIC bonus B4")
    print("=" * 64)
    print("Controls:")
    print("  Space    : pause / resume")
    print("  R        : reset particles")
    print("  G        : toggle tank wireframe")
    print("  M        : switch PIC/FLIP blend and APIC, then reset")
    print("  Tab      : cycle base / speed / density / pressure colors")
    print("  RMB drag : orbit camera")
    print("=" * 64)


def draw_status_panel(
    gui,
    paused: bool,
    frame_dt: float,
    flip_ratio: float,
    transfer_mode: int,
    color_mode: int,
):
    gui.begin("Lab 2 B4 Controls", 0.02, 0.02, 0.28, 0.22)
    reset_requested = gui.button("Reset")
    paused = gui.checkbox("Paused", paused)
    frame_dt = gui.slider_float("frame dt", frame_dt, 1.0 / 240.0, 1.0 / 20.0)
    flip_ratio = gui.slider_float("flipRatio", flip_ratio, 0.0, 1.0)
    gui.text(f"transfer attribute: {transfer_mode_name(transfer_mode, flip_ratio)}")
    gui.text(f"color attribute: {color_mode_name(color_mode)}")
    gui.end()
    return (
        paused,
        frame_dt,
        flip_ratio,
        reset_requested,
    )


def main():
    global camera_pos, camera_target

    initialize_scene()
    print_controls()

    frame_dt = DEFAULT_FRAME_DT
    num_substeps = DEFAULT_SUBSTEPS
    num_particle_iters = 1
    num_pressure_iters = DEFAULT_PRESSURE_ITERS
    over_relaxation = DEFAULT_OVER_RELAXATION
    flip_ratio = DEFAULT_FLIP_RATIO
    transfer_mode = TRANSFER_MODE_BLEND
    color_mode = DEFAULT_COLOR_MODE
    separate_particles = True
    compensate_drift = True
    paused = False
    show_tank = True

    window = ti.ui.Window("Taichi FLIP / APIC Fluid - Bonus B4", res=WINDOW_RES, vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    gui = window.get_gui()

    camera_pos = np.array([1.30, 0.82, 1.42], dtype=np.float32)
    camera_target = np.array([0.0, -0.08, 0.0], dtype=np.float32)
    refresh_particle_visualization(color_mode)
    rotating = False
    last_mouse_x = 0.0
    last_mouse_y = 0.0

    while window.running:
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.ESCAPE:
                window.running = False
            elif event.key == ti.ui.SPACE:
                paused = not paused
            elif event.key == "r":
                initialize_scene()
                refresh_particle_visualization(color_mode)
            elif event.key == "g":
                show_tank = not show_tank
            elif event.key == "m":
                transfer_mode = next_transfer_mode(transfer_mode)
                initialize_scene()
                refresh_particle_visualization(color_mode)
            elif event.key == ti.ui.TAB:
                color_mode = next_color_mode(color_mode)

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
            simulate_frame(
                frame_dt,
                num_substeps,
                flip_ratio,
                transfer_mode,
                num_particle_iters,
                num_pressure_iters,
                over_relaxation,
                separate_particles,
                compensate_drift,
            )

        refresh_particle_visualization(color_mode)

        camera.position(*camera_pos)
        camera.lookat(*camera_target)
        camera.up(0.0, 1.0, 0.0)
        scene.set_camera(camera)
        scene.ambient_light((0.72, 0.72, 0.72))
        scene.point_light((1.8, 2.2, 1.6), (1.15, 1.15, 1.15))
        canvas.set_background_color((0.84, 0.86, 0.90))

        if show_tank:
            scene.lines(tank_line_verts, color=(0.20, 0.22, 0.24), width=1.4)

        scene.particles(
            particle_pos,
            radius=PARTICLE_DRAW_RADIUS,
            per_vertex_color=particle_color,
        )

        canvas.scene(scene)
        (
            paused,
            frame_dt,
            flip_ratio,
            reset_requested,
        ) = draw_status_panel(
            gui,
            paused,
            frame_dt,
            flip_ratio,
            transfer_mode,
            color_mode,
        )
        if reset_requested:
            initialize_scene()
            refresh_particle_visualization(color_mode)
        window.show()


if __name__ == "__main__":
    main()
