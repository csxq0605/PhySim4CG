"""
正弦摆动球体阵列
- 多个球在平面上按正弦规律摆动
- 形成波浪效果
"""

import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# 参数
grid_x = 15  # x方向球数
grid_z = 15  # z方向球数
n_balls = grid_x * grid_z
ball_radius = 0.08

# 生成单个球的模板（低多边形球）
def create_ball_mesh(radius, segments=8):
    """创建球体网格"""
    vertices = []
    indices = []

    # 生成顶点（不包括极点）
    for j in range(1, segments):
        theta = np.pi * j / segments
        for i in range(segments):
            phi = 2 * np.pi * i / segments
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.cos(theta)
            z = radius * np.sin(theta) * np.sin(phi)
            vertices.append([x, y, z])

    # 添加两极
    vertices.append([0, radius, 0])   # 北极
    vertices.append([0, -radius, 0])  # 南极

    n_ring = segments * (segments - 1)
    north_pole = n_ring
    south_pole = n_ring + 1

    # 生成三角形索引
    # 北极三角形
    for i in range(segments):
        next_i = (i + 1) % segments
        indices.extend([north_pole, i, next_i])

    # 中间区域
    for j in range(segments - 2):
        for i in range(segments):
            next_i = (i + 1) % segments
            curr_row = j * segments + i
            next_row = (j + 1) * segments + i
            curr_row_next = j * segments + next_i
            next_row_next = (j + 1) * segments + next_i
            indices.extend([curr_row, next_row, curr_row_next])
            indices.extend([curr_row_next, next_row, next_row_next])

    # 南极三角形
    last_row_start = (segments - 2) * segments
    for i in range(segments):
        next_i = (i + 1) % segments
        indices.extend([last_row_start + i, south_pole, last_row_start + next_i])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)

# 创建球体模板
ball_verts, ball_indices = create_ball_mesh(ball_radius, segments=6)
n_verts_per_ball = len(ball_verts)

# 为每个球生成颜色（基于位置）
def get_ball_color(i, j):
    """根据网格位置返回颜色"""
    t = (i + j) / (grid_x + grid_z)
    return [
        0.3 + 0.7 * np.sin(t * np.pi),
        0.5 + 0.5 * np.cos(t * np.pi * 0.5),
        0.8 - 0.4 * np.sin(t * np.pi * 0.3)
    ]

# 创建所有球的顶点和索引
all_vertices = []
all_indices = []
all_colors = []

for i in range(grid_x):
    for j in range(grid_z):
        ball_id = i * grid_z + j
        # 基础位置（平面上）
        base_x = (i - grid_x / 2) * ball_radius * 3
        base_z = (j - grid_z / 2) * ball_radius * 3

        # 为每个球添加顶点
        for v in ball_verts:
            all_vertices.append([
                v[0] + base_x,
                v[1],  # y由正弦函数控制，这里先设为0
                v[2] + base_z
            ])

        # 添加索引（偏移）
        offset = ball_id * n_verts_per_ball
        for idx in ball_indices:
            all_indices.append(idx + offset)

        # 添加颜色
        color = get_ball_color(i, j)
        for _ in range(n_verts_per_ball):
            all_colors.append(color)

all_vertices = np.array(all_vertices, dtype=np.float32)
all_indices = np.array(all_indices, dtype=np.int32)
all_colors = np.array(all_colors, dtype=np.float32)

# 创建Taichi场
N_VERTS = len(all_vertices)
N_INDICES = len(all_indices)

vertex_pos = ti.Vector.field(3, dtype=ti.f32, shape=N_VERTS)
index_buf = ti.field(dtype=ti.i32, shape=N_INDICES)
vertex_color = ti.Vector.field(3, dtype=ti.f32, shape=N_VERTS)

index_buf.from_numpy(all_indices)
vertex_color.from_numpy(all_colors)

# 存储基础位置用于动画
base_positions = np.array(all_vertices, dtype=np.float32)


@ti.kernel
def update_positions(
    verts: ti.template(),
    base: ti.types.ndarray(),
    t: ti.f32,
    freq: ti.f32,
    amp: ti.f32
):
    """更新球体位置（正弦波动）"""
    for i in range(N_VERTS):
        # 计算这是第几个球
        ball_id = i // n_verts_per_ball
        ball_i = ball_id // grid_z
        ball_j = ball_id % grid_z

        # 基础位置
        bx = base[i, 0]
        bz = base[i, 2]

        # 正弦波动（基于位置和时间的相位）
        phase = (ball_i * 0.3 + ball_j * 0.3)
        wave = ti.sin(t * freq + phase) * amp

        verts[i] = [bx, base[i, 1] + wave, bz]


def main():
    window = ti.ui.Window("Sine Wave Balls (Taichi GGUI)", (1000, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    # 相机参数
    camera_distance = 4.0
    camera_azimuth = 0.5
    camera_elevation = 0.4

    last_mouse_x = 0.0
    last_mouse_y = 0.0
    is_dragging = False

    # 动画参数
    time = 0.0
    frequency = 2.0  # 波动频率
    amplitude = 0.15  # 波动幅度
    paused = False

    print("=" * 50)
    print("Sine Wave Balls - 正弦摆动球体阵列")
    print("=" * 50)
    print(f"球体数量: {grid_x} x {grid_z} = {n_balls}")
    print("控制:")
    print("  鼠标左键拖拽: 旋转视角")
    print("  空格: 暂停/继续")
    print("=" * 50)

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == ti.ui.SPACE:
                paused = not paused

        # 鼠标控制
        if window.is_pressed(ti.ui.LMB):
            mouse_x, mouse_y = window.get_cursor_pos()
            if not is_dragging:
                is_dragging = True
                last_mouse_x = mouse_x
                last_mouse_y = mouse_y
            else:
                dx = mouse_x - last_mouse_x
                dy = mouse_y - last_mouse_y
                camera_azimuth -= dx * 2.0
                camera_elevation += dy * 2.0
                camera_elevation = max(-np.pi/2 + 0.1, min(np.pi/2 - 0.1, camera_elevation))
                last_mouse_x = mouse_x
                last_mouse_y = mouse_y
        else:
            is_dragging = False

        # 更新时间
        if not paused:
            time += 0.016  # 约60fps

        # 更新球体位置
        update_positions(vertex_pos, base_positions, time, frequency, amplitude)

        # 设置相机
        cam_x = camera_distance * np.cos(camera_elevation) * np.sin(camera_azimuth)
        cam_y = camera_distance * np.sin(camera_elevation)
        cam_z = camera_distance * np.cos(camera_elevation) * np.cos(camera_azimuth)

        camera.position(cam_x, cam_y, cam_z)
        camera.lookat(0, 0, 0)
        camera.up(0, 1, 0)
        camera.fov(60)

        scene.set_camera(camera)
        scene.ambient_light((0.6, 0.6, 0.6))
        scene.point_light(pos=(3, 5, 3), color=(1, 1, 1))
        scene.point_light(pos=(-3, -2, -3), color=(0.4, 0.4, 0.6))

        # 绘制球体
        scene.mesh(vertex_pos, indices=index_buf, per_vertex_color=vertex_color, two_sided=True)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
