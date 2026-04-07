"""
四面体网格(Tetrahedral Mesh)可视化 - 共享顶点版本
- 8个共享顶点定义长方体
- 6个四面体面片索引
- 显示边线（线框）
- 正弦摆动动画
"""

import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# 网格参数
grid_x, grid_y, grid_z = 10, 6, 6  # 网格数量

# 6个四面体的顶点索引（相对于8个角点的局部索引）
tet_vertex_indices = [
    [0, 1, 3, 4],
    [1, 3, 4, 5],
    [2, 3, 1, 6],
    [3, 4, 5, 7],
    [3, 5, 6, 7],
    [1, 3, 5, 6]
]

# 四面体的6条边（局部索引对）
tet_edges_local = [
    [0, 1], [0, 2], [0, 3],
    [1, 2], [1, 3], [2, 3]
]


def create_mesh():
    """创建四面体网格（共享顶点）"""

    # 总顶点数
    n_vertices = (grid_x + 1) * (grid_y + 1) * (grid_z + 1)

    vertices = []
    colors = []
    face_indices = []  # 面片索引
    edge_indices = []  # 边线索引

    # 生成立方体网格顶点（共享）
    dx, dy, dz = 1.0 / grid_x, 1.0 / grid_y, 1.0 / grid_z
    offset_x = -0.5
    offset_y = -0.3
    offset_z = -0.3

    # 创建顶点映射：网格坐标 -> 全局顶点索引
    def get_vert_idx(i, j, k):
        return i * (grid_y + 1) * (grid_z + 1) + j * (grid_z + 1) + k

    # 生成所有顶点
    for i in range(grid_x + 1):
        for j in range(grid_y + 1):
            for k in range(grid_z + 1):
                x = offset_x + i * dx
                y = offset_y + j * dy
                z = offset_z + k * dz
                vertices.append([x, y, z])

                # 颜色基于位置
                t = (i + j + k) / (grid_x + grid_y + grid_z)
                color = [
                    0.3 + 0.7 * np.sin(t * np.pi),
                    0.5 + 0.5 * np.cos(t * np.pi * 0.7),
                    0.8
                ]
                colors.append(color)

    # 生成四面体面片和边线索引
    for i in range(grid_x):
        for j in range(grid_y):
            for k in range(grid_z):
                # 当前小长方体的8个角点的全局索引
                v = [
                    get_vert_idx(i, j, k),       # 0
                    get_vert_idx(i+1, j, k),     # 1
                    get_vert_idx(i+1, j+1, k),   # 2
                    get_vert_idx(i, j+1, k),     # 3
                    get_vert_idx(i, j, k+1),     # 4
                    get_vert_idx(i+1, j, k+1),   # 5
                    get_vert_idx(i+1, j+1, k+1), # 6
                    get_vert_idx(i, j+1, k+1),   # 7
                ]

                # 6个四面体
                for tet_local in tet_vertex_indices:
                    # 获取这个四面体的4个全局顶点索引
                    tet_global = [v[idx] for idx in tet_local]

                    # 4个三角形面
                    faces = [
                        [tet_global[0], tet_global[1], tet_global[2]],
                        [tet_global[0], tet_global[1], tet_global[3]],
                        [tet_global[0], tet_global[2], tet_global[3]],
                        [tet_global[1], tet_global[2], tet_global[3]]
                    ]
                    for face in faces:
                        face_indices.extend(face)

                    # 6条边
                    for edge in tet_edges_local:
                        edge_indices.append(tet_global[edge[0]])
                        edge_indices.append(tet_global[edge[1]])

    # 确保边线索引数是偶数（每条边需要2个顶点）
    if len(edge_indices) % 2 != 0:
        edge_indices.append(edge_indices[-1])  # 复制最后一个使之为偶数

    return (np.array(vertices, dtype=np.float32),
            np.array(face_indices, dtype=np.int32),
            np.array(edge_indices, dtype=np.int32),
            np.array(colors, dtype=np.float32))


vertices, face_indices, edge_indices, colors = create_mesh()
N_VERTS = len(vertices)
N_FACE_INDICES = len(face_indices)
N_EDGE_INDICES = len(edge_indices)

print(f"顶点数: {N_VERTS}")
print(f"面片索引数: {N_FACE_INDICES}")
print(f"边线索引数: {N_EDGE_INDICES}")
print(f"边数: {N_EDGE_INDICES // 2}")

# Taichi场
vertex_pos = ti.Vector.field(3, dtype=ti.f32, shape=N_VERTS)
face_buf = ti.field(dtype=ti.i32, shape=N_FACE_INDICES)
edge_buf = ti.field(dtype=ti.i32, shape=N_EDGE_INDICES)
vertex_color = ti.Vector.field(3, dtype=ti.f32, shape=N_VERTS)

# 边线顶点field（预分配，避免每帧创建）
line_verts = ti.Vector.field(3, dtype=ti.f32, shape=N_EDGE_INDICES)

vertex_pos.from_numpy(vertices)
face_buf.from_numpy(face_indices)
edge_buf.from_numpy(edge_indices)
vertex_color.from_numpy(colors)

# 基础位置
base_positions = np.array(vertices, dtype=np.float32)


@ti.kernel
def get_line_vertices(
    verts: ti.template(),
    edge_idx: ti.template(),
    line_verts: ti.template()
):
    """将边线索引转换为顶点位置"""
    for i in range(edge_idx.shape[0]):
        v_idx = edge_idx[i]
        line_verts[i] = verts[v_idx]


@ti.kernel
def update_wave(
    verts: ti.template(),
    base: ti.types.ndarray(),
    t: ti.f32
):
    """更新顶点位置 - 正弦波动"""
    for i in range(N_VERTS):
        # 计算属于哪个网格
        tmp = i
        x = tmp // ((grid_y + 1) * (grid_z + 1))
        tmp = tmp % ((grid_y + 1) * (grid_z + 1))
        y = tmp // (grid_z + 1)
        z = tmp % (grid_z + 1)

        bx = base[i, 0]
        by = base[i, 1]
        bz = base[i, 2]

        # 正弦波动
        phase = (x * 0.5 + y * 0.3 + z * 0.4)
        wave = ti.sin(t * 2.0 + phase) * 0.1

        verts[i] = [bx, by + wave, bz]


def main():
    window = ti.ui.Window("Tetrahedral Mesh", (1000, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    camera_distance = 2.5
    camera_azimuth = 0.5
    camera_elevation = 0.2

    last_mouse_x = 0.0
    last_mouse_y = 0.0
    is_dragging = False

    time = 0.0
    paused = False
    show_edges = True  # 是否显示边线
    show_faces = True  # 是否显示面片

    print("=" * 50)
    print("Tetrahedral Mesh - 四面体网格")
    print("=" * 50)
    print(f"网格: {grid_x}x{grid_y}x{grid_z}")
    print(f"顶点数: {N_VERTS} (共享)")
    print("控制:")
    print("  鼠标左键拖拽: 旋转视角")
    print("  空格: 暂停/继续")
    print("  F: 显示/隐藏面片")
    print("  E: 显示/隐藏边线")
    print("=" * 50)

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == ti.ui.SPACE:
                paused = not paused
            elif e.key == 'f':
                show_faces = not show_faces
                print(f"显示面片: {show_faces}")
            elif e.key == 'e':
                show_edges = not show_edges
                print(f"显示边线: {show_edges}")

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

        if not paused:
            time += 0.016

        # 更新波动
        update_wave(vertex_pos, base_positions, time)

        # 如果显示边线，同步更新边线顶点位置
        if show_edges:
            get_line_vertices(vertex_pos, edge_buf, line_verts)

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

        # 绘制面片
        if show_faces:
            scene.mesh(vertex_pos, indices=face_buf, per_vertex_color=vertex_color,
                      two_sided=True)

        # 绘制边线（白色线框）- line_verts已在上面更新
        if show_edges:
            scene.lines(line_verts, color=(1.0, 1.0, 1.0), width=1.0)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
