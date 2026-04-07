"""
带面渲染的3D盒子 - 使用Taichi GGUI高效渲染
支持鼠标拖拽旋转
"""

import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# 盒子8个顶点
vertices = np.array([
    [-0.5,  0.5,  0.5],  # 0
    [ 0.5,  0.5,  0.5],  # 1
    [ 0.5,  0.5, -0.5],  # 2
    [-0.5,  0.5, -0.5],  # 3
    [-0.5, -0.5,  0.5],  # 4
    [ 0.5, -0.5,  0.5],  # 5
    [ 0.5, -0.5, -0.5],  # 6
    [-0.5, -0.5, -0.5],  # 7
], dtype=np.float32)

# 6个面，每个面2个三角形
indices = np.array([
    0, 1, 2,  0, 2, 3,  # 顶面
    4, 6, 5,  4, 7, 6,  # 底面
    0, 4, 5,  0, 5, 1,  # 前面
    2, 6, 7,  2, 7, 3,  # 后面
    0, 3, 7,  0, 7, 4,  # 左面
    1, 5, 6,  1, 6, 2,  # 右面
], dtype=np.int32)

# 每个三角面的颜色
colors = np.array([
    [0.47, 0.81, 0.67], [0.47, 0.81, 0.67],  # 顶面
    [0.35, 0.61, 0.50], [0.35, 0.61, 0.50],  # 底面
    [0.53, 0.87, 0.71], [0.53, 0.87, 0.71],  # 前面
    [0.40, 0.70, 0.58], [0.40, 0.70, 0.58],  # 后面
    [0.45, 0.75, 0.62], [0.45, 0.75, 0.62],  # 左面
    [0.50, 0.80, 0.65], [0.50, 0.80, 0.65],  # 右面
], dtype=np.float32)

# 边线索引（预定义）
edges = np.array([0,1, 1,2, 2,3, 3,0, 4,5, 5,6, 6,7, 7,4, 0,4, 1,5, 2,6, 3,7], dtype=np.int32)

# 创建Taichi场
N_VERTS = 8
N_FACES = 12
N_EDGE_VERTS = 24  # 12条边 * 2个顶点

vertex_pos = ti.Vector.field(3, dtype=ti.f32, shape=N_VERTS)
index_buf = ti.field(dtype=ti.i32, shape=N_FACES * 3)
vertex_color = ti.Vector.field(3, dtype=ti.f32, shape=N_FACES * 3)
edge_idx_buf = ti.field(dtype=ti.i32, shape=N_EDGE_VERTS)
line_verts = ti.Vector.field(3, dtype=ti.f32, shape=N_EDGE_VERTS)

vertex_pos.from_numpy(vertices)
index_buf.from_numpy(indices)
vertex_color.from_numpy(colors.repeat(3, axis=0))
edge_idx_buf.from_numpy(edges)

# 预计算边线顶点（只执行一次）
@ti.kernel
def init_line_verts():
    for i in range(N_EDGE_VERTS):
        line_verts[i] = vertex_pos[edge_idx_buf[i]]


def main():
    window = ti.ui.Window("Box Solid (Taichi GGUI)", (800, 600))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    # 球坐标相机参数
    camera_distance = 3.0
    camera_azimuth = 0.5
    camera_elevation = 0.3

    # 鼠标状态
    last_mouse_x = 0.0
    last_mouse_y = 0.0
    is_dragging = False

    # 初始化边线顶点
    init_line_verts()

    print("=" * 40)
    print("Box Solid - Taichi GGUI渲染")
    print("=" * 40)
    print("鼠标左键拖拽: 旋转视角")
    print("=" * 40)

    while window.running:
        # 处理事件
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False

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

        # 根据球坐标计算相机位置
        cam_x = camera_distance * np.cos(camera_elevation) * np.sin(camera_azimuth)
        cam_y = camera_distance * np.sin(camera_elevation)
        cam_z = camera_distance * np.cos(camera_elevation) * np.cos(camera_azimuth)

        camera.position(cam_x, cam_y, cam_z)
        camera.lookat(0, 0, 0)
        camera.up(0, 1, 0)
        camera.fov(60)

        # 设置场景
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))

        # 绘制盒子
        scene.mesh(vertex_pos, indices=index_buf, per_vertex_color=vertex_color, two_sided=True)

        # 绘制边线（直接使用预计算的line_verts）
        scene.lines(line_verts, color=(1, 1, 1), width=2)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
