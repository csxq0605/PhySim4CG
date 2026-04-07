"""
2D 基础形状绘制示例
展示 Taichi GGUI 2D Canvas 的所有基础绘制功能

坐标系统说明:
- 使用归一化坐标 (0~1)
- (0, 0) = 左下角, (1, 1) = 右上角
- 所有形状保持正方形比例（不随窗口拉伸）
"""

import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# 窗口尺寸
width, height = 1000, 800

# 统一的显示区域（保持正方形，居中）
DISPLAY_SCALE = 0.7  # 占窗口的比例
DISPLAY_OFFSET_X = 0.5 - DISPLAY_SCALE / 2  # 居中偏移
DISPLAY_OFFSET_Y = 0.5 - DISPLAY_SCALE / 2


def to_screen(x, y):
    """将局部坐标(0~1)转换为屏幕坐标"""
    return DISPLAY_OFFSET_X + x * DISPLAY_SCALE, DISPLAY_OFFSET_Y + y * DISPLAY_SCALE


@ti.kernel
def init_circles(pos: ti.template(), colors: ti.template(), n: ti.i32):
    """初始化圆点位置"""
    for i in range(n):
        angle = i * 6.28318 / n
        radius = 0.2 + 0.1 * ti.sin(i * 0.5)
        # 在局部坐标系中计算，然后转换到屏幕坐标
        lx = 0.5 + radius * ti.cos(angle)
        ly = 0.5 + radius * ti.sin(angle)
        pos[i] = [DISPLAY_OFFSET_X + lx * DISPLAY_SCALE,
                  DISPLAY_OFFSET_Y + ly * DISPLAY_SCALE]
        t = i / n
        colors[i] = [0.5 + 0.5 * ti.sin(t * 6.28),
                     0.5 + 0.5 * ti.cos(t * 6.28), 0.8]


@ti.kernel
def update_circles(pos: ti.template(), n: ti.i32, time: ti.f32):
    """动画更新圆点位置"""
    for i in range(n):
        angle = i * 6.28318 / n + time
        radius = 0.2 + 0.05 * ti.sin(time * 2 + i * 0.3)
        lx = 0.5 + radius * ti.cos(angle)
        ly = 0.5 + radius * ti.sin(angle)
        pos[i] = [DISPLAY_OFFSET_X + lx * DISPLAY_SCALE,
                  DISPLAY_OFFSET_Y + ly * DISPLAY_SCALE]


def main():
    window = ti.ui.Window("2D Primitives - 基础形状示例", (width, height))
    canvas = window.get_canvas()

    # 1. 圆点 (circles) - 在中心区域
    n_circles = 30
    circle_pos = ti.Vector.field(2, dtype=ti.f32, shape=n_circles)
    circle_colors = ti.Vector.field(3, dtype=ti.f32, shape=n_circles)
    init_circles(circle_pos, circle_colors, n_circles)

    # 2. 线段 (lines) - 左上角星形
    line_verts = ti.Vector.field(2, dtype=ti.f32, shape=20)
    for i in range(10):
        angle1 = i * 6.28318 / 10
        angle2 = (i + 1) * 6.28318 / 10
        r1 = 0.12 if i % 2 == 0 else 0.06
        r2 = 0.12 if (i + 1) % 2 == 0 else 0.06
        # 左上角区域 (局部坐标 0~0.4)
        cx, cy = 0.25, 0.75
        x1 = cx + r1 * np.cos(angle1)
        y1 = cy + r1 * np.sin(angle1)
        x2 = cx + r2 * np.cos(angle2)
        y2 = cy + r2 * np.sin(angle2)
        line_verts[i * 2] = list(to_screen(x1 * 0.5, y1 * 0.5 + 0.5))
        line_verts[i * 2 + 1] = list(to_screen(x2 * 0.5, y2 * 0.5 + 0.5))

    # 3. 三角形 (triangles) - 右上角彩色扇形
    tri_verts = ti.Vector.field(2, dtype=ti.f32, shape=12)
    tri_indices = ti.field(dtype=ti.i32, shape=12)
    tri_colors = ti.Vector.field(3, dtype=ti.f32, shape=12)
    cx, cy = 0.75, 0.75
    for i in range(6):
        angle = i * 6.28318 / 6
        next_angle = (i + 1) * 6.28318 / 6
        r = 0.15
        # 中心点
        tri_verts[i * 3] = list(to_screen(cx, cy))
        # 周围两点
        tri_verts[i * 3 + 1] = list(to_screen(cx + r * np.cos(angle) * 0.5,
                                              cy + r * np.sin(angle) * 0.5))
        tri_verts[i * 3 + 2] = list(to_screen(cx + r * np.cos(next_angle) * 0.5,
                                              cy + r * np.sin(next_angle) * 0.5))
        tri_indices[i * 3] = i * 3
        tri_indices[i * 3 + 1] = i * 3 + 1
        tri_indices[i * 3 + 2] = i * 3 + 2
        # 彩虹色
        tri_colors[i * 3] = [1.0, 0.0, 0.0]
        tri_colors[i * 3 + 1] = [0.0, 1.0, 0.0]
        tri_colors[i * 3 + 2] = [0.0, 0.0, 1.0]

    # 4. 网格点 - 左下角
    grid_verts = ti.Vector.field(2, dtype=ti.f32, shape=100)
    for i in range(10):
        for j in range(10):
            # 左下角区域 (0~0.4)
            gx = 0.05 + i * 0.035
            gy = 0.05 + j * 0.035
            grid_verts[i * 10 + j] = list(to_screen(gx, gy))

    # 5. 边框矩形 - 右下角
    border_verts = ti.Vector.field(2, dtype=ti.f32, shape=8)
    bx, by = 0.75, 0.25
    size = 0.15
    # 四条边
    border_verts[0] = list(to_screen(bx - size, by - size))
    border_verts[1] = list(to_screen(bx + size, by - size))
    border_verts[2] = list(to_screen(bx + size, by - size))
    border_verts[3] = list(to_screen(bx + size, by + size))
    border_verts[4] = list(to_screen(bx + size, by + size))
    border_verts[5] = list(to_screen(bx - size, by + size))
    border_verts[6] = list(to_screen(bx - size, by + size))
    border_verts[7] = list(to_screen(bx - size, by - size))

    # 6. 坐标轴
    axis_verts = ti.Vector.field(2, dtype=ti.f32, shape=4)
    # X轴
    axis_verts[0] = list(to_screen(0, 0.5))
    axis_verts[1] = list(to_screen(1, 0.5))
    # Y轴
    axis_verts[2] = list(to_screen(0.5, 0))
    axis_verts[3] = list(to_screen(0.5, 1))

    # 动画时间
    time = 0.0
    paused = False

    print("=" * 50)
    print("2D Primitives - 基础形状绘制示例")
    print("=" * 50)
    print("坐标系统:")
    print("  - 归一化坐标 (0~1)")
    print("  - (0,0)=左下角, (1,1)=右上角")
    print("  - 保持正方形比例（不随窗口拉伸）")
    print("=" * 50)
    print("展示内容:")
    print("  1. 彩色圆点阵列 (circles) - 中心")
    print("  2. 星形线段 (lines) - 左上")
    print("  3. 彩色扇形 (triangles) - 右上")
    print("  4. 网格点 (circles) - 左下")
    print("  5. 矩形边框 (lines) - 右下")
    print("  6. 坐标轴 (lines) - 中心十字")
    print("=" * 50)
    print("控制:")
    print("  空格: 暂停/继续动画")
    print("=" * 50)

    while window.running:
        # 事件处理
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == ti.ui.SPACE:
                paused = not paused

        if not paused:
            time += 0.016
            update_circles(circle_pos, n_circles, time)

        # 绘制坐标轴
        canvas.lines(axis_verts, width=1, color=(0.3, 0.3, 0.3))

        # 绘制 1: 圆点 (带动画)
        canvas.circles(circle_pos, radius=0.012, per_vertex_color=circle_colors)

        # 绘制 2: 星形线段
        canvas.lines(line_verts, width=2, color=(1.0, 0.8, 0.2))

        # 绘制 3: 彩色三角形扇形
        canvas.triangles(tri_verts, indices=tri_indices,
                         per_vertex_color=tri_colors)

        # 绘制 4: 网格点 (白色小圆点)
        canvas.circles(grid_verts, radius=0.004, color=(0.9, 0.9, 0.9))

        # 绘制 5: 边框矩形
        canvas.lines(border_verts, width=2, color=(0.5, 0.8, 0.5))

        window.show()


if __name__ == "__main__":
    main()
