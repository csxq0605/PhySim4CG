# Lab 2 FLIP 作业报告

姓名：`苏王捷`

学号：`2300011075`

仓库提交记录：见 [lab2_log.txt](./lab2_log.txt)

## 1. 作业概述

本次作业使用 Python + Taichi 实现了一个基于粒子-网格混合表示的 3D 液体求解器，并完成了 2 个 Bonus 任务：

- 基础 Demo：`taichi_lab2_flip.py`
- Bonus B1：`taichi_lab2_bonus_b1.py`
- Bonus B4：`taichi_lab2_bonus_b4_apic.py`

三份代码共用的核心求解流程如下：

```python
for _ in range(num_substeps):
    integrate_particles(sub_dt)
    handle_particle_collisions()
    if separate_particles:
        push_particles_apart(num_particle_iters)
    handle_particle_collisions()
    transfer_velocities(True, flip_ratio[, transfer_mode])
    update_particle_density()
    solve_incompressibility(num_pressure_iters, sub_dt, over_relaxation, compensate_drift)
    transfer_velocities(False, flip_ratio[, transfer_mode])
```

整体实现基于 MAC 网格保存速度分量，粒子保存位置和速度；压力投影阶段在网格上用 Gauss-Seidel 消除散度，最后再把网格速度回传到粒子。

为兼顾课程要求和实时运行速度，我在三个文件中统一采用了以下数值策略：

- 网格分辨率固定为 `24^3`，与课程文档建议范围一致。
- 使用线性插值完成 `P2G / G2P`。
- 通过 `flipRatio` 在 PIC 与 FLIP 之间切换，基础版默认使用 `FLIP95`。
- 补充了 `grid velocity extrapolation`，避免自由表面附近的无效空气速度在回采样时把粒子速度拉成 0。
- 补充了 cheap culling：对不属于主液团、且粒子数很少的孤立 fluid cell 直接清空，以抑制长时间悬空的小液滴。

## 2. 基础 Demo：`taichi_lab2_flip.py`

### 2.1 核心实现思路

基础部分对应课程要求中的 3 个核心功能：时间步长交互、PIC/FLIP 混合、不可压投影。

主要实现点如下：

- 粒子状态使用 `particle_pos`、`particle_vel` 表示，网格速度使用 MAC 交错网格上的 `grid_u / grid_v / grid_w` 表示。
- `transfer_velocities(True, flip_ratio)` 负责粒子向网格传输速度，并保存上一时刻网格速度，用于 FLIP 增量更新。
- `grid_to_particles(flip_ratio)` 中同时计算 PIC 速度和 FLIP 增量速度，再按 `flipRatio` 做线性混合。
- `updateParticleDensity + solveIncompressibility` 在网格上估计密度并求解压力，压力投影后再更新网格速度。
- `push_particles_apart` 使用空间哈希做近邻搜索，解决粒子重叠，减轻数值爆炸。
- 为了减轻自由表面速度场缺失导致的阻尼现象补了 velocity extrapolation；为减轻空中散落小液滴补了 cheap culling。

从效果上看：

- `flipRatio = 0` 时更接近 PIC，数值更稳，但耗散更明显。
- `flipRatio = 1` 时更接近 FLIP，细节更多，但更容易出现噪声和小喷溅。
- `flipRatio = 0.95` 在稳定性和保留运动细节之间更均衡。

### 2.2 交互方式

- `Space`：暂停 / 继续
- `R`：重置粒子
- `G`：显示 / 隐藏 tank 线框
- `鼠标右键拖动`：绕 tank 中心旋转相机
- GUI `frame dt`：调节每帧时间步长
- GUI `flipRatio`：在 PIC / FLIP 之间连续切换

### 2.3 Demo 展示

![complex](.\demo\base.gif)

---

## 3. Bonus B1：可视化与交互增强 `taichi_lab2_bonus_b1.py`

### 3.1 任务目标

B1 的目标是增强液体的可视化表达，并加入可交互障碍物，帮助更直观地观察速度、密度和压力分布，以及液体与运动边界之间的耦合效果。

### 3.2 核心实现思路

本文件在基础版之上新增了两部分功能。

第一部分是粒子着色可视化：

- 支持 `base / speed / density / pressure` 四种颜色模式。
- `compute_visualization_stats()` 先统计全局最大速度、密度、压力，用于颜色归一化。
- `update_particle_colors()` 再根据当前模式将粒子映射到颜色渐变带。
- 这样可以在同一个求解器里直接观察不同物理量，而不用额外导出数据。

第二部分是球形障碍物交互：

- 使用 `obstacle_pos / obstacle_vel` 保存障碍物中心和速度。
- 先在网格中标记 obstacle cells，再在速度约束阶段将这些面速度设置为障碍物速度。
- 粒子碰撞阶段对进入球体内部的粒子做位置投影，并移除法向向内速度。
- 鼠标左键拖拽时，通过“相机射线与拖拽平面求交”的方式得到新的球心位置，再由相邻两帧位置差估计障碍物速度。

在这个版本里，基础版的 velocity extrapolation 和 cheap culling 仍然保留，因此障碍物交互时的自由表面会更平滑，空中悬浮的小液滴也会减少。

### 3.3 交互方式

- `Space`：暂停 / 继续
- `R`：重置粒子与障碍物
- `G`：显示 / 隐藏 tank 线框
- `Tab`：在 `base / speed / density / pressure` 四种着色模式之间循环
- `鼠标左键拖动`：在屏幕平面内拖动球形障碍物
- `鼠标右键拖动`：绕 tank 中心旋转相机
- GUI `frame dt`：调节时间步长
- GUI `flipRatio`：调节 PIC / FLIP 混合系数

### 3.4 结果说明

从这个任务中可以观察到：

- `speed` 模式下，障碍物附近被挤压和甩动的粒子会显示更亮的高速颜色。
- `density` 模式下，局部堆积区域会更明显，适合观察自由表面和局部压缩现象。
- `pressure` 模式下，障碍物推动液体时，其前方会形成更高压力区域。

### 3.5 Demo 展示

![complex](.\demo\b1.gif)

---

## 4. Bonus B4：APIC 对比 `taichi_lab2_bonus_b4_apic.py`

### 4.1 任务目标

B4 的目标是在同一套 FLIP 框架上实现 APIC，并与 PIC / FLIP 进行对比。

### 4.2 核心实现思路

本文件保留了基础版的 PIC / FLIP 路径，同时新增了 APIC 所需的粒子 affine 信息：

- 每个粒子额外维护一个 `particle_c`，表示局部仿射速度场。
- `scatter_particles_to_grid_apic()` 在 `P2G` 时不仅传输粒子平动速度，还把 affine 项一起映射到各个 MAC face。
- `grid_to_particles_apic()` 在 `G2P` 时从网格回收粒子速度，并根据三线性权重梯度更新新的 `particle_c`。
- 为了避免边界碰撞后遗留错误的 affine 信息，粒子与边界碰撞或被 cheap culling 删除时，都会同步清零 `particle_c`。

交互层面上：

- `M` 用于在 `PIC/FLIP blend` 和 `APIC` 之间切换，默认开始为 `APIC` 模式，切换时会立刻重置场景（注意到切换过程较慢会有明显卡顿）。
- `Tab` 用于在 `base / speed / density / pressure` 四种着色模式之间切换。

本文件同样保留了 velocity extrapolation 和 cheap culling，因此三种传输模式都在同样的自由表面修正条件下比较，避免因为边界处理不一致导致比较失真。

### 4.3 交互方式

- `Space`：暂停 / 继续
- `R`：重置粒子
- `G`：显示 / 隐藏 tank 线框
- `M`：在 `PIC/FLIP blend` 与 `APIC` 之间切换，并自动重置
- `Tab`：切换颜色显示属性
- `RMB drag`：绕 tank 中心旋转相机
- GUI `frame dt`：调节时间步长
- GUI `flipRatio`：在 `transfer_mode = PIC/FLIP blend` 时调节混合系数

### 4.4 对 Bonus 问题的回答

1. PIC、FLIP、APIC 的主要差别是什么？

- PIC 直接从网格插值回粒子，数值更稳定，但耗散最强。
- FLIP 回传的是网格速度增量，能保留更多运动细节，但更容易把网格噪声带回粒子。
- APIC 在粒子上额外保存局部仿射速度场，因此比 PIC 更能保留旋转和剪切信息，同时通常比纯 FLIP 更平滑。

2. 在本实现中三者的表现差异如何？

- PIC 的液面最“粘”，小尺度涡和飞溅更容易被抹平。
- FLIP95 能保留更多翻涌和喷溅，但自由表面也更活跃，更容易出现离散小液滴。
- APIC 在保留局部旋转和细节方面比 PIC 更好，同时相较纯 FLIP 更不容易出现过强噪声，因此整体观感更均衡。

### 4.5 Demo 展示

![complex](.\demo\b4.gif)

---

## 5. 总结

本次 Lab 2 最终完成了：

- 基础 FLIP 液体求解器
- `flipRatio` 控制下的 PIC / FLIP 连续切换
- 基于 Gauss-Seidel 的不可压投影
- B1：速度 / 密度 / 压力可视化与可交互球形障碍物
- B4：APIC 传输与 PIC / FLIP / APIC 对比