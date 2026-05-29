# Go2W Height Scan 与 Stepit 对齐排查记录

本文记录当前对 IsaacLab `height_scan` 的观察、对 Stepit/MuJoCo 侧高程图语义的疑问，以及后续排查清单。目标是确认 sim2sim 时输入给策略的 187 维高程图在语义、符号、单位、维度和排列顺序上是否一致。

## 1. IsaacLab 侧已确认的公式

IsaacLab 的 `mdp.height_scan` 原始公式是：

```text
height_scan = sensor_z - hit_z - offset
offset 默认值 = 0.5
```

对应代码：

```python
return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
```

当前 Go2W play debug 中使用的是：

```text
height_scan = sensor_z - hit_z - 0.5
```

其中：

- `sensor_z` 来自 `height_scanner` 的 `sensor.data.pos_w[:, 2]`
- `hit_z` 来自 `sensor.data.ray_hits_w[..., 2]`
- `hit_z` 基本可以认为是世界系命中点绝对高度
- 输出单位是米
- 输出维度是 `187 = 17 x 11`

## 2. IsaacLab 平地观测

平地 debug 样例：

```text
sensor_z ~= 0.3514
hit_z ~= 0
height_scan ~= 0.3514 - 0 - 0.5 = -0.1486
```

观察：

- 平地上的 187 个 `hit_z` 基本都在 `0` 附近，std 在 `1e-6` 量级。
- 平地上的 `height_scan` 基本是常数 `-0.1485` 左右。
- 这说明当前雷达参考高度约为 `0.351 m`，但公式里减的是固定基准 `0.5 m`。
- 因此平地不是 `0`，而是约 `-0.149`。

这点很关键：如果 Stepit/MuJoCo 侧平地高程图被做成零均值或接近 `0`，就已经和 IsaacLab 训练输入不一致。

## 3. IsaacLab 楼梯前观测

准备上楼梯前 debug 样例：

```text
sensor_z ~= 0.363 ~ 0.367
hit_z min ~= 0
hit_z max ~= 0.3948
height_scan max ~= -0.133
height_scan min ~= -0.532
```

根据公式：

```text
地面 ray: height_scan ~= 0.36 - 0.0 - 0.5 = -0.14
台阶 ray: height_scan ~= 0.36 - 0.395 - 0.5 = -0.535
```

观察：

- 在 IsaacLab 语义里，前方更高的台阶会让 `height_scan` 变得更负。
- 也就是说，高障碍不是更正，而是更负。
- 这是 Stepit/MuJoCo 侧最需要对齐的符号约定。

可以把 IsaacLab 目标语义写成：

```text
height_scan[i] = radar_ref_z_world - terrain_height_world[i] - 0.5
```

不要把它理解成普通的激光距离图。

## 4. 仍需确认的 IsaacLab 疑问

1. `sensor_z` 是否严格等于 radar link 的世界系 z？

当前 Go2W 配置里 `height_scanner.prim_path` 绑定到：

```text
{ENV_REGEX_NS}/Robot/base/radar
```

play debug 中 `sensor_z ~= 0.35`，看起来像 radar link 或其参考帧高度。但还需要直接打印 radar link world pose 做一次确认。

2. `RayCasterCfg.offset=(0, 0, 20)` 和 `sensor_z ~= 0.35` 的关系是什么？

基础环境里 `height_scanner` 配置有：

```python
offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0))
```

但 debug 里的 `sensor_z` 不是 20m 量级。因此需要进一步确认：

- `sensor.data.pos_w` 是否包含这个 offset
- 实际 ray 起点 `ray_starts_w[..., 2]` 是不是在高处
- `sensor_z` 和 `ray_starts_w` 是否是两个不同概念

建议在 IsaacLab play debug 中额外打印：

```text
sensor.data.pos_w[:, 2]
sensor._ray_starts_w[..., 2] 的 min/max/mean
sensor.ray_starts[..., 2] 的 min/max/mean
height_scanner prim 的 world pose z
```

3. `raw_height_scan` 和 `height_scan slice` 有轻微差异的原因是什么？

目前差异很小，通常在 `1e-5` 到 `1e-4` 以内，但仍建议确认：

- 是否存在 observation manager 的时序差异
- 是否存在 clip/scale/noise 的残留处理
- play 中 `policy.enable_corruption=False` 是否完全生效

## 5. MuJoCo / Stepit 侧当前观察

MuJoCo raycaster 插件配置中，Go2 的 `height_scan` 类似：

```xml
<plugin name="height_scan" plugin="mujoco.sensor.ray_caster"
        objtype="camera" objname="ray_caster">
  <config key="resolution" value="0.1"/>
  <config key="size" value="1.6 1.0"/>
  <config key="dis_range" value="0.1 3.0"/>
  <config key="type" value="yaw"/>
  <config key="sensor_data_types" value="pos_b"/>
</plugin>
```

关键点：

- `resolution=0.1`
- `size=1.6 1.0`
- 理论维度也是 `17 x 11 = 187`
- `type=yaw` 与 IsaacLab 的 `ray_alignment="yaw"` 思路接近
- `sensor_data_types=pos_b` 表示输出命中点在传感器/机体系下的位置，不是世界系绝对高度

MuJoCo raycaster 中 `pos_b.z` 很可能表示：

```text
pos_b.z ~= hit_z - sensor_z
```

如果这个判断成立，则要构造 IsaacLab 同语义输入，应使用：

```text
isaaclab_like_height = -pos_b.z - 0.5
```

而不是直接使用 `pos_b.z`，也不是对它做零均值。

## 6. Stepit 侧最需要排查的问题

1. Stepit 最终吃的是哪条高程图链路？

需要确认实际运行时到底是：

```text
MuJoCo raycaster Float32MultiArray -> policy
```

还是：

```text
MuJoCo pointcloud -> grid_map -> HeightmapSubscriber -> policy
```

这两条链路语义不同，不能混着比较。

2. 是否做了 `zero_mean`？

当前看到的 MuJoCo 发布器配置中有：

```yaml
raycaster_sensors:
  height_scan:
    output_format: "array"
    flatten_xyz: false
    zero_mean: true
    replace_nan: "zero"
    distance_type: ""
```

如果 `zero_mean: true` 生效，则平地会接近 `0`，这与 IsaacLab 平地 `-0.1485` 不一致。

Stepit 的 `heightmap.yml` 模板里也有：

```yaml
elevation_zero_mean: true
```

如果这条链路生效，也会破坏 IsaacLab 的绝对偏置语义。

3. 维度是否真的是 187？

IsaacLab 当前是：

```text
17 x 11 = 187
```

但 Stepit 模板中看到过：

```yaml
dimension: [17, 13]
```

这会得到：

```text
17 x 13 = 221
```

需要核对实际运行时使用的 `heightmap.yml`，不能只看 template。

4. 排列顺序是否一致？

IsaacLab `GridPatternCfg(size=[1.6, 1.0], resolution=0.1, ordering="xy")` 的默认含义是：

```text
x 从 -0.8 到 0.8
y 从 -0.5 到 0.5
flatten 后 x 快变，y 慢变
```

MuJoCo raycaster 默认 `_get_idx(v, h) = v * h_ray_num + h`，也是 row-major，通常也是一行内 x 快变。

仍需用实际打印确认是否存在：

- x/y 转置
- y 方向翻转
- 前后方向反了
- 左右方向反了

5. 符号是否一致？

IsaacLab 中：

```text
平地 ~= -0.15
更高台阶 ~= -0.53
```

因此 Stepit 侧最终输入也应该满足：

```text
平地接近 -0.15
台阶区域比平地更负
```

如果 Stepit 侧看到的是：

```text
平地 ~= 0
台阶为正
```

那就和 IsaacLab 训练输入不一致。

## 7. 建议的最小对齐实验

在 IsaacLab 和 Stepit/MuJoCo 里都固定机器人静止姿态，分别测三种场景：

1. 平地
2. 前方单级台阶
3. 左右不对称障碍

每次都打印并保存：

```text
原始 sensor/radar world z
原始 hit/world terrain z
原始 pos_b.z
最终送进 policy 的 187 维
min/max/mean/std
reshape(11, 17) 后的矩阵或 heatmap
```

判断标准：

```text
IsaacLab raw height_scan ~= Stepit final policy height input
```

至少要同时满足：

- 维度一致：187
- 单位一致：米
- 平地均值一致：约 `-0.15`
- 台阶符号一致：台阶更负
- 空间排列一致：前方台阶出现在相同矩阵区域
- 没有额外 zero mean
- 没有额外 abs/euclidean distance 语义变化

## 8. 当前推荐的 Stepit 对齐方向

如果目标是复现 IsaacLab 训练策略的输入语义，优先考虑让 Stepit/MuJoCo 最终策略输入显式变成：

```text
height[i] = -pos_b.z[i] - 0.5
```

并关闭：

```text
zero_mean
elevation_zero_mean
不必要的 abs/euclidean distance 转换
```

如果走 grid_map 链路，则要额外确认 grid_map 的 `elevation` 是否是世界系绝对高度，并在送进策略前构造：

```text
radar_ref_z_world - elevation[i] - 0.5
```

否则 grid_map 采样出来的只是地形绝对高度或零均值局部高度图，不等价于 IsaacLab 的 `height_scan`。

## 9. 优先排查清单

按优先级建议：

1. 找到 Stepit 实际运行时使用的 `heightmap.yml`，确认 `dimension`、`grid_size`、`x_major`、`elevation_zero_mean`。
2. 找到 Stepit 最终送进策略的 heightmap 输入来源，是 ray array 还是 grid_map subscriber。
3. 在 MuJoCo 中打印 raw `pos_b.z` 平地值，确认是否约为 `-0.35`。
4. 计算并打印 `-pos_b.z - 0.5`，看平地是否约为 `-0.15`。
5. 关闭所有 zero mean 后，再比较 IsaacLab 与 Stepit 的 `min/max/mean/std`。
6. 将 187 维 reshape 为 `11 x 17`，对比台阶区域位置和符号。

