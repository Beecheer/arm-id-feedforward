# 采集数据与参数文件

动态采集 CSV 的最低要求是 `device_time_us`、`sample_sequence`、`status`、
`q1_rad`～`q7_rad`、`tau1_nm`～`tau7_nm`。列顺序不限；反馈位置和力矩必须
使用 README 中相同的关节坐标与单位。伴随的同名 JSON 可以记录 `urdf_sha256`
和 `transition_duration_s`；存在 URDF 哈希时，辨识器会验证它与当前模型一致。

辨识预处理会在时间戳倒退或较长反馈缺口处切段，分别滤波并对实测位置求导。
同一文件内的目标位置不参与回归。质量提示与辨识结果分别保存，质量提示本身
不会删除原始记录。

`ega-arm-identify-many` 接受至少两个独立训练 CSV。岭系数只通过训练文件之间的
整组留出选择，然后用全部训练记录拟合一套 91 维共享参数。若指定验证文件，
它只在参数冻结后评估。输出的 `shared_full_model.json` 可直接传给
`ega-arm-feedforward --model`。

仓库内的 `examples/measured_excerpt.csv` 是 12 秒真实反馈片段，可用于检查
解析与预处理。它不包含完整轨迹覆盖，不应当作完整参数辨识训练集。
