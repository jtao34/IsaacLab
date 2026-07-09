# training-service

配置驱动的 Isaac Lab 训练层。客户/前端只填一份 YAML(或表单),不碰命令行和 Python。
**任务无关**:launcher/schema 通用,每个任务一份 `profiles/<task>.yaml`,加任务不改代码。

## 结构

```
training-service/
├── schema.py            # 请求 schema + 校验(通用底座 + 任务专属段)
├── launcher.py          # 读请求 → 校验 → 拼 Hydra 命令 → 执行(支持 --dry-run)
├── profiles/            # 每任务一份:task id 映射 + 可调旋钮 + 预设 + DR 开关
│   ├── reach.yaml
│   └── lift.yaml
└── example_request.yaml # 请求示例(= 前端最终产物)
```

## 两层设计

- **通用底座**(所有任务一致):`model` / `task` / `training_budget`(quick/standard/thorough) /
  `sim2real_robustness` / `record_video` / `seed` / `output_name` / `behavior`
- **任务专属段**(profile 描述):`goal`(空间范围,如 reach 目标区域、lift 物体/目标区域)

## 用法

```bash
# 在镜像内(/workspace/isaaclab/training-service)
python launcher.py --config request.yaml            # 校验 + 起训练
python launcher.py --config request.yaml --dry-run  # 只打印命令
```

前端后台也可 `import launcher; launcher.build_command(request_dict)` 拿命令,或先
`import schema; schema.validate(req)` 做校验(失败抛 `ValidationError`,message 面向客户可读)。

## 旋钮 → Hydra 映射(免改代码的原理)

所有旋钮最终是对已存在配置值的 Hydra 覆盖:
- 目标范围 → `env.commands.*.ranges.*` / `env.events.reset_*.params.pose_range.*`
- 行为预设 → `env.rewards.*.weight`
- 训练规模 → `--num_envs` / `--max_iterations`(由 training_budget 映射)
- sim2real 关 → 把 DR 事件参数中和成 no-op(见各 profile 的 `dr_off_overrides`)

## sim2real 稳健性(物理域随机化)

DR 事件已内置进 soarm101 的 reach/lift `EventCfg`(电机增益、关节摩擦;lift 另含物体质量/摩擦),
参考 isaaclab `manipulation/deploy/reach`。`sim2real_robustness: true` 时默认生效,`false` 时
launcher 用 `dr_off_overrides` 中和。locomotion 等任务本身自带 DR。

## 加新任务

1. 该任务是 manager-based(reward/DR 才能调;direct 任务只支持通用底座)
2. 在 `profiles/` 加一份 `<task>.yaml`:填 `task_ids`、`goal_zones`、`behavior_presets`、
   (可选)`dr_off_overrides`
3. 若任务缺物理 DR,给其 `EventCfg` 补随机化事件(参考 deploy/reach 或 velocity)

## 训练时长档标定(TODO)

`schema.py:BUDGET_PRESETS` 的 num_envs/iters 目前是估计值,需在 A30 上标定实际耗时后校准
(让 quick/standard/thorough 对应"约 X 分钟"能对客户兑现)。
