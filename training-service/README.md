# training-service

配置驱动的 Isaac Lab 训练层。客户/前端只填一份 YAML(或表单),不碰命令行和 Python。
**机器人/任务无关**:加机器人或任务只改 `profiles/`,不动 schema/launcher。

## 结构(两层,与前端控制台一致)

```
training-service/
├── schema.py            # 请求 schema + 校验
├── launcher.py          # 读请求 → 校验 → 拼 Hydra 命令 → 执行(支持 --dry-run)
├── profiles/
│   ├── robots.yaml      # 机器人目录: robot → 支持任务(task id) + DR 能力(true/false/builtin)
│   └── tasks/           # 任务类型 profile(与机器人无关)
│       ├── reach.yaml   #   goal_zones / behavior_presets / dr_off_overrides
│       ├── lift.yaml
│       └── velocity.yaml
└── example_request.yaml # 请求示例(= 前端最终产物)
```

- **robots.yaml**:回答"哪个机器人能做哪些任务、有没有域随机化"
- **tasks/<task>.yaml**:回答"这个任务有哪些可调旋钮(空间范围、行为预设、DR 中和)"
- 机器人和任务解耦:同一个 reach profile 被 SO-ARM / Franka / UR 共用,只是 task id 不同

## 当前支持

| 机器人 | 任务 | DR | 状态 |
|--------|------|----|------|
| SO-ARM100 / 101 | reach, lift | 可开关(已加) | ✅ 训练验证过 |
| Franka Panda | reach, lift | 无 | reach 验证过;lift 同结构 |
| UR10 | reach | 无 | 同 Franka reach 结构 |
| Anymal-C | velocity(运动) | 内置 | ✅ 验证过 |

## 用法

```bash
# 在镜像内(/workspace/isaaclab/training-service)
../isaaclab.sh -p launcher.py --config request.yaml            # 校验 + 起训练
../isaaclab.sh -p launcher.py --config request.yaml --dry-run  # 只打印命令
```

前端后台也可 `import launcher; launcher.build_command(req)` 拿命令,或 `import schema; schema.validate(req)`
先校验(失败抛 `ValidationError`,message 面向客户可读)。

## 完整示例:起一次训练

以"训一个 SO-ARM101 抓取策略,要精准、开真机稳健、标准时长"为例。

**第 1 步 — 拿到请求 YAML**(二选一):
- 从前端控制台选好参数,点「复制 YAML」;或
- 手写:

```yaml
# my_pick.yaml
robot: so101
task: lift
training_budget: standard      # quick | standard | thorough
behavior: precise              # balanced | precise | smooth
sim2real_robustness: true
seed: 42
output_name: pick_v1
goal:
  object_start_zone: {x: [-0.08, 0.08], y: [-0.15, 0.15]}
  target_zone:       {x: [0.15, 0.30], y: [-0.15, 0.15], z: [0.10, 0.25]}
```

**第 2 步 — 起训练**:

```bash
kubectl exec -it isaaclab -n default -- bash
cd /workspace/isaaclab/training-service
../isaaclab.sh -p launcher.py --config my_pick.yaml            # 真跑
# ../isaaclab.sh -p launcher.py --config my_pick.yaml --dry-run  # 先看命令
```

launcher 自动:校验(物体范围是否超工作空间)→ 查 robots.yaml + tasks/lift.yaml → 拼出完整命令
(task id `Isaac-SO-ARM101-Lift-Cube-v0`、`--num_envs 4096`、`env.rewards.object_goal_tracking_fine_grained.weight=10.0`、
各范围 override…)→ 执行。**用户全程不碰 task id / Hydra 路径 / reward 名。**

**第 3 步 — 产物**:checkpoint 落在 `logs/rsl_rl/lift/<时间戳>/model_*.pt`。

### 对比:有没有 training-service

| | 没有 | 有 |
|---|---|---|
| 用户要写 | 一长串带 task id + Hydra 路径 + reward 名的命令 | 一份人话 YAML |
| 出错 | 路径写错默默跑歪 | 超范围/不支持组合**当场拦下**并提示 |
| 换机器人 | 改一堆参数 | 改一行 `robot:` |

> 说明:当前是在 pod 内跑,训练跟随 exec 会话(前台)。接 API 后可从网页一键起 K8s Job(后台),见项目下一步。

## DR / sim2real 逻辑

- 机器人 `dr: true`(SO-ARM)→ sim2real 开关有效;关闭时 launcher 用任务 profile 的 `dr_off_overrides` 中和
- `dr: false`(Franka/UR)→ 该任务未接 DR,sim2real 开关被忽略(不吐 dr_off)
- `dr: builtin`(Anymal velocity)→ 任务自带 DR,始终开启

## 加机器人 / 任务

- **加机器人**:在 `robots.yaml` 加一段(name / group / dr / tasks)
- **加任务类型**:在 `tasks/` 加一份 profile(goal_zones / behavior_presets / dr_off_overrides);
  若该任务是 manager-based 且要 sim2real 开关,给对应 EventCfg 补 DR 事件(参考 deploy/reach / velocity)

## 训练时长档标定(TODO)

`schema.py:BUDGET_PRESETS` 的 num_envs/iters 目前是估值,需在 A30 上标定实际耗时后校准。
