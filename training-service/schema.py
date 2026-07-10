"""训练请求的 schema 定义 + 校验。

结构(与前端一致的两层):
  - 机器人目录  profiles/robots.yaml     : robot → 支持的任务(task id)+ DR 能力
  - 任务 profile profiles/tasks/<task>.yaml: goal_zones / behavior_presets / dr_off_overrides

请求两层:
  - 通用底座: robot / task / training_budget / sim2real_robustness / record_video / seed / output_name / behavior
  - 任务专属: goal(空间/速度范围)

只依赖标准库 + PyYAML,可独立运行(前端后台调用它先校验再交给 launcher)。
"""
from __future__ import annotations

import os
from typing import Any

import yaml

# 通用预设:训练时长档 → 规模(A30 上标定后可调)
BUDGET_PRESETS: dict[str, dict[str, int]] = {
    "quick": {"num_envs": 256, "max_iterations": 200},
    "standard": {"num_envs": 4096, "max_iterations": 1000},
    "thorough": {"num_envs": 8192, "max_iterations": 3000},
}

PROFILES_DIR = os.path.join(os.path.dirname(__file__), "profiles")
ROBOTS_PATH = os.path.join(PROFILES_DIR, "robots.yaml")
TASKS_DIR = os.path.join(PROFILES_DIR, "tasks")


class ValidationError(Exception):
    """校验失败,message 面向客户可读。"""


def load_robots() -> dict[str, Any]:
    with open(ROBOTS_PATH) as f:
        return yaml.safe_load(f)


def load_task(task: str) -> dict[str, Any]:
    path = os.path.join(TASKS_DIR, f"{task}.yaml")
    if not os.path.isfile(path):
        raise ValidationError(f"未知任务类型 '{task}'")
    with open(path) as f:
        return yaml.safe_load(f)


def robot_dr(robot: str, robots: dict | None = None) -> Any:
    """返回机器人的 DR 能力:True / False / 'builtin'。"""
    robots = robots or load_robots()
    return robots.get(robot, {}).get("dr", False)


def _check_range(name: str, val: Any, limits: list[float]) -> list[str]:
    if not (isinstance(val, (list, tuple)) and len(val) == 2):
        return [f"{name}: 需要形如 [下限, 上限] 的两个数,得到 {val!r}"]
    lo, hi = val
    if not all(isinstance(x, (int, float)) for x in (lo, hi)):
        return [f"{name}: 上下限必须是数字"]
    errs = []
    if lo > hi:
        errs.append(f"{name}: 下限 {lo} 不能大于上限 {hi}")
    lmin, lmax = limits
    if lo < lmin or hi > lmax:
        errs.append(f"{name}: 范围 [{lo}, {hi}] 超出可行域 [{lmin}, {lmax}]")
    return errs


def validate(request: dict[str, Any]) -> dict[str, Any]:
    """校验训练请求。失败抛 ValidationError(拼所有错误);成功返回补齐默认值的请求。"""
    req = dict(request)
    errors: list[str] = []
    robots = load_robots()

    # --- 机器人 & 任务 ---
    robot = req.get("robot")
    if robot not in robots:
        raise ValidationError(f"未知机器人 '{robot}'。支持:{', '.join(robots)}")
    rob = robots[robot]

    task = req.get("task")
    if task not in rob["tasks"]:
        raise ValidationError(
            f"机器人 '{robot}' 不支持任务 '{task}'(它支持:{', '.join(rob['tasks'])})"
        )
    profile = load_task(task)

    # --- 通用底座 ---
    budget = req.setdefault("training_budget", "standard")
    if budget not in BUDGET_PRESETS:
        errors.append(f"training_budget 必须是 {list(BUDGET_PRESETS)} 之一,得到 {budget!r}")

    behavior = req.setdefault("behavior", "balanced")
    presets = profile.get("behavior_presets", {})
    if behavior not in presets:
        errors.append(f"behavior 必须是 {list(presets)} 之一,得到 {behavior!r}")

    req.setdefault("sim2real_robustness", True)
    if not isinstance(req["sim2real_robustness"], bool):
        errors.append("sim2real_robustness 必须是 true/false")

    req.setdefault("record_video", False)
    req.setdefault("seed", 42)
    if not isinstance(req["seed"], int):
        errors.append("seed 必须是整数")
    req.setdefault("output_name", f"{robot}_{task}")

    # --- 任务专属:goal_zones ---
    zones_spec = profile.get("goal_zones", {})
    goal = req.setdefault("goal", {})
    if not isinstance(goal, dict):
        raise ValidationError("goal 必须是对象")
    for zone_name, zone_val in goal.items():
        if zone_name not in zones_spec:
            errors.append(f"未知区域 '{zone_name}'(该任务支持:{list(zones_spec)})")
            continue
        axes = zones_spec[zone_name]["axes"]
        for axis, axis_val in zone_val.items():
            if axis not in axes:
                errors.append(f"{zone_name}.{axis}: 该区域无此轴")
                continue
            label = axes[axis].get("name", axis)
            errs = _check_range(f"{zones_spec[zone_name]['label']} · {label}", axis_val, axes[axis]["limits"])
            errors += errs

    if errors:
        raise ValidationError("配置校验失败:\n  - " + "\n  - ".join(errors))
    return req
