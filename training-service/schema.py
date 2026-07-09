"""训练请求的 schema 定义 + 校验。

设计:两层
  - 通用底座(所有任务一致):model / task / training_budget / sim2real_robustness /
    record_video / seed / output_name / behavior
  - 任务专属段(由 profiles/<task>.yaml 描述):goal_zones(空间范围)

校验只依赖标准库 + PyYAML,可独立运行(前端后台调用它先校验再交给 launcher)。
"""
from __future__ import annotations

import os
from typing import Any

import yaml

# --- 通用预设:训练时长档 → 规模(A30 上标定后可调) ---
BUDGET_PRESETS: dict[str, dict[str, int]] = {
    "quick": {"num_envs": 256, "max_iterations": 200},       # ~几分钟,冒烟
    "standard": {"num_envs": 4096, "max_iterations": 1000},  # ~1 小时,可用
    "thorough": {"num_envs": 8192, "max_iterations": 3000},  # ~数小时,高质量
}

VALID_MODELS = ("so100", "so101")
PROFILES_DIR = os.path.join(os.path.dirname(__file__), "profiles")


class ValidationError(Exception):
    """校验失败,message 面向客户可读。"""


def available_tasks() -> list[str]:
    """当前支持的任务(= profiles 目录下的 yaml)。"""
    if not os.path.isdir(PROFILES_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(PROFILES_DIR) if f.endswith(".yaml"))


def load_profile(task: str) -> dict[str, Any]:
    path = os.path.join(PROFILES_DIR, f"{task}.yaml")
    if not os.path.isfile(path):
        raise ValidationError(
            f"未知任务 '{task}'。当前支持:{', '.join(available_tasks()) or '(无)'}"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def _check_range(name: str, val: Any, limits: list[float]) -> list[str]:
    """校验一个 [lo, hi] 区间:格式、lo<=hi、落在 limits 内。"""
    errs: list[str] = []
    if not (isinstance(val, (list, tuple)) and len(val) == 2):
        return [f"{name}: 需要形如 [下限, 上限] 的两个数,得到 {val!r}"]
    lo, hi = val
    if not all(isinstance(x, (int, float)) for x in (lo, hi)):
        return [f"{name}: 上下限必须是数字"]
    if lo > hi:
        errs.append(f"{name}: 下限 {lo} 不能大于上限 {hi}")
    lmin, lmax = limits
    if lo < lmin or hi > lmax:
        errs.append(f"{name}: 范围 [{lo}, {hi}] 超出允许区间 [{lmin}, {lmax}]")
    return errs


def validate(request: dict[str, Any], profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """校验一个训练请求。失败抛 ValidationError(拼所有错误);成功返回补齐默认值的请求。"""
    req = dict(request)  # 浅拷贝,不改原对象
    errors: list[str] = []

    # --- 通用底座 ---
    task = req.get("task")
    if not task:
        raise ValidationError("缺少 'task'")
    if profile is None:
        profile = load_profile(task)

    model = req.get("model")
    if model not in VALID_MODELS:
        errors.append(f"model 必须是 {VALID_MODELS} 之一,得到 {model!r}")
    elif model not in profile["task_ids"]:
        errors.append(f"任务 '{task}' 不支持型号 '{model}'(支持:{list(profile['task_ids'])})")

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

    req.setdefault("output_name", f"{model}_{task}")

    # --- 任务专属段:goal_zones ---
    zones_spec = profile.get("goal_zones", {})
    goal = req.setdefault("goal", {})
    if not isinstance(goal, dict):
        raise ValidationError("goal 必须是对象")
    for zone_name, zone_val in goal.items():
        if zone_name not in zones_spec:
            errors.append(f"未知区域 '{zone_name}'(该任务支持:{list(zones_spec)})")
            continue
        for axis, axis_val in zone_val.items():
            axis_spec = zones_spec[zone_name].get(axis)
            if axis_spec is None:
                errors.append(f"{zone_name}.{axis}: 该区域无此轴")
                continue
            errors += _check_range(f"{zone_name}.{axis}", axis_val, axis_spec["limits"])

    if errors:
        raise ValidationError("配置校验失败:\n  - " + "\n  - ".join(errors))
    return req
