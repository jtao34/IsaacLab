"""通用训练 launcher:把客户填的训练请求(YAML)翻译成 Isaac Lab 训练命令并执行。

用法:
    python launcher.py --config request.yaml            # 校验 + 起训练
    python launcher.py --config request.yaml --dry-run  # 只打印将执行的命令

任务无关:所有任务专属知识都在 profiles/<task>.yaml,加新任务不改本文件。
前端后台也可直接 import build_command(request) 拿到命令。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any

import yaml

import schema

ISAACLAB_ROOT = os.environ.get("ISAACLAB_ROOT", "/workspace/isaaclab")
ISAACLAB_SH = os.path.join(ISAACLAB_ROOT, "isaaclab.sh")


def _fmt(value: Any) -> str:
    """把值格式化成 Hydra override 右值。列表 -> [a,b](无空格,避免 shell/hydra 解析问题)。"""
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(v) for v in value) + "]"
    return str(value)


def build_command(request: dict[str, Any]) -> list[str]:
    """校验请求并构造训练命令(argv 列表,不经 shell,免转义烦恼)。"""
    profile = schema.load_profile(request["task"])
    req = schema.validate(request, profile)

    task_id = profile["task_ids"][req["model"]]
    budget = schema.BUDGET_PRESETS[req["training_budget"]]
    train_script = os.path.join(ISAACLAB_ROOT, profile["train_script"])

    argv: list[str] = [
        ISAACLAB_SH, "-p", train_script,
        "--task", task_id,
        "--headless",
        "--num_envs", str(budget["num_envs"]),
        "--max_iterations", str(budget["max_iterations"]),
        "--seed", str(req["seed"]),
    ]
    if req.get("record_video"):
        argv += ["--video", "--video_length", "200", "--video_interval", "2000"]

    # --- Hydra overrides ---
    overrides: list[str] = []

    # 1) 任务专属:goal_zones -> commands/events 的范围
    zones_spec = profile.get("goal_zones", {})
    for zone_name, zone_val in req.get("goal", {}).items():
        for axis, axis_val in zone_val.items():
            path = zones_spec[zone_name][axis]["path"]
            overrides.append(f"{path}={_fmt(axis_val)}")

    # 2) 行为预设 -> reward 权重
    for path, val in profile.get("behavior_presets", {}).get(req["behavior"], {}).items():
        overrides.append(f"{path}={_fmt(val)}")

    # 3) sim2real 关 -> 把 DR 项中和成 no-op
    if not req["sim2real_robustness"]:
        for path, val in profile.get("dr_off_overrides", {}).items():
            overrides.append(f"{path}={_fmt(val)}")

    # 4) 高级:专家自定义 override(原样透传)
    overrides += req.get("advanced", {}).get("overrides", [])

    return argv + overrides


def main() -> int:
    ap = argparse.ArgumentParser(description="Isaac Lab 训练 launcher")
    ap.add_argument("--config", required=True, help="训练请求 YAML")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令,不执行")
    args = ap.parse_args()

    with open(args.config) as f:
        request = yaml.safe_load(f)

    try:
        cmd = build_command(request)
    except schema.ValidationError as e:
        print(f"[校验失败] {e}", file=sys.stderr)
        return 2

    printable = " ".join(cmd)
    if args.dry_run:
        print(printable)
        return 0

    print(f"[launcher] 执行:\n{printable}\n", flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
