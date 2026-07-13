"""渲染场景预览:把(套了模板的)任务场景渲染成一张 PNG,让用户在 UI 里看到真实几何。

复用 template_env 建场景(所见即所训:预览里的障碍/换的物体 = 训练时一样),
再照相机 tutorial 放一个相机截图。用 isaac python 跑。

用法:
  python render_preview.py --task <base_task> [--template t.yaml] --out preview.png
"""
from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser(description="渲染任务场景预览")
parser.add_argument("--task", type=str, required=True, help="base 任务 id")
parser.add_argument("--template", type=str, default=None, help="场景/reward 模板 YAML")
parser.add_argument("--out", type=str, default="/tmp/preview.png", help="输出 PNG 路径")
parser.add_argument("--eye", type=float, nargs=3, default=[0.9, 0.9, 0.7], help="相机位置")
parser.add_argument("--target", type=float, nargs=3, default=[0.2, 0.0, 0.15], help="相机看向")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
args_cli, _ = parser.parse_known_args()

# 取出参数后清空 sys.argv,避免 AppLauncher 误解析
_out, _task, _tmpl = args_cli.out, args_cli.task, args_cli.template
_eye, _target, _w, _h = args_cli.eye, args_cli.target, args_cli.width, args_cli.height
sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, enable_cameras=True).app  # 渲染必须 enable_cameras

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import isaaclab.envs.mdp as core_mdp  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402  注册基础任务
import isaac_so_arm101.tasks  # noqa: F401,E402  注册 SO-ARM 任务
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, "/workspace/isaaclab/training-service")
import template_env  # noqa: E402


def _disable_debug_vis(env_cfg):
    """关掉目标位姿/坐标系等调试标记(彩色箭头),给客户一张干净的场景图。"""
    if getattr(env_cfg, "commands", None) is not None:
        for term in vars(env_cfg.commands).values():
            if hasattr(term, "debug_vis"):
                term.debug_vis = False
    for asset in vars(env_cfg.scene).values():
        if hasattr(asset, "debug_vis"):
            asset.debug_vis = False


def main() -> int:
    env_cfg = parse_env_cfg(_task, num_envs=1)

    # 套模板(场景/reward)—— 和训练用同一个生成器
    if _tmpl:
        with open(_tmpl) as f:
            tmpl = yaml.safe_load(f)
        notes = template_env.apply_template(env_cfg, tmpl, core_mdp)
        print("[preview] 模板已应用:", notes, flush=True)

    _disable_debug_vis(env_cfg)  # 去掉调试箭头

    # 加一个预览相机到场景
    env_cfg.scene.preview_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/preview_cam",
        height=_h, width=_w, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=tuple(_eye), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )

    import gymnasium as gym
    env = gym.make(_task, cfg=env_cfg)
    env.reset()

    cam = env.unwrapped.scene["preview_cam"]
    cam.set_world_poses_from_view(
        eyes=torch.tensor([_eye], device=env.unwrapped.device),
        targets=torch.tensor([_target], device=env.unwrapped.device),
    )
    # 多步几帧让渲染稳定
    for _ in range(6):
        env.unwrapped.sim.step()
        cam.update(dt=env.unwrapped.sim.get_physics_dt())

    rgb = cam.data.output["rgb"][0].cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    rgb = rgb.astype(np.uint8)

    from PIL import Image
    Image.fromarray(rgb).save(_out)
    print(f"[preview] 已渲染 -> {_out}  ({rgb.shape[1]}x{rgb.shape[0]})", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        app.close()
    raise SystemExit(rc)
