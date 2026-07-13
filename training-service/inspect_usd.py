"""检查上传的 USD 是否含足够物理信息,给用户可读提示。

用独立的 usd-core(standalone pxr),不启动 Isaac Sim app —— 快(~1s)、可靠。
usd-core 装在独立目录 /opt/usd-core(见 Dockerfile),只给本脚本用,不污染训练环境。
API 在用户上传后调它,把 messages 回显给前端。

用法: python inspect_usd.py <usd_path> [out.json]
  报告以 "__USD_REPORT__ <json>" 打印一行,并可选写入 out.json。
"""
from __future__ import annotations

import json
import sys

# 独立 usd-core,避免和训练时 Kit 的 pxr 冲突(仅本进程用)
sys.path.insert(0, "/opt/usd-core")


def inspect(path: str) -> dict:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(path)
    if not stage:
        return {"ok": False, "self_describing": False,
                "messages": ["无法打开 USD:文件可能损坏,或不是有效的 .usd。"]}

    has_rigid = has_collision = has_mass = False
    mass = None
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            has_rigid = True
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            has_collision = True
        if prim.HasAPI(UsdPhysics.MassAPI):
            has_mass = True
            m = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            if m:
                mass = float(m)

    try:
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        rng = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
        size = rng.GetSize() if not rng.IsEmpty() else Gf.Vec3d(0, 0, 0)
    except Exception:
        size = None
    size_m = [round(float(size[i]), 3) for i in range(3)] if size else [0, 0, 0]
    max_dim = max(size_m) if size_m else 0

    messages = []
    if not has_rigid:
        messages.append("USD 未定义刚体(RigidBody)—— 将自动添加,使物体可被抓取/移动。")
    if not has_collision:
        messages.append("USD 未包含碰撞体(Collision)—— 将用凸包近似,复杂形状抓取精度可能一般。")
    if not has_mass:
        messages.append("USD 未指定质量(Mass)—— 将按默认密度估算,可在预览里调整。")
    if max_dim == 0:
        messages.append("USD 里没找到可见几何,请检查文件。")
    elif max_dim > 1.0:
        messages.append(f"物体尺寸约 {max_dim} m,偏大 —— 单位可能是 mm/cm,建议设置缩放(scale)。")
    elif max_dim < 0.005:
        messages.append(f"物体尺寸约 {max_dim} m,偏小 —— 单位可能不对,建议设置缩放。")

    self_describing = has_rigid and has_collision and has_mass
    return {
        "ok": max_dim > 0,
        "self_describing": self_describing,   # True=USD 自带完整物理,可直接用;False=需补默认
        "has_rigid": has_rigid, "has_collision": has_collision, "has_mass": has_mass,
        "mass": mass, "size_m": size_m,
        "messages": messages or ["USD 信息完整,可直接使用。"],
    }


def main() -> int:
    usd_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        report = inspect(usd_path)
    except Exception as e:  # noqa: BLE001
        report = {"ok": False, "self_describing": False, "messages": [f"检测失败:{type(e).__name__}: {e}"]}
    s = json.dumps(report, ensure_ascii=False)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(s)
    print("__USD_REPORT__ " + s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
