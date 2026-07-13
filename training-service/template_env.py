"""模板 → env cfg 生成器(阶段 1+2)。

思路:继承一个已注册 base_task 的 env cfg,按模板对它做"增删改":
  - scene.assets:换机器人/物体 USD、加障碍/物体(阶段 1)
  - rewards / events:改权重/参数,或从 mdp 目录加新项(阶段 2)

只改"值"用 Hydra 就够;这里处理 Hydra 做不到的"增/删结构项"。
本模块 import isaaclab,只能在容器内 isaac python 里跑。
"""
from __future__ import annotations

from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg  # noqa: F401  (供 params 里引用)

# 内置资产库:friendly ref → USD 路径(可持续扩充)
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

BUILTIN_USD = {
    "dex_cube": f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    "seattle_lab_table": f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
}
# 上传资产在 Job 内的落地目录(从 TOS 下载到这)
UPLOAD_DIR = "/assets"

# 图元 shape → spawn cfg 类
PRIMITIVES = {
    "box": sim_utils.CuboidCfg,
    "sphere": sim_utils.SphereCfg,
    "cylinder": sim_utils.CylinderCfg,
    "capsule": sim_utils.CapsuleCfg,
    "cone": sim_utils.ConeCfg,
}


def _spawn_from_spec(spec: dict[str, Any], is_rigid: bool):
    """把资产的 spawn 规格 → isaaclab spawn cfg。支持 builtin / upload(USD)/ primitive(图元)。

    is_rigid 决定是否附刚体:
      - True(kind=rigid,会动的物体):加 RigidBodyProperties + Mass + Collision
      - False(kind=static,固定障碍/桌子):只加 Collision(静态碰撞体)
    """
    sp = spec.get("spawn", {})
    src = sp.get("source", "builtin")
    phys = spec.get("physics", {})
    if phys.get("from_usd"):
        # USD 自带完整物理(inspect_usd 判定 self_describing)→ 不覆盖,用 USD 自己的
        rigid = mass = collision = None
    else:
        # 几何-only USD / 图元 → 补默认物理(质量/碰撞/刚体),用户无需填惯量等天书
        rigid = sim_utils.RigidBodyPropertiesCfg() if is_rigid else None
        mass = sim_utils.MassPropertiesCfg(mass=phys.get("mass", 0.1)) if is_rigid else None
        collision = sim_utils.CollisionPropertiesCfg()

    if src in ("builtin", "upload"):
        usd_path = BUILTIN_USD[sp["ref"]] if src == "builtin" else f"{UPLOAD_DIR}/{sp['ref']}"
        scale = tuple(sp.get("scale", (1.0, 1.0, 1.0)))
        return sim_utils.UsdFileCfg(
            usd_path=usd_path, scale=scale,
            rigid_props=rigid, mass_props=mass, collision_props=collision,
        )
    if src == "primitive":
        cls = PRIMITIVES[sp["shape"]]
        kw = {"rigid_props": rigid, "mass_props": mass, "collision_props": collision,
              "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(sp.get("color", (0.6, 0.6, 0.6))))}
        if sp["shape"] == "box":
            kw["size"] = tuple(sp.get("size", (0.05, 0.05, 0.05)))
        else:
            kw["radius"] = sp.get("radius", 0.03)
            if sp["shape"] in ("cylinder", "capsule", "cone"):
                kw["height"] = sp.get("height", 0.06)
        return cls(**kw)
    raise ValueError(f"未知 spawn source: {src}")


def _build_asset(spec: dict[str, Any]):
    """资产规格 → isaaclab 资产 cfg。kind: rigid(会动)| static(固定,只碰撞)。"""
    is_rigid = spec["kind"] == "rigid"
    spawn = _spawn_from_spec(spec, is_rigid)
    prim = f"{{ENV_REGEX_NS}}/{spec['name'].capitalize()}"
    init = None
    if "pos" in spec or "rot" in spec:
        init_kw = {}
        if "pos" in spec:
            init_kw["pos"] = tuple(spec["pos"])
        if "rot" in spec:
            init_kw["rot"] = tuple(spec["rot"])
        init = (RigidObjectCfg if spec["kind"] == "rigid" else AssetBaseCfg).InitialStateCfg(**init_kw)
    Cfg = RigidObjectCfg if spec["kind"] == "rigid" else AssetBaseCfg
    kw = {"prim_path": prim, "spawn": spawn}
    if init is not None:
        kw["init_state"] = init
    return Cfg(**kw)


def _fix_entity_refs(env_cfg, asset_name: str) -> list[str]:
    """替换某个场景实体后,把 events/rewards/terminations/curriculum 里引用它、且写死了 body_names 的
    SceneEntityCfg 改成 body_names=None(用 root)。避免新 USD 内部 body 命名不同导致解析失败。"""
    from isaaclab.managers import SceneEntityCfg as _SEC

    fixed = []
    for mgr_name in ("events", "rewards", "terminations", "curriculum"):
        mgr = getattr(env_cfg, mgr_name, None)
        if mgr is None:
            continue
        for term_name, term in vars(mgr).items():
            params = getattr(term, "params", None)
            if not isinstance(params, dict):
                continue
            for v in params.values():
                if isinstance(v, _SEC) and v.name == asset_name and v.body_names:
                    v.body_names = None
                    fixed.append(f"{mgr_name}.{term_name}")
    return fixed


def apply_scene(env_cfg, scene_spec: dict[str, Any]) -> list[str]:
    """按模板改场景:增/换 rigid|static 资产。返回改动说明。"""
    notes = []
    if "num_envs" in scene_spec:
        env_cfg.scene.num_envs = scene_spec["num_envs"]
    if "env_spacing" in scene_spec:
        env_cfg.scene.env_spacing = scene_spec["env_spacing"]
    for spec in scene_spec.get("assets", []):
        name = spec["name"]
        if spec["kind"] in ("rigid", "static"):
            replacing = hasattr(env_cfg.scene, name) and getattr(env_cfg.scene, name) is not None
            setattr(env_cfg.scene, name, _build_asset(spec))
            notes.append(f"scene.{name} ← {spec['kind']} ({spec.get('spawn', {}).get('source')})")
            if replacing:  # 换掉已有实体 → 修掉引用它的 body_names
                refs = _fix_entity_refs(env_cfg, name)
                if refs:
                    notes.append(f"  修正引用 {name} 的 body_names: {', '.join(refs)}")
    return notes


def apply_rewards(env_cfg, rewards_spec: list[dict], mdp) -> list[str]:
    """改/加 reward 项。存在则改 weight/params;不存在则用 mdp 目录里的 func 新建。"""
    notes = []
    for r in rewards_spec:
        name = r["name"]
        term = getattr(env_cfg.rewards, name, None)
        if term is not None:
            if "weight" in r:
                term.weight = r["weight"]
            if "params" in r:
                term.params.update(r["params"])
            notes.append(f"rewards.{name} weight={term.weight}")
        else:
            func = getattr(mdp, r["func"])
            setattr(env_cfg.rewards, name, RewTerm(func=func, weight=r["weight"], params=r.get("params", {})))
            notes.append(f"rewards.{name} += {r['func']} w={r['weight']}")
    return notes


def apply_events(env_cfg, events_spec: list[dict], mdp) -> list[str]:
    notes = []
    for e in events_spec:
        name = e["name"]
        term = getattr(env_cfg.events, name, None)
        if term is not None and "params" in e:
            term.params.update(e["params"])
            notes.append(f"events.{name} params updated")
        elif term is None:
            func = getattr(mdp, e["func"])
            setattr(env_cfg.events, name, EventTerm(func=func, mode=e.get("mode", "reset"), params=e.get("params", {})))
            notes.append(f"events.{name} += {e['func']}")
    return notes


def apply_template(env_cfg, template: dict[str, Any], mdp) -> list[str]:
    """把模板应用到 base env_cfg(原地修改)。mdp = 该任务的 mdp 模块(含 reward/event 函数)。"""
    notes = []
    if "scene" in template:
        notes += apply_scene(env_cfg, template["scene"])
    if "rewards" in template:
        notes += apply_rewards(env_cfg, template["rewards"], mdp)
    if "events" in template:
        notes += apply_events(env_cfg, template["events"], mdp)
    if "env" in template and "episode_length_s" in template["env"]:
        env_cfg.episode_length_s = template["env"]["episode_length_s"]
    return notes
