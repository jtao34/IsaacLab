"""训练控制台后端 API。

职责:
  - GET  /api/catalog          → 机器人目录 + 任务 profile + 训练时长档(前端据此渲染,单一数据源)
  - POST /api/validate         → 校验请求 + 返回将执行的命令(= launcher dry-run)
  - POST /api/train            → 校验 → 建 ConfigMap + K8s Job(申请 GPU) → 返回 job 名
  - GET  /api/jobs/{name}      → Job 状态 + 最近日志
  - GET  /                     → 前端页面(同源伺服,避开跨站限制)

MVP 跑法:本地 uvicorn(用本地 kubeconfig 建 Job);之后可打成镜像放集群内(改用 in-cluster 配置)。
复用 training-service 的 schema/launcher 作为单一数据源。
"""
from __future__ import annotations

import os
import sys
import uuid

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

# 复用 training-service 的校验/翻译逻辑
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "training-service"))
import schema  # noqa: E402
import launcher  # noqa: E402

NAMESPACE = os.environ.get("K8S_NAMESPACE", "default")
# 训练镜像:CI 出新 tag 后改这个环境变量(默认先用当前已验证的镜像)
TRAIN_IMAGE = os.environ.get(
    "TRAIN_IMAGE",
    "iaas-us-cn-beijing.cr.volces.com/physicalai/isaaclab:abd9ed344b3e44108a62d0e4dfd6ff0d",
)
# 挂给 Job 的 TOS 凭证 Secret 名(envFrom);没有则训练照跑、只是不上传
TOS_SECRET = os.environ.get("TOS_SECRET", "tos-creds")

app = FastAPI(title="Robot Training Console API")


# ---------------- Kubernetes ----------------
def _k8s():
    from kubernetes import client, config

    try:
        config.load_incluster_config()  # 部署到集群内时
    except Exception:
        config.load_kube_config()  # 本地开发
    return client.BatchV1Api(), client.CoreV1Api(), client


def _job_manifest(k8s, name: str, req: dict) -> "object":
    """构造训练 Job:挂请求 ConfigMap、申请 1 GPU、跑 launcher 再传 TOS。"""
    prefix = req.get("output_name") or name
    cmd = (
        "set -e; cd /workspace/isaaclab/training-service; "
        "../isaaclab.sh -p launcher.py --config /config/request.yaml; "
        f"../isaaclab.sh -p upload_tos.py --prefix {prefix}"
    )
    container = k8s.V1Container(
        name="train",
        image=TRAIN_IMAGE,
        command=["/bin/bash", "-lc", cmd],
        resources=k8s.V1ResourceRequirements(limits={"nvidia.com/gpu": "1"}),
        volume_mounts=[
            k8s.V1VolumeMount(name="cfg", mount_path="/config"),
            k8s.V1VolumeMount(name="dshm", mount_path="/dev/shm"),
        ],
        # TOS 凭证:Secret 存在就注入;不存在训练也不受影响(upload 跳过)
        env_from=[k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name=TOS_SECRET, optional=True))],
    )
    pod_spec = k8s.V1PodSpec(
        restart_policy="Never",
        containers=[container],
        volumes=[
            k8s.V1Volume(name="cfg", config_map=k8s.V1ConfigMapVolumeSource(name=name)),
            k8s.V1Volume(name="dshm", empty_dir=k8s.V1EmptyDirVolumeSource(medium="Memory", size_limit="16Gi")),
        ],
    )
    return k8s.V1Job(
        metadata=k8s.V1ObjectMeta(name=name, labels={"app": "robot-training", "task": req["task"]}),
        spec=k8s.V1JobSpec(
            backoff_limit=0,
            ttl_seconds_after_finished=86400,
            template=k8s.V1PodTemplateSpec(
                metadata=k8s.V1ObjectMeta(labels={"app": "robot-training", "job": name}),
                spec=pod_spec,
            ),
        ),
    )


# ---------------- API ----------------
@app.get("/api/catalog")
def catalog():
    robots = schema.load_robots()
    task_types = {tk for r in robots.values() for tk in r["tasks"]}
    tasks = {t: schema.load_task(t) for t in task_types}
    return {"robots": robots, "tasks": tasks, "budgets": schema.BUDGET_PRESETS}


@app.post("/api/validate")
def validate(req: dict):
    try:
        cmd = launcher.build_command(req)
        return {"ok": True, "command": " ".join(cmd)}
    except schema.ValidationError as e:
        return {"ok": False, "errors": str(e)}


@app.post("/api/train")
def train(req: dict):
    try:
        r = schema.validate(req)
        launcher.build_command(r)  # 再确认能拼出命令
    except schema.ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    name = f"train-{r['task']}-{uuid.uuid4().hex[:6]}"
    try:
        batch, core, k8s = _k8s()
        # 请求写进 ConfigMap,Job 挂载
        core.create_namespaced_config_map(
            NAMESPACE,
            k8s.V1ConfigMap(metadata=k8s.V1ObjectMeta(name=name), data={"request.yaml": yaml.safe_dump(r, allow_unicode=True)}),
        )
        batch.create_namespaced_job(NAMESPACE, _job_manifest(k8s, name, r))
    except Exception as e:  # 集群连接/建 Job 失败 → 返回可读 JSON,而非 500 纯文本
        raise HTTPException(status_code=502, detail=f"起 Job 失败(多为集群连接不稳,请重试):{type(e).__name__}: {str(e)[:200]}")
    return {"job": name, "task_id": schema.load_robots()[r["robot"]]["tasks"][r["task"]]}


def _render_job_manifest(k8s, name: str, task_id: str, prefix: str):
    """渲染预览 Job:跑 render_preview 出图 → 传 TOS(供前端取;无凭证会跳过)。申请 1 GPU。"""
    cmd = (
        "set -e; cd /workspace/isaaclab/training-service; mkdir -p /tmp/preview_out; "
        f"../isaaclab.sh -p render_preview.py --task {task_id} --template /config/template.yaml --out /tmp/preview_out/preview.png; "
        f"../isaaclab.sh -p upload_tos.py --local-dir /tmp/preview_out --prefix {prefix} || true"
    )
    container = k8s.V1Container(
        name="render", image=TRAIN_IMAGE, command=["/bin/bash", "-lc", cmd],
        resources=k8s.V1ResourceRequirements(limits={"nvidia.com/gpu": "1"}),
        volume_mounts=[k8s.V1VolumeMount(name="cfg", mount_path="/config")],
        env_from=[k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name=TOS_SECRET, optional=True))],
    )
    pod_spec = k8s.V1PodSpec(
        restart_policy="Never", containers=[container],
        volumes=[k8s.V1Volume(name="cfg", config_map=k8s.V1ConfigMapVolumeSource(name=name))],
    )
    return k8s.V1Job(
        metadata=k8s.V1ObjectMeta(name=name, labels={"app": "robot-render"}),
        spec=k8s.V1JobSpec(
            backoff_limit=0, ttl_seconds_after_finished=3600,
            template=k8s.V1PodTemplateSpec(
                metadata=k8s.V1ObjectMeta(labels={"app": "robot-render", "job": name}), spec=pod_spec),
        ),
    )


@app.post("/api/render")
def render(req: dict):
    """渲染一张当前(带模板的)场景预览,让用户确认场景符合预期。"""
    try:
        r = schema.validate(req)
    except schema.ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    name = f"render-{r['task']}-{uuid.uuid4().hex[:6]}"
    task_id = schema.load_robots()[r["robot"]]["tasks"][r["task"]]
    template = r.get("template", {"scene": {"assets": []}})
    prefix = f"previews/{r.get('output_name') or name}"
    try:
        batch, core, k8s = _k8s()
        core.create_namespaced_config_map(
            NAMESPACE,
            k8s.V1ConfigMap(metadata=k8s.V1ObjectMeta(name=name),
                            data={"template.yaml": yaml.safe_dump(template, allow_unicode=True)}),
        )
        batch.create_namespaced_job(NAMESPACE, _render_job_manifest(k8s, name, task_id, prefix))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"起渲染 Job 失败(集群连接?):{type(e).__name__}: {str(e)[:200]}")
    # 前端渲染完从 TOS 取 prefix/preview.png(TOS 接好后)
    return {"job": name, "preview_key": f"{prefix}/preview.png"}


@app.get("/api/jobs/{name}")
def job_status(name: str):
    try:
        batch, core, _ = _k8s()
        job = batch.read_namespaced_job_status(name, NAMESPACE)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"查询 Job 失败(集群连接不稳,请重试):{type(e).__name__}")
    s = job.status
    phase = "运行中"
    if s.succeeded:
        phase = "已完成"
    elif s.failed:
        phase = "失败"
    elif not s.active:
        phase = "排队中"
    # 最近日志
    logs = ""
    try:
        pods = core.list_namespaced_pod(NAMESPACE, label_selector=f"job={name}")
        if pods.items:
            logs = core.read_namespaced_pod_log(pods.items[0].metadata.name, NAMESPACE, tail_lines=40)
    except Exception:
        pass
    return {"job": name, "phase": phase, "active": s.active, "succeeded": s.succeeded, "failed": s.failed, "logs": logs}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
        return f.read()
