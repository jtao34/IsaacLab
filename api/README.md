# api —— 训练控制台后端

让前端「开始训练」按钮真正在集群 GPU 上起训练(K8s Job),checkpoint 传 TOS。
复用 `../training-service` 的 schema/launcher 作为单一数据源。

## 端点

| 方法 | 路径 | 作用 |
|------|------|------|
| GET | `/api/catalog` | 机器人目录 + 任务 profile + 训练时长档(前端据此渲染) |
| POST | `/api/validate` | 校验请求 + 返回将执行的命令(dry-run) |
| POST | `/api/train` | 校验 → 建 ConfigMap + K8s Job(申请 GPU) → 返回 job 名 |
| GET | `/api/jobs/{name}` | Job 状态 + 最近日志 |
| GET | `/` | 前端页面 |

## MVP 跑法(本地)

```bash
cd api
python3 -m pip install -r requirements.txt
# 需要本地 kubeconfig 能访问集群(/train 才建得了 Job)
uvicorn server:app --port 8000
# 浏览器打开 http://localhost:8000
```

- `/api/catalog`、`/api/validate` 纯读,不碰集群,立即可用
- `/api/train` 用本地 kubeconfig 在集群建 Job

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `TRAIN_IMAGE` | 当前已验证镜像 tag | **CI 出带 tos SDK 的新镜像后,改成新 tag** |
| `K8S_NAMESPACE` | default | Job 建在哪个 namespace |
| `TOS_SECRET` | tos-creds | 挂给 Job 的 TOS 凭证 Secret 名(envFrom,optional) |

## TOS 凭证 Secret(checkpoint 上传前置)

用 IAM 子账号密钥建(别用主账号):

```bash
kubectl create secret generic tos-creds -n default \
  --from-literal=TOS_ACCESS_KEY=<你的AK> \
  --from-literal=TOS_SECRET_KEY=<你的SK> \
  --from-literal=TOS_BUCKET=isaaclab-ckpt-test \
  --from-literal=TOS_ENDPOINT=tos-cn-beijing.ivolces.com \
  --from-literal=TOS_REGION=cn-beijing
```

Secret 不存在也不影响训练:`upload_tos.py` 检测不到凭证会跳过上传(产物仍在容器内)。

## Job 干了什么

API 建的 Job(申请 1 GPU、挑空闲 A30):
1. 挂载请求 ConfigMap 到 `/config/request.yaml`
2. `launcher.py --config /config/request.yaml` → 训练
3. `upload_tos.py --prefix <output_name>` → 把 `logs/` 传到 TOS

## 上集群(之后)

打成镜像(python:slim + 本目录 + training-service)放集群内跑,`server.py` 已用
`load_incluster_config()` 优先;给它一个能建 Job/ConfigMap、读 Pod 日志的 ServiceAccount(RBAC)。
