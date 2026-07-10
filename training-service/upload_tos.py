"""训练产物上传到火山 TOS。训练 Job 训完调它,把 checkpoint/日志传上去(替代无盘 pod 的持久化)。

凭证从环境变量读(由 Job 挂载的 K8s Secret 提供),不写死:
  TOS_ACCESS_KEY / TOS_SECRET_KEY / TOS_BUCKET
  TOS_ENDPOINT(默认北京内网)/ TOS_REGION(默认 cn-beijing)

用法:
  python upload_tos.py --prefix <run-name> [--local-dir /workspace/isaaclab/logs]
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="上传训练产物到 TOS")
    ap.add_argument("--local-dir", default="/workspace/isaaclab/logs", help="要上传的本地目录")
    ap.add_argument("--prefix", required=True, help="TOS 上的对象前缀(通常用 output_name / job 名)")
    args = ap.parse_args()

    try:
        ak = os.environ["TOS_ACCESS_KEY"]
        sk = os.environ["TOS_SECRET_KEY"]
        bucket = os.environ["TOS_BUCKET"]
    except KeyError as e:
        print(f"[tos] 缺少环境变量 {e};跳过上传(训练已完成,产物仍在容器内 {args.local_dir})", file=sys.stderr)
        return 0  # 不因缺凭证让整个 Job 失败

    endpoint = os.environ.get("TOS_ENDPOINT", "tos-cn-beijing.ivolces.com")
    region = os.environ.get("TOS_REGION", "cn-beijing")

    if not os.path.isdir(args.local_dir):
        print(f"[tos] 本地目录不存在,无产物可传:{args.local_dir}")
        return 0

    import tos  # 火山 TOS python SDK,已装进镜像

    client = tos.TosClientV2(ak, sk, endpoint, region)
    n, total = 0, 0
    for root, _, files in os.walk(args.local_dir):
        for f in files:
            fp = os.path.join(root, f)
            key = f"{args.prefix.strip('/')}/{os.path.relpath(fp, args.local_dir)}"
            client.put_object_from_file(bucket, key, fp)
            n += 1
            total += os.path.getsize(fp)
            print(f"[tos] {fp} -> tos://{bucket}/{key}", flush=True)

    print(f"[tos] 完成:{n} 个文件 / {total/1e6:.1f} MB -> tos://{bucket}/{args.prefix.strip('/')}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
