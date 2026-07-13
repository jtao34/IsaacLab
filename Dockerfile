FROM iaas-us-cn-beijing.cr.volces.com/physicalai/isaaclab:2.3.2

COPY . /workspace/isaaclab/

RUN git config --system url."https://ghproxy.cc/github.com/".insteadOf "https://github.com/" \
      && git config --system http.sslVerify false

RUN cd /workspace/isaaclab && ./isaaclab.sh --install

RUN cd /workspace/isaaclab/soarm101 && \
      /workspace/isaaclab/isaaclab.sh -p -m pip install -e . --no-deps

# TOS python SDK：训练 Job 训完用 training-service/upload_tos.py 把 checkpoint 传到火山 TOS
RUN /workspace/isaaclab/isaaclab.sh -p -m pip install tos

# 独立 usd-core（standalone pxr）：给 inspect_usd.py 检测上传 USD 用，装独立目录不污染训练环境
RUN /workspace/isaaclab/isaaclab.sh -p -m pip install --target /opt/usd-core usd-core
