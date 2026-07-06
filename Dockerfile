FROM iaas-us-cn-beijing.cr.volces.com/physicalai/isaaclab:2.3.2

COPY . /workspace/isaaclab/

RUN cd /workspace/isaaclab && ./isaaclab.sh --install
