# Custom kind node image that requests the GPU. With Docker's default runtime set to
# nvidia, these env vars make the nvidia-container-runtime inject /dev/dxg + the WSL
# driver libs (libcuda, libnvidia-ptxjitcompiler, ...) into the kind node container at
# creation; exactly what `docker run --gpus all` does. The pods then bind-mount those
# from the node (WSL-native GPU, no device plugin).
FROM kindest/node:v1.32.0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
