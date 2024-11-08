export TRITON_CACHE_DIR=./local_triton_cache
export CUDA_VISIBLE_DEVICES=3,4  # 使用指定的 GPU 设备
export HF_HOME="./cache"  # 指定缓存目录

# 启动 DeepSpeed 训练
deepspeed train.py
