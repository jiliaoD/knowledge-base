# test/GPU验证.py

import torch

print('=== GPU 信息检测 ===')
print(f'GPU 可用：{torch.cuda.is_available()}')

if torch.cuda.is_available():
    # 获取显卡数量
    gpu_count = torch.cuda.device_count()
    print(f'显卡数量：{gpu_count}')

    # 遍历每块显卡的详细信息
    for i in range(gpu_count):
        print(f'\n--- 显卡 {i} ---')
        print(f'名称：{torch.cuda.get_device_name(i)}')
        print(f'计算能力：{torch.cuda.get_device_capability(i)}')

        # 获取显存信息（单位：GB）
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f'总显存：{total_memory:.2f} GB')

        # 当前显存使用情况
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        print(f'已分配显存：{allocated:.2f} GB')
        print(f'已预留显存：{reserved:.2f} GB')

        # CUDA 版本
        print(f'CUDA 版本：{torch.version.cuda}')
        print(f'cuDNN 版本：{torch.backends.cudnn.version()}')
else:
    print('未检测到可用的 GPU，将使用 CPU')

