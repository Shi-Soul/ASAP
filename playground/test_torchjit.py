import torch
import time

# 定义一个简单的算子（矩阵乘法）
def custom_matrix_multiplication(a, b):
    return  a*b + b

# 使用 @torch.jit.script 装饰器
@torch.jit.script
def jit_custom_matrix_multiplication(a, b):
    return  a*b + b

# 生成随机矩阵
matrix_size = 1024
a = torch.randn(matrix_size, matrix_size).cuda()
b = torch.randn(matrix_size, matrix_size).cuda()

# 重复测试次数
num_repeats = 1000

# 测试原始函数的性能
start_time = time.time()
torch.cuda.synchronize()  # 添加 CUDA 流同步
for _ in range(num_repeats):
    result = custom_matrix_multiplication(a, b)
torch.cuda.synchronize()  # 添加 CUDA 流同步
original_time = time.time() - start_time

# 测试 JIT 函数的性能
start_time = time.time()
torch.cuda.synchronize()  # 添加 CUDA 流同步
for _ in range(num_repeats):
    result = jit_custom_matrix_multiplication(a, b)
torch.cuda.synchronize()  # 添加 CUDA 流同步
jit_time = time.time() - start_time

# 计算加速比
speedup_ratio = original_time / jit_time

print(f"Original time: {original_time:.4f} seconds")
print(f"JIT time: {jit_time:.4f} seconds")
print(f"Speedup ratio: {speedup_ratio:.2f}x")