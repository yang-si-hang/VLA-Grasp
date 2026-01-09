"""
检查 LeRobot 数据集中的元数据维度与实际读取数据维度是否一致
Date: 2026-1-9
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch

# 替换为你的数据集路径或 repo_id
repo_id = "202601061638-Aboslute" 
root_path = "data/grasp_soda/202601061638-Aboslute" # 如果是本地数据

# 1. 加载数据集
dataset = LeRobotDataset(repo_id=repo_id, root=root_path)

# 2. 检查元数据中定义的维度
print(f"--- 元数据检查 ---")
state_feature = dataset.meta.features.get("observation.state")
if state_feature:
    print(f"info.json 中定义的 shape: {state_feature['shape']}")
    if "names" in state_feature:
        print(f"定义的维度名称个数: {len(state_feature['names'])}")
        print(f"名称列表: {state_feature['names']}")

# 3. 检查实际读取到的数据维度
print(f"\n--- 实际数据检查 ---")
first_frame = dataset[0] # 读取第一帧
actual_state = first_frame["observation.state"]

print(f"读取到的 state 类型: {type(actual_state)}")
print(f"读取到的 state shape: {actual_state.shape}")

# 4. 自动比对
expected_dim = state_feature['shape'][0]
actual_dim = actual_state.shape[0]

if expected_dim != actual_dim:
    print(f"\n[错误] 维度不匹配！预期 {expected_dim}, 实际得到 {actual_dim}")
else:
    print(f"\n[正常] 维度一致。")