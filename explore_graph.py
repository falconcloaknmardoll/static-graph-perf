import numpy as np

# Point to the ResNet50 file
file_path = "data/predict-ai-model-runtime/npz_all/npz/layout/xla/random/train/resnet50.4x4.bf16.npz"

print(f"Loading {file_path}...\n")
graph_data = np.load(file_path)

print("--- 1. The Graph Topology (The Code) ---")
print(f"Node Features (node_feat): {graph_data['node_feat'].shape}")
print(f"Node Opcodes (node_opcode): {graph_data['node_opcode'].shape}")
print(f"Edges (edge_index): {graph_data['edge_index'].shape}")

print("\n--- 2. The Compiler Configurations (The Search Space) ---")
print(f"Configurable Nodes (node_config_ids): {graph_data['node_config_ids'].shape}")
print(f"Config Features (node_config_feat): {graph_data['node_config_feat'].shape}")

print("\n--- 3. The Ground Truth (What we want to predict) ---")
print(f"Target Runtimes (config_runtime): {graph_data['config_runtime'].shape}")